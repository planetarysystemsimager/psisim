from psisim import spectrum
import numpy as np
import scipy.interpolate as si
import scipy.integrate as integrate
import astropy.units as u
from scipy.ndimage import gaussian_filter
import copy
import matplotlib.pyplot as plt
import warnings

def simulate_observation(telescope,instrument,planet_table_entry,planet_spectrum,wvs,spectrum_R,
    inject_noise=True,verbose=False,post_processing_gain = 1,return_noise_components=False,stellar_spec=None,
    apply_lsf=False,integrate_delta_wv=False,no_speckle_noise=False,plot=False, sky_on=True):
    '''
    A function that simulates an observation

    Inputs:
    Telescope     - A Telescope object
    Instrument     - An Instrument object
    planet_table_entry - an entry/row from a Universe planet table
    planet_spectrum - A planet spectrum from simulate spectrum given in contrast units
    observing_configs - To be defined

    Kwargs:
    stellar_spectrum - an optional argument to pass if the user has already generated a stellar spectrum. Expected units are photons/s/cm^2/angstrom
    post_processing_gain - optional argument by which to reduce the speckle noise (set to np.inf to replace "no_speckle_noise")
    sky_on - optional bool to apply atmospheric transmission (default True to apply transmission). Replaces old _nosky function
    
    Outputs: 
    F_lambda, F_lambda_error
    '''


    ##### ALL UNITS NEED TO BE PROPERLY EXAMINED #####

    #Some relevant planet properties
    separation = planet_table_entry['AngSep'].to(u.arcsec)

    star_aomag = planet_table_entry['StarAOmag']
    star_spt = planet_table_entry['StarSpT']

    if stellar_spec is None:
        #Get the stellar spectrum at the wavelengths of interest. 
        #The stellar spectrum will be in units of photons/s/cm^2/angstrom
        stellar_spectrum = spectrum.get_stellar_spectrum(planet_table_entry,wvs,instrument.current_R,
            verbose=verbose)
    else: 
        stellar_spectrum = copy.deepcopy(stellar_spec)
    #TODO: Add check that the input stellar spectrum as the correct units
    
    #Multiply the stellar spectrum by the collecting area and a factor of 10,000
    #to convert from m^2 to cm^2 and get the stellar spectrum in units of photons/s
    stellar_spectrum *= telescope.collecting_area.to(u.cm**2)

    #Multiply by atmospheric transmission if requested
    if sky_on:
        stellar_spectrum *= telescope.get_atmospheric_transmission(wvs,R=instrument.current_R)

    #Multiply by telescope throughput
    stellar_spectrum *= telescope.get_telescope_throughput(wvs,band=instrument.current_filter)

    #Multiply by filter transmission
    stellar_spectrum *= instrument.get_filter_transmission(wvs,instrument.current_filter)

    # Convert planet from contrast units back into spectrum units
      # Thus planet spectrum now has throughputs above from stellar spectrum accounted for
    #This assumes that you have properly carried around 'wvs' 
    #and that the planet_spectrum is given at the wvs wavelengths. 
    scaled_spectrum = planet_spectrum*stellar_spectrum

    # Account for instrument throughput for stellar spectrum and planet spectrum individually
    stellar_spectrum *= instrument.get_inst_throughput(wvs,planet_flag=False)
    scaled_spectrum  *= instrument.get_inst_throughput(wvs,planet_flag=True,planet_sep=separation)
    
    #Multiply by quantum efficiency
    stellar_spectrum *= instrument.qe #now units of e-/s/Angstrom
    scaled_spectrum *= instrument.qe #now units of e-/s/Angstrom

    #Get Sky thermal background in photons/s/Angstrom
    thermal_sky = telescope.get_sky_background(wvs,R=instrument.current_R) #Assumes diffraction limited PSF had to multiply by solid angle of PSF.
    thermal_sky *= telescope.collecting_area.to(u.cm**2) #Multiply by collecting area - units of photons/s/Angstom
    thermal_sky *= telescope.get_telescope_throughput(wvs,band=instrument.current_filter) #Multiply by telescope throughput
    thermal_sky *= instrument.get_inst_throughput(wvs) #Multiply by instrument throughput

    #Get Telescope thermal background in photons/s/Angstrom
    thermal_telescope = telescope.get_thermal_emission(wvs,band=instrument.current_filter)
    thermal_telescope *= telescope.collecting_area.to(u.cm**2) #Now in units of photons/s/Angstrom
    thermal_telescope *= instrument.get_inst_throughput(wvs) #Multiply by telescope throughput

    #Get the Instrument thermal background in photons/s/Angstrom
    diffraction_limit = (wvs/telescope.diameter.to(u.micron)*u.radian).to(u.arcsec)
    solidangle = diffraction_limit**2 * 1.13
    thermal_inst = instrument.get_instrument_background(wvs,solidangle) #In units of photons/s/Angstrom

    thermal_flux = thermal_sky + thermal_telescope + thermal_inst
    # thermal_flux *= instrument.get_filter_transmission(wvs, instrument.current_filter) # Comment this out for now. It should be part of get_instrument_background
    thermal_flux *= instrument.qe #e-/s/Angstrom

    #Apply the line-spread function if the user wants to. 
    if apply_lsf:
        dwvs = np.abs(wvs - np.roll(wvs, 1))
        dwvs[0] = dwvs[1]
        dwv_mean = np.mean(dwvs)
        lsf_fwhm = (instrument.lsf_width/dwv_mean).decompose() #Get the lsf_fwhm in units of current wavelength spacing
        lsf_sigma = lsf_fwhm/(2*np.sqrt(2*np.log(2))) #Convert to sigma

        stellar_spectrum = gaussian_filter(stellar_spectrum, lsf_sigma.value) * stellar_spectrum.unit
        scaled_spectrum = gaussian_filter(scaled_spectrum, lsf_sigma.value) * scaled_spectrum.unit

    #Downsample to instrument wavelength sampling
    intermediate_spectrum = si.interp1d(wvs, scaled_spectrum,fill_value="extrapolate",bounds_error=False)
    intermediate_stellar_spectrum = si.interp1d(wvs, stellar_spectrum,fill_value="extrapolate",bounds_error=False)
    intermediate_thermal_spectrum = si.interp1d(wvs, thermal_flux,fill_value="extrapolate",bounds_error=False)
    
    if integrate_delta_wv:
        detector_spectrum = []
        detector_stellar_spectrum = []
        detector_thermal_flux = []
        #Intergrate over the delta_lambda between each wavelength value.
        for inst_wv, inst_dwv in zip(instrument.current_wvs, instrument.current_dwvs):
            wv_start = inst_wv - inst_dwv/2.
            wv_end = inst_wv + inst_dwv/2.

            flux = 1e4*u.AA/u.micron*integrate.quad(intermediate_spectrum, wv_start.value, wv_end.value)[0]*scaled_spectrum.unit # detector spectrum now in e-/s (1e4 is for micron to angstrom conversion)
            stellar_flux = 1e4*u.AA/u.micron*integrate.quad(intermediate_stellar_spectrum, wv_start.value, wv_end.value)[0]*stellar_spectrum.unit # detector spectrum now in e-/s
            thermal_flux = 1e4*u.AA/u.micron*integrate.quad(intermediate_thermal_spectrum, wv_start.value, wv_end.value)[0]*thermal_flux.unit # detector spectrum now in e-/s
            detector_spectrum.append(flux)
            detector_stellar_spectrum.append(stellar_flux)
            detector_thermal_flux.append(thermal_flux)

        detector_spectrum = np.array(detector_spectrum)
        detector_stellar_spectrum = np.array(detector_stellar_spectrum)
        detector_thermal_flux = np.array(detector_thermal_flux)
    else:
        detector_spectrum = 1e4*u.AA/u.micron*intermediate_spectrum(instrument.current_wvs)*instrument.current_dwvs*scaled_spectrum.unit
        detector_stellar_spectrum = 1e4*u.AA/u.micron*intermediate_stellar_spectrum(instrument.current_wvs)*instrument.current_dwvs*stellar_spectrum.unit
        detector_thermal_flux = 1e4*u.AA/u.micron*intermediate_thermal_spectrum(instrument.current_wvs)*instrument.current_dwvs*thermal_flux.unit

    #Multiply by the exposure time
    detector_spectrum *= instrument.exposure_time #The detector spectrum is now in e-
    detector_stellar_spectrum *= instrument.exposure_time #The detector spectrum is now in e-
    detector_thermal_flux *= instrument.exposure_time

    #Multiply by the number of exposures
    detector_spectrum *= instrument.n_exposures
    detector_stellar_spectrum *= instrument.n_exposures
    detector_thermal_flux *= instrument.n_exposures

    #Calculate dark noise. 
    dark_current = instrument.dark_current*instrument.exposure_time*instrument.n_exposures*instrument.spatial_sampling
    detector_thermal_flux +=dark_current    
    ########################################
    ##### Now get the various noise sources:

    speckle_noise,read_noise,photon_noise = get_noise_components(separation,star_aomag,instrument,
        instrument.current_wvs,star_spt,detector_stellar_spectrum,detector_spectrum,detector_thermal_flux,telescope,plot=plot)

    if no_speckle_noise:
        #speckle_noise = speckle_noise*0.
        warnings.warn('no_speckle_noise is being phased out in PSISIM. Please set post_processing_gain=np.inf in the future', FutureWarning)
        print('observation.py: Setting post_processing_gain=np.inf')
        post_processing_gain = np.inf

    #Apply a post-processing gain
    speckle_noise /= post_processing_gain

    print(speckle_noise.unit)
    print(read_noise.unit)
    print(photon_noise.unit)
    ## Sum it all up
    total_noise = np.sqrt(speckle_noise**2+read_noise**2+photon_noise**2)

    # Inject noise into spectrum
    if inject_noise:
        # For each point in the spectrum, draw from a normal distribution,
        # with a mean centered on the spectrum and the standard deviation
        # equal to the noise
        for i,noise in enumerate(total_noise):
            # import pdb; pdb.set_trace()
            ## TODO: Make this poisson so that it's valid still in low photon count regime. 
            if detector_spectrum[i] != detector_spectrum[i]:
                detector_spectrum[i] = np.nan
            else:
                # import pdb; pdb.set_trace()
                detector_spectrum[i] = np.random.normal(detector_spectrum[i].value,noise.value)*noise.unit

    #TODO: Currently everything is in e-. We likely want it in a different unit at the end. 
    
    if return_noise_components:
        return detector_spectrum, total_noise, detector_stellar_spectrum,detector_thermal_flux, np.array([speckle_noise,read_noise,photon_noise])
    else:
        return detector_spectrum, total_noise, detector_stellar_spectrum,detector_thermal_flux

def simulate_observation_nosky(telescope,instrument,planet_table_entry,planet_spectrum,wvs,spectrum_R,
    inject_noise=True,verbose=False,post_processing_gain = 1,return_noise_components=False,stellar_spec=None,
    apply_lsf=False,integrate_delta_wv=False,no_speckle_noise=False,plot=False):
    '''
    DEPRECATED: This function is now just an alias to simulate_observation() with the sky_on flag set to =False accordingly
    '''
    
    # Warn the user that they are calling a deprecated function
    warnings.warn('simulate_observation_nosky is being phased out in PSISIM. Please call simulate_observation with sky_on=False in the future', FutureWarning)
    print('observation.py: Calling simulate_observation with sky_on=False')
    
    if return_noise_components:
        # Call simulate_observation with return_noise_components
        detector_spectrum, total_noise, detector_stellar_spectrum,detector_thermal_flux, noise_comp = simulate_observation(telescope,instrument,planet_table_entry,planet_spectrum,wvs,spectrum_R,inject_noise=inject_noise,
verbose=verbose,post_processing_gain=post_processing_gain,return_noise_components=return_noise_components,stellar_spec=stellar_spec,
    apply_lsf=apply_lsf,integrate_delta_wv=integrate_delta_wv,no_speckle_noise=no_speckle_noise,plot=plot,sky_on=False)
        
        return detector_spectrum, total_noise, detector_stellar_spectrum,detector_thermal_flux, noise_comp
    else:
        # Call simulate_observation without return_noise_components
        detector_spectrum, total_noise, detector_stellar_spectrum,detector_thermal_flux = simulate_observation(telescope,instrument,planet_table_entry,planet_spectrum,wvs,spectrum_R,inject_noise=inject_noise,
verbose=verbose,post_processing_gain=post_processing_gain,return_noise_components=return_noise_components,stellar_spec=stellar_spec,
    apply_lsf=apply_lsf,integrate_delta_wv=integrate_delta_wv,no_speckle_noise=no_speckle_noise,plot=plot,sky_on=False)
        
        return detector_spectrum, total_noise, detector_stellar_spectrum,detector_thermal_flux

def get_noise_components(separation,star_aomag,instrument,wvs,star_spt,stellar_spectrum,detector_spectrum,thermal_spectrum,telescope,plot=False):
    '''
    Calculate all of the different noise contributions
    '''

    # First is speckle noise.
    # Instrument.get_speckle_noise should return things in contrast units relative to the star
    speckle_noise = instrument.get_speckle_noise(separation,star_aomag,instrument.current_filter,wvs,star_spt,telescope)[0]

    #Convert the speckle noise to photons
    # import pdb;pdb.set_trace()
    speckle_noise = stellar_spectrum*speckle_noise

    # Multiply the read noise by sqrt(n_exposures)
    # import pdb; pdb.set_trace() 
    read_noise = np.ones(np.shape(wvs.value))*np.sqrt(instrument.n_exposures)*instrument.read_noise*instrument.spatial_sampling
    
    # detector_spectrum += dark_current
    # dark_noise = np.ones(np.shape(wvs.value))*np.sqrt(dark_current.value) * u.electron

    #Photon noise - including planet spectrum, thermal spectrum and speckle photon noise. Detector_spectrum should be in total of e- now.

    if plot:
        fig = plt.figure(figsize=(30,10))
        plt.plot(wvs,np.sqrt(detector_spectrum.decompose()), label="Object Spectrum")
        plt.plot(wvs,np.sqrt(thermal_spectrum.decompose()), label="Thermal Spectrum")
        plt.plot(wvs,np.sqrt(speckle_noise.decompose()),label="Speckle Noise")
        plt.legend()
        plt.grid()
        plt.show()
        plt.ylim(0,100)


    # import pdb; pdb.set_trace()
    photon_noise = np.sqrt(detector_spectrum.decompose().value + thermal_spectrum.decompose().value + speckle_noise.decompose().value) * u.electron

    # import pdb;pdb.set_trace()

    return speckle_noise,read_noise,photon_noise

def simulate_observation_set(telescope, instrument, planet_table,planet_spectra,wvs,spectra_R,inject_noise=False,
    post_processing_gain=10,return_noise_components=False):
    '''
    Simulates observations of multiple planets, with the same observing configs
    
    Inputs:
    Telescope     - A Telescope object
    Instrument     - An Instrument object
    planet_table - a Universe planet table
    planet_spectra_list - A list of planet spectra. One for each entry in the planet table
    inject_noise - choose whether or not to inject noise into the spectrum now or not


    Outputs: 
    F_lambdas, F_lambda_errors
    '''

    n_planets = np.size(planet_table) #Not sure this will work

    F_lambdas = []
    F_lambdas_stellar = []
    F_lambda_errors = []
    noise_components = []

    for i,planet in enumerate(planet_table):
        if return_noise_components:
            new_F_lambda,new_F_lambda_errors,new_F_lambda_stellar,F_lambda_noise_components = simulate_observation(telescope,instrument,
                planet,planet_spectra[i], wvs, spectra_R, inject_noise = inject_noise, post_processing_gain=post_processing_gain,
                return_noise_components=return_noise_components)
            F_lambdas.append(new_F_lambda)
            F_lambdas_stellar.append(new_F_lambda_stellar)
            F_lambda_errors.append(new_F_lambda_errors)
            noise_components.append(F_lambda_noise_components)
        else:
            new_F_lambda,new_F_lambda_errors,new_F_lambda_stellar = simulate_observation(telescope,instrument,
                planet,planet_spectra[i], wvs, spectra_R, inject_noise = inject_noise, post_processing_gain=post_processing_gain)
            F_lambdas.append(new_F_lambda)
            F_lambdas_stellar.append(new_F_lambda_stellar)
            F_lambda_errors.append(new_F_lambda_errors)


    F_lambdas = np.array(F_lambdas)
    F_lambda_stellar = np.array(F_lambdas_stellar)
    F_lambda_errors = np.array(F_lambda_errors)
    noise_components = np.array(noise_components)
    
    if return_noise_components:
        return F_lambdas,F_lambda_errors,F_lambdas_stellar, noise_components
    else:
        return F_lambdas,F_lambda_errors,F_lambdas_stellar




