
import pysynphot as psyn
from psisim import spectrum
import numpy as np
import scipy.interpolate as si
import scipy.integrate as integrate

def simulate_observation(telescope,instrument,planet_table_entry,planet_spectrum,wvs,spectrum_R,
    inject_noise=True,verbose=False,post_processing_gain = 10,return_noise_components=False):
    '''
    A function that simulates an observation

    Inputs:
    Telescope     - A Telescope object
    Instrument     - An Instrument object
    planet_table_entry - an entry/row from a Universe planet table
    planet_spectrum - A planet spectrum from simulate spectrum given in contrast units
    observing_configs - To be defined

    Outputs: 
    F_lambda, F_lambda_error
    '''


    ##### ALL UNITS NEED TO BE PROPERLY EXAMINED #####

    #Some relevant planet properties
    separation = planet_table_entry['AngSep']/1000
    star_imag = planet_table_entry['StarImag']
    star_spt = planet_table_entry['StarSpT']

    #Get the stellar spectrum at the wavelengths of interest. 
    #The stellar spectrum will be in units of photons/s/cm^2/angstrom
    stellar_spectrum = spectrum.get_stellar_spectrum(planet_table_entry,wvs,instrument.current_R,
        verbose=verbose)

    #Multiply the stellar spectrum by the collecting area and a factor of 10,000
    #to convert from m^2 to cm^2 and get the stellar spectrum in units of photons/s
    stellar_spectrum *= telescope.collecting_area*10000 # A factor of 10000 to convert the tles

    #Multiply by atmospheric transmission
    stellar_spectrum *= telescope.get_atmospheric_transmission(wvs)

    #Multiply by instrument throughputs
    stellar_spectrum *= instrument.get_inst_throughput(wvs)
    stellar_spectrum *= instrument.get_filter_transmission(wvs,instrument.current_filter)

    #Multiply by the quantum efficiency
    stellar_spectrum *= instrument.qe

    #Now let's put the planet spectrum back into physical units
    #This assumes that you have properly carried around 'wvs' 
    #and that the planet_spectrum is given at the wvs wavelengths. 
    scaled_spectrum = planet_spectrum*stellar_spectrum

    # Instrument and Sky thermal background in photons/s/cm^2/Angstrom
    thermal_sky = telescope.get_sky_background(wvs)
    thermal_sky *= instrument.get_inst_throughput(wvs)
    thermal_inst = instrument.get_instrument_background(wvs) # need to think about this
    thermal_flux = thermal_sky + thermal_inst
    thermal_flux *= telescope.collecting_area*10000 # phtons/s/Angstrom
    thermal_flux *= instrument.get_filter_transmission(wvs, instrument.current_filter)
    thermal_flux *= instrument.qe #e-/s/Angstrom

    #Downsample to instrument wavelength sampling
    detector_spectrum = []
    detector_stellar_spectrum = []
    detector_thermal_flux = []
    intermediate_spectrum = si.interp1d(wvs, scaled_spectrum)
    intermediate_stellar_spectrum = si.interp1d(wvs, stellar_spectrum)
    intermediate_thermal_spectrum = si.interp1d(wvs, thermal_flux)
    for inst_wv, inst_dwv in zip(instrument.current_wvs, instrument.current_dwvs):
        wv_start = inst_wv - inst_dwv/2.
        wv_end = inst_wv + inst_dwv/2.

        flux = 1e4*integrate.quad(intermediate_spectrum, wv_start, wv_end)[0] # detector spectrum now in e-/s (1e4 is for micron to angstrom conversion)
        stellar_flux = 1e4*integrate.quad(intermediate_stellar_spectrum, wv_start, wv_end)[0] # detector spectrum now in e-/s
        thermal_flux = 1e4*integrate.quad(intermediate_thermal_spectrum, wv_start, wv_end)[0] # detector spectrum now in e-/s
        detector_spectrum.append(flux)
        detector_stellar_spectrum.append(stellar_flux)
        detector_thermal_flux.append(thermal_flux)

    detector_spectrum = np.array(detector_spectrum)
    detector_stellar_spectrum = np.array(detector_stellar_spectrum)
    detector_thermal_flux = np.array(detector_thermal_flux)

    #Multiply by the exposure time
    detector_spectrum *= instrument.exposure_time #The detector spectrum is now in e-
    detector_stellar_spectrum *= instrument.exposure_time #The detector spectrum is now in e-
    detector_thermal_flux *= instrument.exposure_time

    #Multiply by the number of exposures
    detector_spectrum *= instrument.n_exposures
    detector_stellar_spectrum *= instrument.n_exposures
    detector_thermal_flux *= instrument.n_exposures

    ########################################
    ##### Now get the various noise sources:

    speckle_noise,read_noise,dark_noise,photon_noise = get_noise_components(separation,star_imag,instrument,
        instrument.current_wvs,star_spt,detector_stellar_spectrum,detector_spectrum,detector_thermal_flux)

    #Apply a post-processing gain
    speckle_noise /= post_processing_gain

    ## Sum it all up
    total_noise = np.sqrt(speckle_noise**2+read_noise**2+dark_noise**2+photon_noise**2)

    # Inject noise into spectrum
    if inject_noise:
        # For each point in the spectrum, draw from a normal distribution,
        # with a mean centered on the spectrum and the standard deviation
        # equal to the noise
        for i,noise in enumerate(total_noise):
            # import pdb; pdb.set_trace()
            detector_spectrum[i] = np.random.normal(detector_spectrum[i],noise)

    #TODO: Currently everything is in e-. We likely want it in a different unit at the end. 
    
    if return_noise_components:
        return detector_spectrum, total_noise, np.array(detector_stellar_spectrum), np.array([speckle_noise,read_noise,dark_noise,photon_noise])
    else:
        return detector_spectrum, total_noise, np.array(detector_stellar_spectrum)

def get_noise_components(separation,star_imag,instrument,wvs,star_spt,stellar_spectrum,detector_spectrum,thermal_spectrum):
    '''
    Calculate all of the different noise contributions
    '''

    #### TODO include photon noise from the speckles
    
    # First is speckle noise.
    # Instrument.get_speckle_noise should return things in contrast units relative to the star
    speckle_noise = instrument.get_speckle_noise(separation,star_imag,instrument.current_filter,wvs,star_spt)[0]

    #Convert the speckle noise to photons
    speckle_noise *= stellar_spectrum 

    # Multiply the read noise by sqrt(n_exposures)
    read_noise = speckle_noise*0.+np.sqrt(instrument.n_exposures)*instrument.read_noise
    
    #Add the dark_current to the spectrum and calculate dark noise. NEVERMIND NOT ADDING TO SPECTRUM RIGHT NOW
    dark_current = instrument.dark_current*instrument.exposure_time*instrument.n_exposures
    # detector_spectrum += dark_current
    dark_noise = speckle_noise*0.+np.sqrt(dark_current)

    #TODO:Add the background noise

    #Photon noise. Detector_spectrum should be in total of e- now.
    photon_noise = np.sqrt(detector_spectrum + thermal_spectrum + speckle_noise)

    return speckle_noise,read_noise,dark_noise,photon_noise

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




