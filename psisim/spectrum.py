import os
import glob
import psisim
import numpy as np
import scipy.ndimage as ndi
import astropy.units as u
import astropy.constants as consts
from astropy.io import fits, ascii
import scipy.interpolate as si
import copy
from scipy.ndimage.interpolation import shift
from scipy.ndimage import gaussian_filter

try: 
    import pysynphot as ps
except ImportError:
    pass

class Spectrum():
    '''
    A class for spectra manipulation

    The main properties will be: 
    wvs    - sampled wavelengths (np.array of floats, in microns)
    spectrum    - current flux values of the spectrum (np.array of floats, [photons/s/cm^2/A])
    R   - spectral resolution (float)

    The main functions will be: 
    downsample_spectrum    - downsample a spectrum from one resolving power to another
    scale_spectrum_to_vegamag    - scale a spectrum to have a given stellar J-magnitude
    scale_spectrum_to_ABmag     - scale a spectrum to have a given stellar J-magnitude
    apply_doppler_shift    - a function to apply a doppler shift to spectrum
    rotationally_broaden    - a function to rotationally broaden a spectrum


    '''

    def __init__(self, wvs, spectrum, R):

        self.wvs = wvs
        self.spectrum = spectrum
        self.R = R

        return

    def downsample_spectrum(self,R_out):
        '''
        Downsample a spectrum from one resolving power to another

        Inputs: 
        R_out	 - The desired resolving power of the output spectrum

        Outputs:
        new_spectrum - The original spectrum, but now downsampled
        '''
        fwhm = self.R/R_out
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        if isinstance(sigma,float):
            new_spectrum = ndi.gaussian_filter(self.spectrum, sigma)
        else:
            new_spectrum = ndi.gaussian_filter(self.spectrum, sigma.value)

        self.spectrum = new_spectrum
        self.R = R_out

        return new_spectrum

    def scale_spectrum_to_vegamag(self,obj_mag,obj_filt,filters):

        '''
        Based on etc.scale_host_to_ABmag

        Scale a spectrum to have a given stellar J-magnitude

        Args: 
        wvs     -   An array of wavelenghts, corresponding to the spectrum. [float array]
        spectrum -  An array of spectrum, in units photons/s/cm^2/A, assumed to have a magnitude of ___ 
        obj_mag   -  The magnitude that we're scaling to in vega mag (immediately converted to AB)
        '''
        
        #conversion from Vega mag input to AB mag
        obj_mag = convert_vegamag_to_ABmag(obj_filt,obj_mag)

        obj_spec = self.scale_spectrum_to_ABmag(obj_mag,obj_filt,filters)
        
        return obj_spec

    def scale_spectrum_to_ABmag(self,obj_mag,obj_filt,filters):

        '''
        Based on etc.scale_host_to_ABmag

        Scale a spectrum to have a given stellar J-magnitude

        Args: 
        wvs     -   An array of wavelenghts, corresponding to the spectrum. [float array]
        spectrum -  An array of spectrum, in units photons/s/cm^2/A, assumed to have a magnitude of ___ 
        obj_mag   - The magnitude that we're scaling to in AB mag
        '''

        import speclite.filters

        this_filter = speclite.filters.load_filters(obj_filt)
        # import pdb; pdb.set_trace() 

        obj_model_mag = this_filter.get_ab_magnitudes(self.spectrum.to(u.erg/u.m**2/u.s/u.Angstrom,equivalencies=u.spectral_density(self.wvs)), self.wvs.to(u.Angstrom))[obj_filt]
        self.spectrum = self.spectrum * 10**(-0.4*(obj_mag-obj_model_mag))

        return self.spectrum
       

    def apply_doppler_shift(self,delta_wv,rv_shift):
        '''
        A function to apply a doppler shift to a given spectrum

        Inputs: 
        delta_wv    - the spectral resolution of a pixel
        rv_shift    - the rv shift to apply
        '''

        #The average resolution of the spetrograph across the current band
        # delta_lb = instrument.get_wavelength_range()[1]/instrument.current_R

        #The resolution in velocity space
        dvelocity = delta_wv*consts.c/self.wvs

        #The radial velocity of the host in resolution elements. We'll shift the spectrum by the mean shift. 
        rv_shift_resel = np.mean(rv_shift / dvelocity) * 1000*u.m/u.km

        # import pdb; pdb.set_trace()
        spec_shifted = shift(self.spectrum.value,rv_shift_resel.value)*self.spectrum.unit
        self.spectrum = spec_shifted

        
        return spec_shifted

    def rotationally_broaden(self,ld,vsini):
        '''
        A function to rotationally broaden a spectrum
        '''
        from PyAstronomy import pyasl
        # import pdb;pdb.set_trace()
        spec_broadened = pyasl.fastRotBroad(self.wvs.to(u.AA).value,self.spectrum.value,ld,vsini.to(u.km/u.s).value)*self.spectrum.unit
        self.spectrum = spec_broadened

        return spec_broadened

psisim_path = os.path.dirname(psisim.__file__)

bex_labels = ['Age', 'Mass', 'Radius', 'Luminosity', 'Teff', 'Logg', 'NACOJ', 'NACOH', 'NACOKs', 'NACOLp', 'NACOMp', 'CousinsR', 'CousinsI', 'WISE1', 'WISE2', 'WISE3', 'WISE4', 
            'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W', 'F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W', 'VISIRB87', 'VISIRSiC', 
            'SPHEREY', 'SPHEREJ', 'SPHEREH', 'SPHEREKs', 'SPHEREJ2', 'SPHEREJ3', 'SPHEREH2', 'SPHEREH3', 'SPHEREK1', 'SPHEREK2']
# initalize on demand when needed
bex_cloudy_mh0 = {}
bex_clear_mh0 = {}


#try: 
#     import picaso
#     from picaso import justdoit as jdi
#     opacity_folder = os.path.join(os.path.dirname(picaso.__file__), '..', 'reference', 'opacities')
#     dbname = "opacity_LR.db"
#     opacity = jdi.opannection(os.path.join(opacity_folder, dbname))
#except ImportError:
#     pass

def generate_picaso_inputs(planet_table_entry, planet_type, clouds=True, planet_mh=1, stellar_mh=0.0122, planet_teq=None, verbose=False):
    '''
    A function that returns the required inputs for picaso, 
    given a row from a universe planet table

    Inputs:
    planet_table_entry - a single row, corresponding to a single planet
                            from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice" or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
    planet_mh - planetary metalicity. 1 = 1x Solar
    stellar_mh - stellar metalicity
    planet_teq - planet's equilibrium temperature. If None, esimate using blackbody equilibrium temperature

    Outputs:
    params - picaso.justdoit.inputs class
    '''
    

    global opacity

    if planet_type != "Jupiter" and verbose:
        print("Only planet_type='Jupiter' spectra are currently implemented")
        print("Generating a Jupiter-like spectrum")

    params = jdi.inputs(chemeq=True)
    params.approx(raman='none')

    #phase angle
    params.phase_angle(planet_table_entry['Phase']) #radians

    #define gravity
    params.gravity(gravity=10**planet_table_entry['PlanetLogg'], gravity_unit=u.Unit('cm/(s**2)'), 
                    mass=planet_table_entry['PlanetMass'], mass_unit=u.earthMass,
                    radius=planet_table_entry['PlanetRad'], radius_unit=u.earthRad) #any astropy units available

    #The current stellar models do not like log g > 5, so we'll force it here for now. 
    star_logG = planet_table_entry['StarLogg']
    if star_logG > 5.0:
        star_logG = 5.0
    #The current stellar models do not like Teff < 3500, so we'll force it here for now. 
    star_Teff = planet_table_entry['StarTeff']
    if star_Teff < 3500:
        star_Teff = 3500
        
    #define star
    params.star(opacity, star_Teff, stellar_mh, star_logG, radius=planet_table_entry['StarRad'], radius_unit=u.solRad) #opacity db, pysynphot database, temp, metallicity, logg

    # define atmosphere PT profile and mixing ratios. 
    # PT from planetary equilibrium temperature
    if planet_teq is None:
        planet_teq = ((planet_table_entry['StarRad'] * u.solRad/(planet_table_entry['SMA'] * u.au)).decompose()**2 * planet_table_entry['StarTeff']**4)**(1./4)
    params.guillot_pt(planet_teq, 150, -0.5, -1)
    # get chemistry via chemical equillibrium
    planet_C_to_O = 0.55 # not currently suggested to change this
    params.chemeq(planet_C_to_O, planet_mh)

    if clouds:
        # may need to consider tweaking these for reflected light
        params.clouds( g0=[0.9], w0=[0.99], opd=[0.5], p = [1e-3], dp=[5])

    return (params, opacity)

def simulate_spectrum(planet_table_entry, wvs, R, atmospheric_parameters, package="picaso"):
    '''
    Simuluate a spectrum from a given package

    Inputs: 
    planet_table_entry - a single row, corresponding to a single planet
                            from a universe planet table [astropy table (or maybe astropy row)]
    wvs				   - a list of wavelengths to consider
    R				   - the resolving power
    atmospheric parameters - To be defined

    Outputs:
    F_lambda
    '''
    global opacity
    if package.lower() == "picaso":
        # global opacity

        params, _ = atmospheric_parameters
        model_wnos, model_alb, fp_thermal = params.spectrum(opacity, calculation='thermal+reflected')
        model_wvs = 1./model_wnos * 1e4 # microns

        model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1))
        model_dwvs[0] = model_dwvs[1]
        model_R = model_wvs/model_dwvs

        highres_fp_reflected =  model_alb * (planet_table_entry['PlanetRadius']*u.earthRad.to(u.au)/planet_table_entry['SMA'])**2 # flux ratio relative to host star
        highres_fp = highres_fp_reflected + fp_thermal

        spec = Spectrum(wvs, highres_fp, np.mean(model_R))

        lowres_fp = spec.downsample_spectrum(R)

        argsort = np.argsort(model_wvs)

        fp = np.interp(wvs, model_wvs[argsort], lowres_fp[argsort])
        spec.spectrum = fp

        return fp

    elif package.lower() == "picaso+pol":
        '''
        This is just like picaso, but it adds a layer of polarization on top, 
        and returns a polarized intensity spectrum
        Based on the peak polarization vs. albedo curve from Madhusudhan+2012. 
        I'm pretty sure this is based on Rayleigh scattering, and may not be valid 
        for all cloud types. 
        '''
        
        # global opacity

        params, _ = atmospheric_parameters
        model_wnos, model_alb = params.spectrum(opacity)
        model_wvs = 1./model_wnos * 1e4 # microns

        model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1))
        model_dwvs[0] = model_dwvs[1]
        model_R = model_wvs/model_dwvs

        highres_fp =  model_alb * (planet_table_entry['PlanetRadius']*u.earthRad.to(u.au)/planet_table_entry['SMA'])**2 # flux ratio relative to host star

        #Get the polarization vs. albedo curve from Madhusudhan+2012, Figure 5
        albedo, peak_pol = np.loadtxt(os.path.dirname(psisim.__file__)+"/data/polarization/PeakPol_vs_albedo_Madhusudhan2012.csv",
            delimiter=",",unpack=True)
        #Interpolate the curve to the model apbleas
        interp_peak_pol = np.interp(model_alb,albedo,peak_pol)

        #Calculate polarized intensity, given the phase and albedo
        planet_phase = planet_table_entry['Phase']
        rayleigh_curve = np.sin(planet_phase)**2/(1+np.cos(planet_phase)**2)
        planet_polarization_fraction = interp_peak_pol*rayleigh_curve
        highres_planet_polarized_intensity = highres_fp*planet_polarization_fraction

        spec = Spectrum(wvs, highres_fp, np.mean(model_R))
        spec_pol = Spectrum(wvs, highres_planet_polarized_intensity, np.mean(model_R))

        lowres_fp = spec.downsample_spectrum(R)
        lowres_pol = spec_pol.downsample_spectrum(R)

        argsort = np.argsort(model_wvs)

        fp = np.interp(wvs, model_wvs[argsort], lowres_fp[argsort])
        pol = np.interp(wvs, model_wvs[argsort], lowres_pol[argsort])
        spec.spectrum = fp
        spec_pol.spectrum = pol

        return fp,pol

    elif package.lower() == "bex-cooling":
        age, band, cloudy = atmospheric_parameters # age in years, band is 'R', 'I', 'J', 'H', 'K', 'L', 'M', cloudy is True/False
        
        if len(bex_cloudy_mh0) == 0:
            # need to load in models. first time using
            load_bex_models()
        
        if cloudy:
            bex_grid = bex_cloudy_mh0
        else:
            bex_grid = bex_clear_mh0

        masses = np.array(list(bex_grid.keys()))
        closest_indices = np.argsort(np.abs(masses - planet_table_entry['PlanetMass']))
        
        mass1 = masses[closest_indices[0]]
        mass2 = masses[closest_indices[1]]

        curve1 = bex_grid[mass1]
        curve2 = bex_grid[mass2]

        if band == 'R':
            bexlabel = 'CousinsR'
            starlabel = 'StarRmag'
        elif band == 'I':
            bexlabel = 'CousinsI'
            starlabel = 'StarImag'
        elif band == 'J':
            bexlabel = 'SPHEREJ'
            starlabel = 'StarJmag'
        elif band == 'H':
            bexlabel = 'SPHEREH'
            starlabel = 'StarHmag'
        elif band == 'K':
            bexlabel = 'SPHEREKs'
            starlabel = 'StarKmag'
        elif band == 'L':
            bexlabel = 'NACOLp'
            starlabel = 'StarKmag'
        elif band == 'M':
            bexlabel = 'NACOMp'
            starlabel = 'StarKmag'
        else:
            raise ValueError("Band needs to be 'R', 'I', 'J', 'H', 'K', 'L', 'M'. Got {0}.".format(band))

        logage = np.log10(age)
    
        # interpolate in age and wavelength space, but extrapolate as necessary
        fp1 = si.interp1d(curve1['Age'], curve1[bexlabel], bounds_error=False, fill_value="extrapolate")(logage)
        fp2 = si.interp1d(curve2['Age'], curve2[bexlabel], bounds_error=False, fill_value="extrapolate")(logage)

        # linear interpolate in log Mass, extrapoalte as necessary
        fp = si.interp1d(np.log10([mass1, mass2]), [fp1, fp2], bounds_error=False, fill_value="extrapolate")(np.log10(planet_table_entry['PlanetMass'])) # magnitude

        # correct for distance
        fp = fp + 5 * np.log10(planet_table_entry['Distance']/10)

        fs = planet_table_entry[starlabel] # magnitude

        fp = 10**(-(fp - fs)/2.5) # flux ratio of planet to star

        # return as many array elements with save planet flux if multiple are requested (we don't have specetral information)
        if not isinstance(wvs, (float,int)):
            fp = np.ones(wvs.shape) * fp

        return fp

    elif package.lower() == "blackbody":
        a_v = atmospheric_parameters # just albedo
        pl_teff = ((1 - a_v)/4  * (planet_table_entry['StarRad'] * u.solRad/(planet_table_entry['SMA'])).decompose()**2 * planet_table_entry['StarTeff']**4)**(1./4)

        nu = consts.c/(wvs) # freq
        bb_arg_pl = (consts.h * nu/(consts.k_B * pl_teff * u.cds.K)).decompose()
        bb_arg_star = (consts.h * nu/(consts.k_B * planet_table_entry['StarTeff'] * u.cds.K)).decompose()

        thermal_flux_ratio = ((planet_table_entry['PlanetRadius'] * u.earthRad)/(planet_table_entry['StarRad'] * u.solRad)).decompose()**2 * np.expm1(bb_arg_star)/np.expm1(bb_arg_pl)
        
        #Lambertian? What is this equation - To verify later. 
        phi = (np.sin(planet_table_entry['Phase']) + (np.pi - planet_table_entry['Phase'].to(u.rad).value)*np.cos(planet_table_entry['Phase']))/np.pi
        reflected_flux_ratio = phi * a_v / 4 * (planet_table_entry['PlanetRadius'] * u.earthRad/(planet_table_entry['SMA'])).decompose()**2

        return thermal_flux_ratio + reflected_flux_ratio

def get_stellar_spectrum(planet_table_entry,wvs,R,model='Castelli-Kurucz',verbose=False,
                        user_params = None,
                        doppler_shift=False,broaden=False,delta_wv=None):
    ''' 
    A function that returns the stellar spectrum for a given spectral type

    Inputs: 
    planet_table_entry - An entry from a Universe Planet Table
    wvs - The wavelengths at which you want the spectrum. Can be an array [microns]
    R   - The spectral resolving power that you want [int or float]
    Model - The stellar spectrum moodels that you want. [string]
    delta_wv - The spectral resolution of a single pixel. To be used for doppler shifting
    doppler_shift - Boolean, to apply a doppler shift or not
    broaden - boolean, to broaden the spectrum or not. 

    Outputs:
     spectrum - returns the stellar spectrum at the desired wavelengths 
                [photons/s/cm^2/A]
    '''

    if model == 'pickles':
        # import pysynphot as ps
        #Get the pickles spectrum in units of photons/s/cm^2/angstrom. 
        #Wavelength units are microns
        sp = get_pickles_spectrum(planet_table_entry['StarSpT'],verbose=verbose)
        
        #pysynphot  Normalizes everthing to have Vmag = 0, so we'll scale the
        #stellar spectrum by the Vmag
        starVmag = planet_table_entry['StarVmag']
        scaling_factor = 10**(starVmag/-2.5)
        full_stellar_spectrum = sp.flux*scaling_factor

        stellar_spectrum = []

        #If wvs is a float then make it a list for the for loop
        if isinstance(wvs,float):
            wvs = [wvs]

        # Initialize Spectrum class
        spec = Spectrum(wvs, full_stellar_spectrum, R) # This R is not correct until downsample spectrum is applied

        #Now get the spectrum!
        for wv in wvs: 
            #Wavelength sampling of the pickles models is at 5 angstrom
            spec.R = wv/0.0005
            #Down-sample the spectrum to the desired wavelength
            ds = spec.downsample_spectrum(R)
            #Interpolate the spectrum to the wavelength we want
            stellar_spectrum.append(si.interp1d(sp.wave,ds)(wv))
        stellar_spectrum = np.array(stellar_spectrum)
    
    elif model == 'Castelli-Kurucz':
        # For now we're assuming a metallicity of 0, because exosims doesn't
        # provide anything different

        #The current stellar models do not like log g > 5, so we'll force it here for now. 
        star_logG = planet_table_entry['StarLogg']
        if star_logG > 5.0:
            star_logG = 5.0
        #The current stellar models do not like Teff < 3500, so we'll force it here for now. 
        star_Teff = planet_table_entry['StarTeff']
        if star_Teff < 3500:
            star_Teff = 3500

        # Get the Castelli-Kurucz models  
        sp = get_castelli_kurucz_spectrum(star_Teff, 0., star_logG)

        # The flux normalization in pysynphot are all over the place, but it allows
        # you to renormalize, so we will do that here. We'll normalize to the Vmag 
        # of the star, assuming Johnsons filters
        sp_norm = sp.renorm(planet_table_entry['StarVmag'],'vegamag', ps.ObsBandpass('johnson,v'))

        # we normally want to put this in the get_castelli_kurucz_spectrum() function but the above line doens't work if we change units
        sp_norm.convert("Micron")
        sp_norm.convert("photlam")

        stellar_spectrum = []

        #If wvs is a float then make it a list for the for loop
        if isinstance(wvs,float):
            wvs = [wvs]

        # Initialize Spectrum class
        spec = Spectrum(wvs, sp_norm.flux, R) # This R is not correct until downsample spectrum is applied

        #Now get the spectrum!
        for wv in wvs: 
            
            #Get the wavelength sampling of the pysynphot sectrum
            dwvs = sp_norm.wave - np.roll(sp_norm.wave, 1)
            dwvs[0] = dwvs[1]
            #Pick the index closest to our wavelength. 
            ind = np.argsort(np.abs((sp_norm.wave*u.micron-wv)))[0]
            dwv = dwvs[ind]

            spec.R = wv/dwv
            #Down-sample the spectrum to the desired wavelength
            ds = spec.downsample_spectrum(R)
            #Interpolate the spectrum to the wavelength we want
            stellar_spectrum.append(si.interp1d(sp_norm.wave,ds)(wv))
        
        stellar_spectrum = np.array(stellar_spectrum)        

    elif model == 'Phoenix':
        
        path,star_filter,star_mag,filters,instrument_filter = user_params

        available_filters = filters.names
        if star_filter not in available_filters:
            raise ValueError("Your stellar filter of {} is not a valid option. Please choose one of: {}".format(star_filter,available_filters))

        try: 
            star_z = planet_table_entry['StarZ']
        except Exception as e: 
            print(e)
            print("Some error in reading your star Z value, setting Z to zero")
            star_z = '-0.0'
        
        try: 
            star_alpha = planet_table_entry['StarAlpha']
        except Exception as e:
            print(e)
            print("Some error in reading your star alpha value, setting alpha to zero")
            star_alpha ='0.0'

        #Read in the model spectrum        
        wave_u,spec_u = get_phoenix_spectrum(planet_table_entry['StarLogg'],planet_table_entry['StarTeff'],star_z,star_alpha,path=path)

        # Initialize Spectrum class
        spec = Spectrum(wave_u,spec_u,R) # This R is not correct until downsample spectrum is applied

        spec_u = spec.scale_spectrum_to_vegamag(star_mag,star_filter,filters)
        new_ABmag = get_obj_ABmag(wave_u,spec_u,instrument_filter,filters)

        #This loop may be very slow for a hi-res spectrum....
        stellar_spectrum = np.zeros(np.shape(wvs))
        
        #Get the wavelength sampling of the stellar spectrum
        dwvs = wave_u - np.roll(wave_u, 1)
        dwvs[0] = dwvs[1]

        mean_R_in = np.mean(wave_u/dwvs)
        spec.R = mean_R_in

        if R < mean_R_in:
            ds = spec.downsample_spectrum(R)
        else:
            if verbose:
                print("Your requested Resolving power is greater than or equal to the native model. We're not upsampling here, but we should.")
            ds = spec_u
        # ds = spec_u
        stellar_spectrum = np.interp(wvs,wave_u,ds)

        #Now get the spectrum at the wavelengths that we want
        # stellar_spectrum = []
        #If wvs is a float then make it a list for the for loop
        # if isinstance(wvs,float):
            # wvs = [wvs]
        # for i,wv in enumerate(wvs):       
        #     #Get the wavelength sampling of the pysynphot sectrum
        #     dwvs = wave_u - np.roll(wave_u, 1)
        #     dwvs[0] = dwvs[1]
        #     #Pick the index closest to our wavelength. 
        #     ind = np.argsort(np.abs((wave_u-wv)))[0]
        #     dwv = dwvs[ind]

        #     R_in = wv/dwv
        #     #Down-sample the spectrum to the desired wavelength
        #     # import pdb; pdb.set_trace()
        #     if R < R_in:
        #         ds = downsample_spectrum(spec_u, R_in, R)
        #     else: 
        #         if verbose:
        #             print("Your requested Resolving power is higher than the native model, only interpolating between points here.")
        #         ds = spec_u

        #     #Interpolate the spectrum to the wavelength we want
        #     stellar_spectrum[i] = np.interp(wv,wave_u,ds)
        #     # stellar_spectrum.append(si.interp1d(wave_u,ds)(wv))

        stellar_spectrum *= spec_u.unit
        spec.spectrum = stellar_spectrum
        spec.wvs = wvs

        #Now scasle the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        
        stellar_spectrum = spec.scale_spectrum_to_ABmag(new_ABmag,instrument_filter,filters)

    elif model == 'Sonora':
        
        path,star_filter,star_mag,filters,instrument_filter = user_params
        
        available_filters = filters.names
        if star_filter not in available_filters:
            raise ValueError("Your stellar filter of {} is not a valid option. Please choose one of: {}".format(star_filter,available_filters))

        #Read in the sonora spectrum
        star_logG = planet_table_entry['StarLogg']
        star_Teff = str(int(planet_table_entry['StarTeff']))
        wave_u,spec_u = get_sonora_spectrum(star_logG,star_Teff,path=path)

        # Initialize Spectrum class
        spec = Spectrum(wave_u,spec_u,R)  # This R is not correct until downsample spectrum is applied
        
        spec_u = spec.scale_spectrum_to_vegamag(star_mag,star_filter,filters)
        new_ABmag = get_obj_ABmag(wave_u,spec_u,instrument_filter,filters)

        #Get the wavelength sampling of the stellar spectrum
        dwvs = wave_u - np.roll(wave_u, 1)
        dwvs[0] = dwvs[1]

        mean_R_in = np.mean(wave_u/dwvs)
        spec.R = mean_R_in

        if R < mean_R_in:
            ds = spec.downsample_spectrum(R)
        else:
            if verbose:
                print("Your requested Resolving power is greater than or equal to the native model. We're not upsampling here, but we should.")
            ds = spec_u
        # ds = spec_u
        stellar_spectrum = np.interp(wvs,wave_u,ds)
        
        #Now get the spectrum at the wavelengths that we want
        # stellar_spectrum = np.zeros(np.shape(wvs))
        #If wvs is a float then make it a list for the for loop
        # if isinstance(wvs,float):
            # wvs = [wvs]

        # #This loop may be very slow for a hi-res spectrum....
        # for i,wv in enumerate(wvs):       
        #     #Get the wavelength sampling of the pysynphot sectrum
        #     dwvs = wave_u - np.roll(wave_u, 1)
        #     dwvs[0] = dwvs[1]
        #     #Pick the index closest to our wavelength. 
        #     ind = np.argsort(np.abs((wave_u-wv)))[0]
        #     dwv = dwvs[ind]

        #     R_in = wv/dwv
        #     #Down-sample the spectrum to the desired wavelength
        #     # import pdb; pdb.set_trace()
        #     if R < R_in:
        #         ds = downsample_spectrum(spec_u, R_in, R)
        #     else: 
        #         if verbose:
        #             print("Your requested Resolving power is higher than the native model, only interpolating between points here.")
        #         ds = spec_u

        #     #Interpolate the spectrum to the wavelength we want
        #     stellar_spectrum[i] = np.interp(wv,wave_u,ds)
        #     # stellar_spectrum.append(si.interp1d(wave_u,ds)(wv))

        stellar_spectrum *= spec_u.unit
        spec.spectrum = stellar_spectrum
        spec.wvs = wvs
        #Now scasle the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        stellar_spectrum = spec.scale_spectrum_to_ABmag(new_ABmag,instrument_filter,filters)

    else:
        if verbose:
            print("We only support 'pickles', 'Castelli-Kurucz', 'Phoenix' and 'Sonora' models for now")
        return -1

    spec.spectrum = stellar_spectrum

    ## Apply a doppler shift if you'd like.
    if doppler_shift:
        if delta_wv is not None:
            if "StarRadialVelocity" in planet_table_entry.keys():
                
                stellar_spectrum = spec.apply_doppler_shift(delta_wv,planet_table_entry['StarRadialVelocity'])

            else:
                raise KeyError("The StarRadialVelocity key is missing from your target table. It is needed for a doppler shift. ")
        else: 
            print("You need to pass a delta_wv keyword to get_stellar_spectrum to apply a doppler shift")
    
    # import pdb;pdb.set_trace()
    ## Rotationally broaden if you'd like
    if broaden:
        if ("StarVsini" in planet_table_entry.keys()) and ("StarLimbDarkening" in planet_table_entry.keys()):
            stellar_spectrum = spec.rotationally_broaden(planet_table_entry['StarLimbDarkening'],planet_table_entry['StarVsini'])
        else:
            raise KeyError("The StarVsini key is missing from your target table. It is needed for a doppler shift. ")

    return spec

def get_pickles_spectrum(spt,verbose=False):
    '''
    A function that retuns a pysynphot pickles spectrum for a given spectral type
    '''

    #Read in the pickles master list. 
    pickles_dir = os.environ['PYSYN_CDBS']+"grid/pickles/dat_uvk/"
    pickles_filename = pickles_dir+"pickles_uk.fits"
    pickles_table = np.array(fits.open(pickles_filename)[1].data)
    pickles_filenames = [x[0].decode().replace(" ","") for x in pickles_table]
    pickles_spts = [x[1].decode().replace(" ","") for x in pickles_table]
    
    #The spectral types output by EXOSIMS are sometimes annoying
    spt = spt.replace(" ","").split("/")[-1]

    #Sometimes there are fractional spectral types. Rounding to nearest integer
    spt_split = spt.split(".")
    if np.size(spt_split) > 1: 
        spt = spt_split[0] + spt_split[1][1:]

    #Get the index of the relevant pickles spectrum filename
    try: 
        ind = pickles_spts.index(spt)
    except: 
        if verbose:
            print("Couldn't match spectral type {} to the pickles library".format(spt))
            print("Assuming 'G0V'")
        ind = pickles_spts.index('G0V')

    sp = ps.FileSpectrum(pickles_dir+pickles_filenames[ind]+".fits")
    sp.convert("Micron")
    sp.convert("photlam")
    
    return sp

def get_castelli_kurucz_spectrum(teff,metallicity,logg):
    '''
    A function that returns the pysynphot spectrum given the parameters
    based on the Castelli-Kurucz Atlas

    Retuns the pysynphot spectrum object with wavelength units of microns
    and flux units of photons/s/cm^2/Angstrom
    '''
    sp = ps.Icat('ck04models',teff,metallicity,logg)

    return sp

def get_phoenix_spectrum(star_logG,star_Teff,star_z,star_alpha,path='/scr3/dmawet/ETC/'):
    '''
    Read in a pheonix spectrum
    '''
    #Read in your logG and make sure it's valid
    available_logGs = [6.00,5.50,5.00,4.50,4.00,3.50,3.00,2.50,2.00,1.50,1.00,0.50]
    if star_logG not in available_logGs:
        raise ValueError("Your star has an invalid logG for Phoenix models. Please pick from {}".format(available_logGs))
    
    #Read in your t_Eff and make sure it's valid
    available_teffs = np.hstack([np.arange(2300,7000,100),np.arange(7000,12200,200)])
    star_Teff = int(star_Teff)
    if star_Teff not in available_teffs:
        raise ValueError("Your star has an invalid T_eff for Phoenix models. Please pick from {}".format(available_teffs))
    
    #Read in your metalicity and make sure it's valid
    available_Z = ['-4.0','-3.0','-2.0','-1.5','-1.0','-0.5','-0.0','+0.5','+1.0']
    if star_z not in available_Z:
        raise ValueError("Your star has an invalid Z for Phoenix models")

    #Read in your alpha value and make sure it's valid
    available_alpha = ['-0.20','0.0','+0.20','+0.40','+0.60','+0.80','+1.00','+1.20']
    if star_alpha not in available_alpha:
        raise ValueError("Your star has an invalid alpha for Phoenix models")

    #Get the right directory and file path
    if star_alpha =='0.0':
        dir_host_model = "Z"+str(star_z)
        # host_filename = 'lte'+str(star_Teff).zfill(5)+'-'+str(star_logG)+str(star_z)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        host_filename = 'lte{}-{:.2f}{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(str(star_Teff).zfill(5),star_logG,star_z)
    else: 
        dir_host_model='Z'+str(star_z)+'.Alpha='+str(host_alpha)
        # host_filename = 'lte'+str(star_Teff).zfill(5)+'-'+str(star_logG)+str(star_z)+'.Alpha='+str(star_alpha)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        host_filename = 'lte{}-{:.2f}{}.Alpha={}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(str(star_Teff).zfill(5),star_logG,star_z,star_alpha)

    path_to_file_host = path+'HIResFITS_lib/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'+dir_host_model+'/'+host_filename
        

    #Now read in the spectrum and put it in the right file
    wave_data = fits.open(path+'HIResFITS_lib/phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')[0].data
    wave_u = wave_data * u.AA
    wave_u = wave_u.to(u.micron)
    hdulist = fits.open(path_to_file_host, ignore_missing_end=True)
    spec_data = hdulist[0].data
    spec_u = spec_data * u.erg/u.s/u.cm**2/u.cm
    #The original code outputs as above, but really we want it in photons/s/cm^2/A
    spec_u = spec_u.to(u.ph/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wave_u))

    return wave_u,spec_u

def get_sonora_spectrum(star_logG,star_Teff,path='/src3/dmawet/ETC/'):
    '''
    A function that returns a sonora spectrum
    '''
    #Read in your logG and make sure it's valid
    logG_dict = {'3.00':10,'3.25':17,'3.50':31,'3.75':56,'4.00':100,'4.25':178,'4.75':562,'5.00':1000,'5.25':1780,'5.50':3160}
    available_logGs = np.array(list(logG_dict.keys()),dtype=np.float64)
    # import pdb; pdb.set_trace()
    if star_logG not in available_logGs:
        raise ValueError("Your star has an invalid logG of {} for Sonora models, please choose from: {}".format(star_logG,available_logGs))
        
    logG_key = logG_dict["{:.2f}".format(star_logG)]
    
    #Read in your t_Eff and make sure it's valid
    available_teffs = ['200','225','250','275','300','325','350','375','400','425','450','475','500','525','550','575','600','650','700','750','800','850','900','950','1000','1100','1200','1300','1400','1500','1600','1700','1800','1900','2000','2100','2200','2300','2400']
    if star_Teff not in available_teffs:
        raise ValueError("Your star has an invalid T_eff for the Sonora models")

    host_filename = 'sp_t'+str(star_Teff)+'g'+str(logG_key)+'nc_m0.0'
    path_to_file = path+'sonora/'+ host_filename

    obj_data = np.genfromtxt(path_to_file,skip_header=2)
    wave_u = obj_data[::-1,0] * u.micron
    spec_u = obj_data[::-1,1] * u.erg / u.cm**2 / u.s / u.Hz
    spec_u = spec_u.to(u.erg/u.s/u.cm**2/u.cm,equivalencies=u.spectral_density(wave_u))
    #Convert to our preferred units of photons/s/cm^2/A
    spec_u = spec_u.to(u.ph/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wave_u))

    return wave_u,spec_u

def load_bex_models():
    """
    Helper function to load in BEX Cooling curves as dictionary of astropy tables on demand

    Saves to global variables bex_cloudy_mh0 and bex_clear_mh0

    """
    # get relevant files for interpolatoin
    package_dir = os.path.dirname(__file__)
    bex_dir = os.path.join(package_dir, 'data', 'bex_cooling')
    # grabbing 0 metalicity grid for now
    cloudy_pattern = "BEX_evol_mags_-2_MH_0.00_fsed_1.00_ME_*.dat"
    clear_pattern = "BEX_evol_mags_-2_MH_0.00_ME_*.dat"

    for bex_dict, pattern in zip([bex_clear_mh0, bex_cloudy_mh0], [clear_pattern, cloudy_pattern]):
        grid_files = glob.glob(os.path.join(bex_dir, pattern))
        grid_files.sort()
        # grab masses from filenames
        masses = [float(path.split("_")[-1][:-4]) for path in grid_files]
        for mass, filename in zip(masses, grid_files):
            dat = ascii.read(filename, names=bex_labels)
            bex_dict[mass] = dat

def convert_vegamag_to_ABmag(filter_name,vega_mag):
    '''
    A simple conversion function to convert from vega magnitudes to AB magnitudes

    Inputs:
    filter_name -   A string that holds the filter name. Must be supported. 
    vega_mag    -   The vega magnitude in the given filter. 
    path        -   The path to filter definition files. 
    '''

    ab_offset_dictionary = {'bessell-V':0.02, 'bessell-R':0.21, 'bessell-I':0.45, 'TwoMASS-J':0.91,'TwoMASS-H':1.39,'TwoMASS-K':1.85}

    if filter_name not in ab_offset_dictionary.keys():
        raise ValueError("I am not able to convert your object magnitude from vegamag to ABmag because your filter choice is not in my conversion library. \n Please choose one of the following {}".format(ab_offset_dictionary.keys()))
    
    return vega_mag+ab_offset_dictionary[filter_name]

def get_obj_ABmag(wavelengths,spec,filter_name,filters):
    '''
    A tool to get an objects magnitude in a given filter.
    Assumes you have a calibrated spectrum in appropriate astropy units
    Returns ABmag

    Inputs: 
    wavelengths -  A vector containing the spectrum of your source [astropy quantity]
    spec    -   The spectrum of your source [astropy quantitiy]
    obj_mag - The object magniude in vega mags in the "obj_filter" filter 
    obj_filter - The filter that the magnitude is given in
    '''
    import speclite.filters
    if filter_name not in filters.names:
        raise ValueError("Your requested filter of {} is not in our filter list: {}".format(filter_name,filters.names))
    
    this_filter = speclite.filters.load_filters(filter_name)
    new_mag = this_filter.get_ab_magnitudes(spec.to(u.erg/u.m**2/u.s/u.Angstrom,equivalencies=u.spectral_density(wavelengths)), wavelengths.to(u.Angstrom))[filter_name]

    return new_mag

def load_filters(path=psisim_path+"/data/filter_profiles/"):
    '''
    Load up some filter profiles and put them into speclite
    '''
    import speclite.filters
    CFHT_Y_data = np.genfromtxt(path+'CFHT_y.txt', skip_header=0)
    J_2MASS_data = np.genfromtxt(path+'2MASS_J.txt', skip_header=0)
    H_2MASS_data = np.genfromtxt(path+'2MASS_H.txt', skip_header=0)
    K_2MASS_data = np.genfromtxt(path+'2MASS_K.txt', skip_header=0)
    CFHT_Y = speclite.filters.FilterResponse(
        wavelength = CFHT_Y_data[:,0]/1000 * u.micron,
        response = CFHT_Y_data[:,1]/100, meta=dict(group_name='CFHT', band_name='Y'))
    TwoMASS_J = speclite.filters.FilterResponse(
        wavelength = J_2MASS_data[:,0] * u.micron,
        response = J_2MASS_data[:,1], meta=dict(group_name='TwoMASS', band_name='J'))
    TwoMASS_H = speclite.filters.FilterResponse(
        wavelength = H_2MASS_data[:,0] * u.micron,
        response = H_2MASS_data[:,1], meta=dict(group_name='TwoMASS', band_name='H'))
    TwoMASS_K = speclite.filters.FilterResponse(
        wavelength = K_2MASS_data[:,0] * u.micron,
        response = K_2MASS_data[:,1], meta=dict(group_name='TwoMASS', band_name='K'))
    filters = speclite.filters.load_filters('bessell-V', 'bessell-R', 'bessell-I','CFHT-Y','TwoMASS-J','TwoMASS-H','TwoMASS-K')
    return filters

def get_model_ABmags(planet_table_entry,filter_name_list, model='Phoenix',verbose=False,user_params = None):
    '''
    Get the AB color between two filters for a given stellar model
    '''

    #First read in the spectrum. This is somewhat redundant with get_stellar_spectrum function, 
    #I've separated it out here though to keep things modularized...

    filters = user_params[3]


    for filter_name in filter_name_list:
        if filter_name not in filters.names:
            raise ValueError("Your filter, {}, is not in the acceptable filter list: {}".format(filter_name,filters.names))
    
    if model == 'Phoenix':
        
        path,star_filter,star_mag,filters,_ = user_params

        available_filters = filters.names
        if star_filter not in available_filters:
            raise ValueError("Your stellar filter of {} is not a valid option. Please choose one of: {}".format(star_filter,available_filters))

        try: 
            star_z = planet_table_entry['StarZ']
        except Exception as e: 
            print(e)
            print("Some error in reading your star Z value, setting Z to zero")
            star_z = '-0.0'
        

        try: 
            star_alpha = planet_table_entry['StarAlpha']
        except Exception as e:
            print(e)
            print("Some error in reading your star alpha value, setting alpha to zero")
            star_alpha ='0.0'

        #Read in the model spectrum        
        wave_u,spec_u = get_phoenix_spectrum(planet_table_entry['StarLogg'],planet_table_entry['StarTeff'],star_z,star_alpha,path=path)
        
        spec = Spectrum(wave_u, spec_u, None) # R is none since it doesn't matter for the needed function
        #Now scasle the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        stellar_spectrum = spec.scale_spectrum_to_vegamag(star_mag,star_filter,filters)

    elif model == 'Sonora':
        
        path,star_filter,star_mag,filters,_ = user_params
        
        available_filters = filters.names
        if star_filter not in available_filters:
            raise ValueError("Your stellar filter of {} is not a valid option. Please choose one of: {}".format(star_filter,available_filters))

        #Read in the sonora spectrum
        star_logG = planet_table_entry['StarLogg']
        star_Teff = str(int(planet_table_entry['StarTeff']))
        wave_u,spec_u = get_sonora_spectrum(star_logG,star_Teff,path=path)

        spec = Spectrum(wave_u, spec_u, None) # R is none since it doesn't matter for the needed function
        #Now scale the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        stellar_spectrum = spec.scale_spectrum_to_vegamag(star_mag,star_filter,filters)


    mags = filters.get_ab_magnitudes(stellar_spectrum.to(u.erg/u.m**2/u.s/u.Angstrom,equivalencies=u.spectral_density(wave_u)),wave_u.to(u.Angstrom))

    mag_list = []
    for filter_name in filter_name_list:
        mag_list.append(mags[filter_name])

    return mag_list





