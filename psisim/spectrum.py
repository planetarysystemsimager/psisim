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
import warnings
from scipy.integrate import trapz
from scipy import interpolate


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

    def downsample_spectrum(self,R_out,new_wvs=None):
        '''
        Downsample a spectrum from one resolving power to another

        Inputs: 
        R_out	 - The desired resolving power of the output spectrum
	    (optional) new_wvs - specify wavelength grid to interpolate downsampled spectrum to

        Outputs:
        new_spectrum - The original spectrum, but now downsampled (and optionally interpolated)
        '''
        fwhm = self.R/R_out
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        if isinstance(sigma,float):
            new_spectrum = ndi.gaussian_filter(self.spectrum, sigma)
        else:
            new_spectrum = ndi.gaussian_filter(self.spectrum, sigma.value)

        if new_wvs is not None:
            new_spectrum = np.interp(new_wvs, self.wvs, new_spectrum)
            self.wvs = new_wvs

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

try: 
    import picaso
    from picaso import justdoit as jdi
except ImportError:
    print("Tried importing picaso, but couldn't do it")


psisim_path = os.path.dirname(psisim.__file__)

bex_labels = ['Age', 'Mass', 'Radius', 'Luminosity', 'Teff', 'Logg', 'NACOJ', 'NACOH', 'NACOKs', 'NACOLp', 'NACOMp', 'CousinsR', 'CousinsI', 'WISE1', 'WISE2', 'WISE3', 'WISE4', 
            'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W', 'F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W', 'VISIRB87', 'VISIRSiC', 
            'SPHEREY', 'SPHEREJ', 'SPHEREH', 'SPHEREKs', 'SPHEREJ2', 'SPHEREJ3', 'SPHEREH2', 'SPHEREH3', 'SPHEREK1', 'SPHEREK2']
# initalize on demand when needed
bex_cloudy_mh0 = {}
bex_clear_mh0 = {}


def load_picaso_opacity(dbname=None,wave_range=None):
    '''
    A function that returns a picaso opacityclass from justdoit.opannection
    
    Inputs:
    dbname  - string filename, with path, for .db opacity file to load
              default None: will use the default file that comes with picaso distro
    wave_range - 2 element float list with wavelength bounds for which to run models
                 default None: will pull the entire grid from the opacity file   
    
    Returns:
    opacity - Opacity class from justdoit.opannection
    '''
    # Not needed anymore but kept here for reference as way to get picaso path
    # opacity_folder = os.path.join(os.path.dirname(picaso.__file__), '..', 'reference', 'opacities')
    
    # Alternate assuming user has set environment variable correctly
    # opacity_folder = os.path.join(os.getenv("picaso_refdata"),'opacities')
    
    # dbname = os.path.join(opacity_folder,dbname)
    print("Loading an opacity file from {}".format(dbname)) 
    return jdi.opannection(filename_db=dbname,wave_range=wave_range)


def generate_picaso_inputs(planet_table_entry, planet_type, opacity,clouds=True, planet_mh=1, stellar_mh=0.0122, planet_teq=None, verbose=False):
    '''
    A function that returns the required inputs for picaso, 
    given a row from a universe planet table

    Inputs:
    planet_table_entry - a single row, corresponding to a single planet
                            from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice", or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
    planet_mh - planetary metalicity. 1 = 1x Solar
    stellar_mh - stellar metalicity
    planet_teq - (float) planet's equilibrium temperature. If None, esimate using blackbody equilibrium temperature

    Outputs: 
    (as a tuple: params, opacity)
      params - picaso.justdoit.inputs class
      opacity - Opacity class from justdoit.opannection
    
    NOTE: this assumes a planet phase of 0. You can change the phase in the resulting params object afterwards.
    '''
    
    planet_type = planet_type.lower()
    
    if (planet_type not in ["gas"]) and verbose:
        print("Only planet_type='Gas' spectra are currently implemented")
        print("Generating a Gas-like spectrum")
        planet_type = 'gas'

    params = jdi.inputs()
    params.approx(raman='none')

    #-- Set phase angle.
    # Note: non-0 phase in reflectance requires a different 
      # geometry so we'll deal with that in the simulate_spectrum() call
    params.phase_angle(0)

    #-- Define gravity; any astropy units available
    pl_mass = planet_table_entry['PlanetMass']
    pl_rad  = planet_table_entry['PlanetRadius']
    pl_logg = planet_table_entry['PlanetLogg']
    # NOTE: picaso gravity() won't use the "gravity" input if mass and radius are provided
    params.gravity(gravity=pl_logg.value,gravity_unit=pl_logg.physical.unit,
                   mass=pl_mass.value,mass_unit=pl_mass.unit,
                   radius=pl_rad.value,radius_unit=pl_rad.unit)

    #-- Define star properties
    #The current stellar models do not like log g > 5, so we'll force it here for now. 
    star_logG = planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value
    if star_logG > 5.0:
        star_logG = 5.0
    #The current stellar models do not like Teff < 3500, so we'll force it here for now. 
    star_Teff = planet_table_entry['StarTeff'].to(u.K).value
    if star_Teff < 3500:
        star_Teff = 3500   
    #define star
      #opacity db, pysynphot database, temp, metallicity, logg
    st_rad = planet_table_entry['StarRad']
    pl_sma = planet_table_entry['SMA']
    params.star(opacity, star_Teff, stellar_mh, star_logG,
                radius=st_rad.value, radius_unit=st_rad.unit,
                semi_major=pl_sma.value, semi_major_unit=pl_sma.unit) 

    #-- Define atmosphere PT profile, mixing ratios, and clouds
    if planet_type == 'gas':
        # PT from planetary equilibrium temperature
        if planet_teq is None:
            planet_teq = ((st_rad/pl_sma).decompose()**2 * star_Teff**4)**(1./4)
        params.guillot_pt(planet_teq, 150, -0.5, -1)
        # get chemistry via chemical equillibrium
        params.channon_grid_high()

        if clouds:
            # may need to consider tweaking these for reflected light
            params.clouds( g0=[0.9], w0=[0.99], opd=[0.5], p = [1e-3], dp=[5])
    elif planet_type == 'terrestrial':
        # TODO: add Terrestrial type
        pass
    elif planet_type == 'ice':
        # TODO: add ice type
        pass

    return (params, opacity)

def simulate_spectrum(planet_table_entry,wvs,R,atmospheric_parameters,package="picaso"):
    '''
    Simuluate a spectrum from a given package

    Inputs: 
    planet_table_entry - a single row, corresponding to a single planet
                            from a universe planet table [astropy table (or maybe astropy row)]
    wvs				   - (astropy Quantity array - micron) a list of wavelengths to consider
    R				   - the resolving power
    atmospheric parameters - To be defined

    Outputs:
    F_lambda
    
    
    Notes:
    - "picaso" mode returns reflected spec [contrast], thermal spec [ph/s/cm2/A], and the raw picaso dataframe
    '''
    if package.lower() == "picaso":

        params, opacity = atmospheric_parameters
        
        # Make sure that picaso wavelengths are within requested wavelength range
        op_wv = opacity.wave # this is identical to the model_wvs we compute below
        if (wvs[0].value < op_wv.min()) or (wvs[-1].value > op_wv.max()):
            rngs = (wvs[0].value,wvs[-1].value,op_wv.min(),op_wv.max())
            err  = "The requested wavelength range [%f, %f] is outside the range selected [%f, %f] "%rngs
            err += "from the opacity model (%s)"%opacity.db_filename
        #    raise ValueError(err) 
            warnings.warn(err)    
        
        # non-0 phases require special geometry which takes longer to run.
          # To improve runtime, we always run thermal with phase=0 and simple geom.
          # and then for non-0 phase, we run reflected with the costly geometry
        phase = planet_table_entry['Phase'].to(u.rad).value
        if phase == 0:
            # Perform the simple simulation since 0-phase allows simple geometry
            df = params.spectrum(opacity,full_output=True,calculation='thermal+reflected')
        else:
            # Perform the thermal simulation as usual with simple geometry
            df1 = params.spectrum(opacity,full_output=True,calculation='thermal')
            # Apply the true phase and change geometry for the reflected simulation
            params.phase_angle(phase, num_tangle=8, num_gangle=8)
            df2 = params.spectrum(opacity,full_output=True,calculation='reflected')
            # Combine the output dfs into one df to be returned
            df = df1.copy(); df.update(df2)
            df['full_output_therm'] = df1.pop('full_output')
            df['full_output_ref'] = df2.pop('full_output')

        # Extract what we need now
        model_wnos = df['wavenumber']
        fpfs_reflected = df['fpfs_reflected']
        fp_thermal = df['thermal']

        # Compute model wavelength sampling
        model_wvs = 1./model_wnos * 1e4 *u.micron        
        model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1))
        model_dwvs[0] = model_dwvs[1]
        model_R = model_wvs/model_dwvs
        
        # Make sure that model resolution is higher than requested resolution
        if R > np.mean(model_R):
            wrn = "The requested resolution (%0.2f) is higher than the opacity model resolution (%0.2f)."%(R,np.mean(model_R))
            wrn += " This is strongly discouraged as we'll be upsampling the spectrum."
            warnings.warn(wrn)

        # model_wvs is reversed so re-sort it and then extract requested wavelengths
        argsort = np.argsort(model_wvs)
        lowres_ref_spec = Spectrum(model_wvs[argsort], fpfs_reflected[argsort], np.mean(model_R))
        lowres_therm_spec = Spectrum(model_wvs[argsort], fp_thermal[argsort], np.mean(model_R))
        
        fpfs_ref = lowres_ref_spec.downsample_spectrum(R, new_wvs=wvs)
        fp_therm = lowres_therm_spec.downsample_spectrum(R, new_wvs=wvs)

        highres_fp_reflected =  model_alb * (planet_table_entry['PlanetRadius']*u.earthRad.to(u.au)/planet_table_entry['SMA'])**2 # flux ratio relative to host star
        highres_fp = highres_fp_reflected + fp_thermal
        
        # fp_therm comes in with units of ergs/s/cm^3, convert to ph/s/cm^2/Angstrom
        fp_therm = fp_therm * u.erg/u.s/u.cm**2/u.cm
        fp_therm = fp_therm.to(u.ph/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wvs))

        return fpfs_ref,fp_therm,df

    elif package.lower() == "picaso+pol":
        '''
        This is just like picaso, but it adds a layer of polarization on top, 
        and returns a polarized intensity spectrum
        Based on the peak polarization vs. albedo curve from Madhusudhan+2012. 
        I'm pretty sure this is based on Rayleigh scattering, and may not be valid 
        for all cloud types. 
        '''

        # TODO: @Max, Dan updated this section to match the new picaso architecture,
        #       following the last section, but I have not tested. You may want to check
        #       if this works.
        
        params, opacity = atmospheric_parameters
        
        # Make sure that picaso wavelengths are within requested wavelength range
        op_wv = opacity.wave # this is identical to the model_wvs we compute below
        if (wvs[0].value < op_wv.min()) or (wvs[-1].value > op_wv.max()):
            rngs = (wvs[0].value,wvs[-1].value,op_wv.min(),op_wv.max())
            err  = "The requested wavelength range [%f, %f] is outside the range selected [%f, %f] "%rngs
            err += "from the opacity model (%s)"%opacity.db_filename
            raise ValueError(err) 
        
        # Create spectrum and extract results
        df = params.spectrum(opacity)
        model_wnos = df['wavenumber']
        model_alb = df['albedo']
        
        # Compute model wavelength sampling
        model_wvs = 1./model_wnos * 1e4 *u.micron
        model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1))
        model_dwvs[0] = model_dwvs[1]
        model_R = model_wvs/model_dwvs

        highres_fpfs =  model_alb * (planet_table_entry['PlanetRadius'].to(u.au)/planet_table_entry['SMA'].to(u.au))**2 # flux ratio relative to host star

        #Get the polarization vs. albedo curve from Madhusudhan+2012, Figure 5
        albedo, peak_pol = np.loadtxt(os.path.dirname(psisim.__file__)+"/data/polarization/PeakPol_vs_albedo_Madhusudhan2012.csv",
            delimiter=",",unpack=True)
        #Interpolate the curve to the model apbleas
        interp_peak_pol = np.interp(model_alb,albedo,peak_pol)

        #Calculate polarized intensity, given the phase and albedo
        planet_phase = planet_table_entry['Phase'].to(u.rad).value
        rayleigh_curve = np.sin(planet_phase)**2/(1+np.cos(planet_phase)**2)
        planet_polarization_fraction = interp_peak_pol*rayleigh_curve
        highres_planet_polarized_intensity = highres_fpfs*planet_polarization_fraction

        argsort = np.argsort(model_wvs)
        spec = Spectrum(model_wvs[argsort], highres_fpfs[argsort], np.mean(model_R))
        spec_pol = Spectrum(model_wvs[argsort], highres_planet_polarized_intensity[argsort], np.mean(model_R))

        fpfs = spec.downsample_spectrum(R,new_wvs=wvs)
        pol = spec_pol.downsample_spectrum(R,new_wvs=wvs)

        # Make sure that model resolution is higher than requested resolution
        if R > np.mean(model_R):
            wrn = "The requested resolution (%0.2f) is higher than the opacity model resolution (%0.2f)."%(R,np.mean(model_R))
            wrn += " This is strongly discouraged as we'll be upsampling the spectrum."
            warnings.warn(wrn)
        
        spec.spectrum = fp
        spec_pol.spectrum = pol

        return fpfs,pol

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
        closest_indices = np.argsort(np.abs(masses - planet_table_entry['PlanetMass'].to(u.earthMass).value))
        
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
        fp = si.interp1d(np.log10([mass1, mass2]), [fp1, fp2], bounds_error=False, fill_value="extrapolate")(np.log10(planet_table_entry['PlanetMass'].to(u.earthMass).value)) # magnitude

        # correct for distance
        fp = fp + 5 * np.log10(planet_table_entry['Distance'].to(u.pc).value/10)

        fs = planet_table_entry[starlabel] # magnitude

        fp = 10**(-(fp - fs)/2.5) # flux ratio of planet to star

        # return as many array elements with save planet flux if multiple are requested (we don't have specetral information)
        if not isinstance(wvs, (float,int)):
            fp = np.ones(wvs.shape) * fp

        return fp

    elif package.lower() == "blackbody":
        a_v = atmospheric_parameters # just albedo
        pl_teff = ((1 - a_v)/4  * (planet_table_entry['StarRad'] / planet_table_entry['SMA']).decompose()**2 * planet_table_entry['StarTeff'].to(u.K).value**4)**(1./4)

        nu = consts.c/(wvs) # freq
        bb_arg_pl = (consts.h * nu/(consts.k_B * pl_teff * u.K)).decompose()
        bb_arg_star = (consts.h * nu/(consts.k_B * planet_table_entry['StarTeff'].to(u.K))).decompose()

        thermal_flux_ratio = (planet_table_entry['PlanetRadius']/planet_table_entry['StarRad']).decompose()**2 * np.expm1(bb_arg_star)/np.expm1(bb_arg_pl)
        
        #Lambertian? What is this equation - To verify later. 
        phi = (np.sin(planet_table_entry['Phase']) + (np.pi - planet_table_entry['Phase'].to(u.rad).value)*np.cos(planet_table_entry['Phase']))/np.pi
        reflected_flux_ratio = phi * a_v / 4 * (planet_table_entry['PlanetRadius']/planet_table_entry['SMA']).decompose()**2

        return thermal_flux_ratio + reflected_flux_ratio

# ---

# +
def scale_stellar(star_mag,star_filter,star_teff):
	"""
	scale spectrum by magnitude
	inputs: 
	so: object with all variables
	mag: magnitude in filter desired

	load new stellar to match bounds of filter since may not match working badnpass elsewhere
	"""
	if star_filter[:7] == 'TwoMASS':s_f='2mass'
	band_star_filter = star_filter[-1]
	zps_star_filter                     = np.loadtxt('/Users/huihaoz/Downloads/psisim-kpic/psisim/data/filter_profiles/zeropoints.txt',dtype=str).T
	izp_star_filter                     = np.where((zps_star_filter[0]==s_f) & (zps_star_filter[1]==band_star_filter))[0]
	filt_zp_star_filter                 = float(zps_star_filter[2][izp_star_filter])
	#zps                     = np.loadtxt('/Users/huihaoz/Downloads/psisim-kpic/psisim/data/filter_profiles/zeropoints.txt',dtype=str).T
	#izp                     = np.where((zps[0]==so.filt.family) & (zps[1]==so.filt.band))[0]
	#so.filt.zp              = float(zps[2][izp])
	# find filter file and load filter
	filter_file         = glob.glob('/Users/huihaoz/Downloads/psisim-kpic/psisim/data/filter_profiles/'  + s_f + '_' +band_star_filter + '.txt')[0]
	#so.filt.filter_file         = glob.glob(so.filt.filter_path + '*' + so.filt.family + '*' +so.filt.band + '.dat')[0]
	filt_xraw, filt_yraw  = np.loadtxt(filter_file).T # nm, transmission out of 1
	if np.max(filt_xraw)>5000: filt_xraw /= 10
	if np.max(filt_xraw) < 10: filt_xraw *= 1000

#	f_filter                       = interpolate.interp1d(filt_xraw, filt_yraw, bounds_error=False,fill_value=0)
#	filt_v, filt_s    = wvs, f_filter(wvs)  #filter profile sampled at stellar

	filt_dl_l                 = np.mean(integrate(filt_xraw, filt_yraw)/filt_xraw) # dlambda/lambda
	filt_center_wavelength    = integrate(filt_xraw,filt_yraw*filt_xraw)/integrate(filt_xraw,filt_yraw)

	if star_teff.value < 2300: # sonora models arent sampled as well so use phoenix as low as can
		g    = '316' # mks units, np.log10(316 * 100)=4.5 to match what im holding for phoenix models.
		teff = str(int(star_teff.value))
		stel_file         = '/Users/huihaoz/Downloads/psisim-kpic/scr3/dmawet/ETC/sonora/' + 'sp_t%sg%snc_m0.0' %(teff,g)
		stelv,stels = load_sonora(stel_file,wav_start=np.min(filt_xraw), wav_end=np.max(filt_xraw))
	else:
		teff = str(int(star_teff.value)).zfill(5)
		stel_file         = '/Users/huihaoz/Downloads/psisim-kpic/scr3/dmawet/ETC/HIResFITS_lib/phoenix.astro.physik.uni-goettingen.de/HIResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/' + 'lte%s-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff)
		stelv,stels = load_phoenix(stel_file,wav_start=np.min(filt_xraw), wav_end=np.max(filt_xraw))
	filt_interp       =  interpolate.interp1d(filt_xraw, filt_yraw, bounds_error=False,fill_value=0)

	filtered_stellar   = stels * filt_interp(stelv)    # filter profile resampled to phoenix times phoenix flux density
	nphot_expected_0   = calc_nphot(filt_dl_l, filt_zp_star_filter , star_mag)    # what's the integrated flux supposed to be in photons/m2/s?
	nphot_phoenix      = integrate(stelv,filtered_stellar)            # what's the integrated flux now? in same units as ^
	
	return nphot_expected_0/nphot_phoenix
def integrate(x,y):
    """
    Integrate y wrt x
    """
    return trapz(y,x=x)
def load_phoenix(stelname,wav_start=750,wav_end=780):
	"""
	load fits file stelname with stellar spectrum from phoenix 
	http://phoenix.astro.physik.uni-goettingen.de/?page_id=15
	
	return subarray 
	
	wav_start, wav_end specified in nm
	
	convert s from egs/s/cm2/cm to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
	"""
	
	# conversion factor

	f = fits.open(stelname)
	spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
	f.close()
	
	path = stelname.split('/')
	wave_file = '/' + os.path.join(*stelname.split('/')[0:-1]) + '/' + \
					'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits' #assume wave in same folder
	f = fits.open(wave_file)
	lam = f[0].data # angstroms
	f.close()
	
	# Convert
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	# Take subarray requested
	isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	# Convert 
	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm
def load_sonora(stelname,wav_start=750,wav_end=780):
	"""
	load sonora model file
	
	return subarray 
	
	wav_start, wav_end specified in nm
	
	convert s from erg/cm2/s/Hz to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf

	wavelenght loaded is microns high to low
	"""
	f = np.loadtxt(stelname,skiprows=2)

	lam  = 10000* f[:,0][::-1] #microns to angstroms, needed for conversiosn
	spec = f[:,1][::-1] # erg/cm2/s/Hz
	
	spec *= 3e18/(lam**2)# convert spec to erg/cm2/s/angstrom
	
	conversion_factor = 5.03*10**7 * lam #lam in angstrom here
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm (my fave)
def calc_nphot(dl_l, zp, mag):
	"""
	http://astroweb.case.edu/ssm/ASTR620/mags.html

	Values are all for a specific bandpass, can refer to table at link ^ for values
	for some bands. Function will return the photons per second per meter squared
	at the top of Earth atmosphere for an object of specified magnitude

	inputs:
	-------
	dl_l: float, delta lambda over lambda for the passband
	zp: float, flux at m=0 in Jansky
	mag: stellar magnitude

	outputs:
	--------
	photon flux
	"""
	phot_per_s_m2_per_Jy = 1.51*10**7# convert to phot/s/m2 from Jansky

	return dl_l * zp * 10**(-0.4*mag) * phot_per_s_m2_per_Jy


# -

# ---

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

        #If wvs is a float then make it a list for the for loop
        if isinstance(wvs,float):
            wvs = [wvs]

        # Initialize Spectrum class
        spec = Spectrum(sp.wave, full_stellar_spectrum, R) # This R is not correct until downsample spectrum is applied

        #Now get the spectrum!
        for wv in wvs: 
            #Wavelength sampling of the pickles models is at 5 angstrom
            spec.R = wv/0.0005
            #Down-sample the spectrum to the desired wavelength and interpolate
            spec.downsample_spectrum(R,new_wvs=wvs)
    
    elif model == 'Castelli-Kurucz':
        # For now we're assuming a metallicity of 0, because exosims doesn't
        # provide anything different

        #The current stellar models do not like log g > 5, so we'll force it here for now. 
        star_logG = planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value
        if star_logG > 5.0:
            star_logG = 5.0
        #The current stellar models do not like Teff < 3500, so we'll force it here for now. 
        star_Teff = planet_table_entry['StarTeff'].to(u.K).value
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
        sp_norm.convert("photlam") #This is photons/s/cm^2/A

        #Astropy units
        sp_units = u.photon/u.s/(u.cm**2)/u.Angstrom
        #If wvs is a float then make it a list for the for loop
        if isinstance(wvs,float):
            wvs = [wvs]

        # Initialize Spectrum class
        spec = Spectrum(sp_norm.wave, sp_norm.flux, R) # This R is not correct until downsample spectrum is applied

        #Now get the spectrum!
        for wv in wvs: 
            
            #Get the wavelength sampling of the pysynphot sectrum
            dwvs = sp_norm.wave - np.roll(sp_norm.wave, 1)
            dwvs[0] = dwvs[1]
            #Pick the index closest to our wavelength. 
            ind = np.argsort(np.abs((sp_norm.wave*u.micron-wv)))[0]
            dwv = dwvs[ind]

            spec.R = wv/dwv
            #Down-sample the spectrum to the desired wavelength and interpolate
            spec.downsample_spectrum(R,new_wvs=wvs) 

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
        wave_u,spec_u = get_phoenix_spectrum(planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value,planet_table_entry['StarTeff'].to(u.K).value,star_z,star_alpha,path=path)

        # Initialize Spectrum class
        spec = Spectrum(wave_u,spec_u,R) # This R is not correct until downsample spectrum is applied

        spec_u = spec.scale_spectrum_to_vegamag(star_mag,star_filter,filters)
        new_ABmag = get_obj_ABmag(wave_u,spec_u,instrument_filter,filters)
        
        #Get the wavelength sampling of the stellar spectrum
        dwvs = wave_u - np.roll(wave_u, 1)
        dwvs[0] = dwvs[1]

        mean_R_in = np.mean(wave_u/dwvs)
        spec.R = mean_R_in

        if R < mean_R_in:
            spec.downsample_spectrum(R,new_wvs=wvs)
        else:
            if verbose:
                print("Your requested Resolving power is greater than or equal to the native model. We're not upsampling here, but we should.")
            spec.spectrum = np.interp(wvs,wave_u,spec_u)
            spec.wvs = wvs

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

        spec.spectrum = spec.spectrum * spec_u.unit

        #Now scasle the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        
        spec.scale_spectrum_to_ABmag(new_ABmag,instrument_filter,filters)

    elif model == 'Sonora':
        
        path,star_filter,star_mag,filters,instrument_filter = user_params
        
        available_filters = filters.names
        if star_filter not in available_filters:
            raise ValueError("Your stellar filter of {} is not a valid option. Please choose one of: {}".format(star_filter,available_filters))

        #Read in the sonora spectrum
        star_logG = planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value
        star_Teff = str(int(planet_table_entry['StarTeff'].to(u.K).value))
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
            ds = spec.downsample_spectrum(R,new_wvs=wvs)
        else:
            if verbose:
                print("Your requested Resolving power is greater than or equal to the native model. We're not upsampling here, but we should.")
            spec.spectrum = np.interp(wvs,wave_u,spec_u)
            spec.wvs = wvs
        
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

        spec.spectrum = spec.spectrum * spec_u.unit
        #Now scasle the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        spec.scale_spectrum_to_ABmag(new_ABmag,instrument_filter,filters)

    else:
        if verbose:
            print("We only support 'pickles', 'Castelli-Kurucz', 'Phoenix' and 'Sonora' models for now")
        return -1

    ## Apply a doppler shift if you'd like.
    if doppler_shift:
        if delta_wv is not None:
            if "StarRadialVelocity" in planet_table_entry.keys():
                
                spec.apply_doppler_shift(delta_wv,planet_table_entry['StarRadialVelocity'])

            else:
                raise KeyError("The StarRadialVelocity key is missing from your target table. It is needed for a doppler shift. ")
        else: 
            print("You need to pass a delta_wv keyword to get_stellar_spectrum to apply a doppler shift")
    
    # import pdb;pdb.set_trace()
    ## Rotationally broaden if you'd like
    if broaden:
        if ("StarVsini" in planet_table_entry.keys()) and ("StarLimbDarkening" in planet_table_entry.keys()):
            spec.rotationally_broaden(planet_table_entry['StarLimbDarkening'],planet_table_entry['StarVsini'])
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
        wave_u,spec_u = get_phoenix_spectrum(planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value,planet_table_entry['StarTeff'].to(u.K).value,star_z,star_alpha,path=path)
        
        spec = Spectrum(wave_u, spec_u, None) # R is none since it doesn't matter for the needed function
        #Now scasle the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        spec.scale_spectrum_to_vegamag(star_mag,star_filter,filters)

    elif model == 'Sonora':
        
        path,star_filter,star_mag,filters,_ = user_params
        
        available_filters = filters.names
        if star_filter not in available_filters:
            raise ValueError("Your stellar filter of {} is not a valid option. Please choose one of: {}".format(star_filter,available_filters))

        #Read in the sonora spectrum
        star_logG = planet_table_entry['StarLogg'].to(u.dex(u.cm/ u.s**2)).value
        star_Teff = str(int(planet_table_entry['StarTeff'].to(u.K).value))
        wave_u,spec_u = get_sonora_spectrum(star_logG,star_Teff,path=path)

        spec = Spectrum(wave_u, spec_u, None) # R is none since it doesn't matter for the needed function
        #Now scale the spectrum so that it has the appropriate vegamagnitude
        #(with an internal AB mag)
        spec.scale_spectrum_to_vegamag(star_mag,star_filter,filters)


    mags = filters.get_ab_magnitudes(spec.spectrum.to(u.erg/u.m**2/u.s/u.Angstrom,equivalencies=u.spectral_density(wave_u)),wave_u.to(u.Angstrom))

    mag_list = []
    for filter_name in filter_name_list:
        mag_list.append(mags[filter_name])

    return mag_list





