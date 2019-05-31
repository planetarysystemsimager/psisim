import os
import glob
import psisim
import numpy as np
import scipy.ndimage as ndi
import astropy.units as u
import picaso
from picaso import justdoit as jdi
from astropy.io import fits, ascii
import pysynphot as ps
import scipy.interpolate as si


bex_labels = ['Age', 'Mass', 'Radius', 'Luminosity', 'Teff', 'Logg', 'NACOJ', 'NACOH', 'NACOKs', 'NACOLp', 'NACOMp', 'CousinsR', 'CousinsI', 'WISE1', 'WISE2', 'WISE3', 'WISE4', 
              'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W', 'F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W', 'VISIRB87', 'VISIRSiC', 
              'SPHEREY', 'SPHEREJ', 'SPHEREH', 'SPHEREKs', 'SPHEREJ2', 'SPHEREJ3', 'SPHEREH2', 'SPHEREH3', 'SPHEREK1', 'SPHEREK2']
# initalize on demand when needed
bex_cloudy_mh0 = {}
bex_clear_mh0 = {}

opacity = jdi.opannection()

def generate_picaso_inputs(planet_table_entry, planet_type, clouds=True):
    '''
    A function that returns the required inputs for picaso, 
    given a row from a universe planet table

    Inputs:
    planet_table_entry - a single row, corresponding to a single planet
                            from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice" or "Gas" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off

    Outputs:
    params - picaso.justdoit.inputs class
    '''
    global opacity

    params = jdi.inputs()

    #phase angle
    params.phase_angle(planet_table_entry['Phase']) #radians

    #define gravity
    params.gravity(gravity=10**planet_table_entry['PlanetLogg'], gravity_unit=u.Unit('cm/(s**2)')) #any astropy units available

    #The current stellar models do not like log g > 5, so we'll force it here for now. 
    star_logG = planet_table_entry['StarLogg']
    if star_logG > 5.0:
        star_logG = 5.0
        
    #define star
    params.star(opacity, planet_table_entry['StarTeff'], 0, star_logG) #opacity db, pysynphot database, temp, metallicity, logg

    # define atmosphere PT profile and mixing ratios. 
    # Hard coded as Jupiters right now. 
    params.atmosphere(filename=jdi.jupiter_pt(), delim_whitespace=True)

    if clouds:
        # use Jupiter cloud deck for now. 
        params.clouds( filename= jdi.jupiter_cld(), delim_whitespace=True)

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
    if package.lower() == "picaso":
        global opacity

        params, opacity = atmospheric_parameters
        model_wnos, model_alb = params.spectrum(opacity)
        model_wvs = 1./model_wnos * 1e4 # microns

        model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1))
        model_dwvs[0] = model_dwvs[1]
        model_R = model_wvs/model_dwvs

        highres_fp =  model_alb * (planet_table_entry['PlanetRadius']*u.earthRad.to(u.au)/planet_table_entry['SMA'])**2 # flux ratio relative to host star

        lowres_fp = downsample_spectrum(highres_fp, np.mean(model_R), R)

        argsort = np.argsort(model_wvs)

        fp = np.interp(wvs, model_wvs[argsort], lowres_fp[argsort])

    elif package.lower() == "bex-cooling":
        age, band, cloudy = atmospheric_parameters # age in years, band is 'R', 'I', 'J', 'H', 'K', cloudy is True/False
        
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
        else:
            raise ValueError("Band needs to be 'R', 'I', 'J', 'H', 'K'. Got {0}.".format(band))

        logage = np.log10(age)

        # linear interolation in log Age
        fp1 = np.interp(logage, curve1['Age'], curve1[bexlabel]) # magnitude
        fp2 = np.interp(logage, curve2['Age'], curve2[bexlabel]) # magnitude

        # linear interpolate in log Mass
        fp = np.interp(np.log10(planet_table_entry['PlanetMass']), np.log10([mass1, mass2]), [fp1, fp2]) # magnitude
        # correct for distance
        fp = fp + 5 * np.log10(planet_table_entry['Distance']/10)

        fs = planet_table_entry[starlabel] # magnitude

        fp = 10**(-(fp - fs)/2.5) # flux ratio of planet to star

        # return as many array elements with save planet flux if multiple are requested (we don't have specetral information)
        if not isinstance(wvs, (float,int)):
            fp = np.ones(wvs.shape) * fp

    return fp

def downsample_spectrum(spectrum,R_in, R_out):
    '''
    Downsample a spectrum from one resolving power to another

    Inputs: 
    spectrum - F_lambda that has a resolving power of R_in
    R_in 	 - The resolving power of the input spectrum
    R_out	 - The desired resolving power of the output spectrum

    Outputs:
    new_spectrum - The original spectrum, but now downsampled
    '''
    fwhm = R_in/R_out
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))

    new_spectrum = ndi.gaussian_filter(spectrum, sigma)

    return new_spectrum


def get_stellar_spectrum(planet_table_entry,wvs,R,model='pickles',verbose=False):
    ''' 
    A function that returns the stellar spectrum for a given spectral type

    Inputs: 
    planet_table_entry - An entry from a Universe Planet Table
    wvs - The wavelengths at which you want the spectrum. Can be an array [microns]
    R   - The spectral resolving power that you want [int or float]
    Model - The stellar spectrum moodels that you want. [string]

    Outputs:
     spectrum - returns the stellar spectrum at the desired wavelengths 
                [photons/s/cm^2/A]
    '''

    if model == 'pickles':

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

        #Now get the spectrum!
        for wv in wvs: 
            #Wavelength sampling of the pickles models is at 5 angstrom
            R_in = wv/0.0005
            #Down-sample the spectrum to the desired wavelength
            ds = downsample_spectrum(full_stellar_spectrum,R_in,R)
            #Interpolate the spectrum to the wavelength we want
            stellar_spectrum.append(si.interp1d(sp.wave,ds)(wv))
        stellar_spectrum = np.array(stellar_spectrum)
    else:
        if verbose:
            print("We only support 'pickles' models for now")
        return -1

    return stellar_spectrum


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