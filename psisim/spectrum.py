import os
import psisim
import numpy as np
import scipy.ndimage as ndi
import astropy.units as u
import picaso
from picaso import justdoit as jdi
from astropy.io import fits
import pysynphot as ps
import scipy.interpolate as si

def generate_picaso_inputs(planet_table_entry, planet_type, clouds=True):
    '''
    A function that returns the required inputs for picaso, 
    given a row from a universe planet table

    Inputs:
    planet_table_entry - a single row, corresponding to a single planet
                            from a universe planet table [astropy table (or maybe astropy row)]
    planet_type - either "Terrestrial", "Ice" or "Giant" [string]
    clouds - cloud parameters. For now, only accept True/False to turn clouds on and off

    Outputs:
    params - picaso.justdoit.inputs class
    '''
    opacity = jdi.opannection()

    params = jdi.inputs()

    #phase angle
    params.phase_angle(planet_table_entry['Phase']) #radians

    #define gravity
    params.gravity(gravity=10**planet_table_entry['PlanetLogg'], gravity_unit=u.Unit('cm/(s**2)')) #any astropy units available

    #define star
    params.star(opacity, planet_table_entry['StarTeff'], 0, planet_table_entry['StarLogg']) #opacity db, pysynphot database, temp, metallicity, logg

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

    elif package.lower() == "hotstart":
        pass

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

