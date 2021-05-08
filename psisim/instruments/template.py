import numpy as np

class Instrument():
    '''
    A class that defines a general instrument

    The main properties will be: 
    read_noise    - The read noise in e-
    filters - A list of strings of filter names
    ao_filter - A string that is the filter name for the AO mag
    gain     - The detector gain in e-/ADU
    dark_current - The dark current in e-/sec/pixel
    qe - quantum efficiency, as a fraction
    pixel_scale - the size of a pixel in angular units

    There will also be a set of "current_setup" properties:
    exposure_time  - The exposure time in seconds [float]
    n_exposures    - The number of exposures [int]
    current_filter - The current filter [string]
    current_R         - The current resolving power (float)
    current_wvs    - The current wavelength sampling (np.array of floats, in microns)
    current_dwvs   - The current width of a wavelength channel (i.e. delta lambda of a single current_wvs element)
    More to come!

    Later we might also have ao_filter2

    The main functions will be: 
    get_inst_throughput()     - A function that returns the total instrument throughput as a function of wavelength

    '''

    def __init__(self,telescope=None):
        pass

        if telescope is not None:
            self.telescope = telescope

    def get_inst_throughput(self, wvs, planet_flag=False, planet_sep=None):
        '''
        A function that returns the instrument throughput at a given set of wavelengths
        '''

        if isinstance(wvs,float):
            return 0.075
        else:
            return np.ones(len(wvs))*0.075

    def get_filter_transmission(self,wvs,band):
        '''
        A function to get the transmission of a given filter at a given set of wavelengths

        User inputs:
        wvs     - A list of desired wavelengths [um]
        filter_name - A string corresponding to a filter in the filter database
        '''

        # if filter_name not in self.filters:
            # ERROR
        
        if isinstance(wvs,float):
            return 1.
        else:
            return np.ones(len(wvs))

    def get_speckle_noise(self,separations,ao_mag,sci_mag,sci_filter,SpT,ao_mag2=None):
        '''
        A function that returns the speckle noise a the requested separations, 
        given the input ao_mag and science_mag. Later we might need to provide
        ao_mags at two input wavelengths

        Inputs:
        separations - A list of separations in arcseconds where want the speckle noise values [float or np.array of floats]
        ao_mag     - The magnitude that the AO system sees [float]
        sci_mag - The magnitude in the science band [float]
        sci_filter - The science band filter name [string]
        SpT     - The spetral type (or maybe the effective temperature, TBD) [string or float]
        
        Keyword Arguments:
        ao_mag2 - PSI blue might have a visible and NIR WFS, so we need two ao_mags

        Outputs:
        speckle_noise -    the speckle_noise in contrast units
        '''
        pass

    def get_instrument_background(self,wvs,solidangle):
        '''
        Return the instrument background at a given set of wavelengths

        Inputs: 
        wvs - a list of wavelengths in microns

        Outputs: 
        backgrounds - a list of background values at a given wavelength. Unit: photons/s/Angstrom/arcsecond**2 at each wavelength
        '''

        if isinstance(wvs,float):
            return 0.
        else:
            return np.zeros(len(wvs))
