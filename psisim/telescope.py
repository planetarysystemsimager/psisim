import numpy as np
import astropy.units as u
import astropy.constants as constants

class Telescope():
    '''
    A general class that describes a telescope and site

    The main properties will be: 
    diameter    - The telescope diameter in meters
    collecting_area    - The telescope collecting area in meters
    
    Properties to consider adding later may be seeing, R0, Tau, etc. 

    The main functions will be: 
    get_sky_background()    - A function to get the sky background
    get_atmospheric_transmission()    - A function to get the atmospheric transmission

    '''
    def __init__(self,diameter, collecting_area=None):
        '''
        Constructor

        '''

        self.diameter = diameter

        #If no collecting area is passed, be naive and assume it's a full circular mirror
        if collecting_area is not None:
            self.collecting_area = collecting_area
        else: 
            self.collecting_area = np.pi*(self.diameter/2)**2

    def get_sky_background(self,wvs):
        '''
        A function that returns the sky background for a given set of wavelengths. 

        Later it might be a function of pressure, temperature and humidity

        Inputs: 
        wvs     - A list of wavelengths 

        Outputs: 
        backgrounds - A list of backgrounds
        '''
        
        if isinstance(wvs,float):
            return 0
        else:
            return np.zeros(len(wvs))


    def get_atmospheric_transmission(self,wvs):
        '''
        A function that returns the atmospheric transmission for a given set of wavelengths. 

        Later it might be a function of pressure, temperature and humidity

        Inputs: 
        wvs     - A list of wavelengths 

        Outputs: 
        transmissions - A list of atmospheric transmissions
        '''

        if isinstance(wvs,float):
            return 1
        else:
            return np.ones(len(wvs))



class TMT(Telescope):
    '''
    An implementation of the Telescope class
    '''
    def __init__(self):
        super(TMT, self).__init__(30)

    def get_sky_background(self ,wvs):
        '''
        A function that returns the sky background for a given set of wavelengths. 

        Later it might be a function of pressure, temperature and humidity

        Inputs: 
        wvs     - A list of wavelengths 

        Outputs: 
        backgrounds - A list of backgrounds
        '''
        returnfloat = False
        if isinstance(wvs, (float, int)):
            returnfloat = True
            wvs = np.array([wvs])
        
        # H band and shorter, just assume background is negligible. For now...
        backgrounds = np.zeros(wvs.shape)

        # Hack where I grabbed these from Allen's Astrophysical Quantities on sky brightness
        # Very approximate step function. 
        backgrounds[np.where((wvs > 1.8) & (wvs <= 2.3))] = 1e-6 # W/m^2/sr/um
        backgrounds[np.where((wvs > 2.3) & (wvs <= 4))] = 1e-2 # W/m^2/sr/um
        backgrounds[np.where(wvs > 4)] = 0.8 # W/m^2/sr/um

        # convert to photons/s/cm^2/angstrom
        diff_size = wvs * u.micron.to(u.m) / self.diameter # radians
        psf_area = np.pi * (diff_size/2)**2 # sr

        backgrounds *= (u.W / (constants.h * constants.c / (wvs * u.micron))).decompose().value * (u.m**-2/u.micron).to(u.cm**-2/u.angstrom) * psf_area

        if returnfloat:
            backgrounds = backgrounds[0]

        return backgrounds

class Keck(Telescope):
    '''
    An implementation of the Telescope class
    '''
    def __init__(self):
        super(Keck, self).__init__(9.85*u.m,collecting_area=)

        self.temperature = 276 * u.K
        self.median_seeing = 0.6 * u.arcsec

    def get_sky_background(self ,wvs):
        '''
        A function that returns the sky background for a given set of wavelengths. 

        Later it might be a function of pressure, temperature and humidity

        Inputs: 
        wvs     - A list of wavelengths 

        Outputs: 
        backgrounds - A list of backgrounds
        '''
        returnfloat = False
        if isinstance(wvs, (float, int)):
            returnfloat = True
            wvs = np.array([wvs])
        
        # H band and shorter, just assume background is negligible. For now...
        backgrounds = np.zeros(wvs.shape)

        # Hack where I grabbed these from Allen's Astrophysical Quantities on sky brightness
        # Very approximate step function. 
        backgrounds[np.where((wvs > 1.8) & (wvs <= 2.3))] = 1e-6 # W/m^2/sr/um
        backgrounds[np.where((wvs > 2.3) & (wvs <= 4))] = 1e-2 # W/m^2/sr/um
        backgrounds[np.where(wvs > 4)] = 0.8 # W/m^2/sr/um

        # convert to photons/s/cm^2/angstrom
        diff_size = wvs * u.micron.to(u.m) / self.diameter # radians
        psf_area = np.pi * (diff_size/2)**2 # sr

        backgrounds *= (u.W / (constants.h * constants.c / (wvs * u.micron))).decompose().value * (u.m**-2/u.micron).to(u.cm**-2/u.angstrom) * psf_area

        if returnfloat:
            backgrounds = backgrounds[0]

        return backgrounds

    def get_telescope_throughput(self,band):
        '''
        Get Telescope throughput for a given observing band. 

        Currently only Y,J,H and K are supported, otherwise 0.88 is returned. 

        Args:
            band (str): A photometric band. 
        
        '''


        throughput = {'Y':0.88,'J':0.88,'H':0.88,'K':0.88}.get(band,0.88)
        
        return throughput
    
    def get_telescope_emissivity(self,band):
        '''
        Get Telescope emissivity for a given observing band. 

        Currently only Y,J,H and K are supported, otherwise 1-0.88 is returned. 

        Args:
            band (str): A photometric band. 
        
        '''
        emissivity = 1-self.get_telescope_throughput(band)

        return emissivity