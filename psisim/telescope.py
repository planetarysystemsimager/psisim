import numpy as np
import astropy.units as u
import astropy.constants as constants
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu

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

        self.diameter = diameter * u.m

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

    def get_telescope_emission(self,wvs):
        '''
        Returns the telescope emission
        '''

        return np.ones(np.shape(wvs))

    def get_telescope_emissivity(self,wvs):
        '''
        Get Telescope emissivity for a given observing band. 

        Currently only Y,J,H and K are supported, otherwise 1-0.88 is returned. 

        '''
        emissivity = 1-self.get_telescope_throughput(wvs)

        return emissivity
    
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
    def __init__(self,airmass = 1.0,water_vapor=  1.0):
        super(Keck, self).__init__(9.85)

        self.temperature = 276 * u.K
        self.median_seeing = 0.6 * u.arcsec
        self.airmass = airmass
        self.water_vapor = water_vapor

    def get_sky_background(self, wvs, path="/scr3/dmawet/ETC/"):
        '''
        A function that returns the sky background for a given set of wavelengths. 
        Inputs: 
        wvs     - A list of wavelengths assumed to be microns

        Outputs: 
        backgrounds - A list of backgrounds
        '''

        #Calculate some stuff
        diffraction_limit = (wvs/self.diameter.to(u.micron)*u.radian).to(u.arcsec)
        solidangle = diffraction_limit**2 * 1.13

        #Read in the background file
        sky_background_tmp = np.genfromtxt(path+'sky/mk_skybg_zm_'+str(self.water_vapor)+'_'+str(self.airmass)+'_ph.dat', skip_header=0)
        sky_background_MK = sky_background_tmp[:,1]
        sky_background_MK_wave = sky_background_tmp[:,0] * u.nm

        #Interpolate it to the wavelengths we care about
        sky_background = np.interp(wvs,sky_background_MK_wave.to(u.micron),sky_background_MK)*u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) 

        #Multiply by the solid angle
        sky_background *= solidangle

        #Return the function in units that we like. 
        return sky_background.to(u.ph/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wvs))

    def get_atmospheric_transmission(self,wave,path="/scr3/dmawet/ETC/"):
        '''
        A function that computes the sky and telescope throughput 

        Arguments 
        ----------
        wave     - A single wavelength or array of wavelengths [microns]
        path    - The path we we can find the transmission files
        '''

        #Read in the sky transmission for the current observing conditions
        sky_trans_tmp = np.genfromtxt(path+'sky/mktrans_zm_'+str(self.water_vapor)+'_'+str(self.airmass)+'.dat', skip_header=0)
        sky_trans = sky_trans_tmp[:,1]
        sky_trans_wave = sky_trans_tmp[:,0]*u.micron #* u.nm

        #Interpolate to the wavelengths that we want. 
        sky_trans_interp = np.interp(wave,sky_trans_wave,sky_trans)

        return sky_trans_interp

    def get_telescope_throughput(self,wvs,band="TwoMass-J"):
        '''
        Get Telescope throughput for a given observing band. 

        Currently only Y,TwoMASS-J,TwoMASS-H and TwoMASS-K are supported, otherwise 0.88 is returned. 

        Args:
            band (str): A photometric band. 
        
        '''

        throughput = {'Y':0.88,"TwoMASS-J":0.88,"TwoMASS-H":0.88,"TwoMASS-K":0.88}.get(band,0.88)
        
        return throughput*np.ones(np.shape(wvs))
    
    def get_telescope_emissivity(self,wvs,band="TwoMass-J"):
        '''
        Get Telescope emissivity for a given observing band. 

        Currently only Y,J,H and K are supported, otherwise 1-0.88 is returned. 

        Args:
            band (str): A photometric band. 
        
        '''
        
        emissivity = 1-self.get_telescope_throughput(wvs,band=band)

        return emissivity
    
    def get_thermal_emission(self,wvs,band="TwoMass-J"):
        '''
        The telescope emission as a function of wavelength

        Outputs:
        thermal_emission - usnits of photons/s/cm**2/angstrom
        '''

        diffraction_limit = (wvs/self.diameter.to(u.micron)*u.radian).to(u.arcsec)
        solidangle = diffraction_limit**2 * 1.13

        thermal_emission = blackbody_lambda(wvs,self.temperature)
        thermal_emission *= solidangle
        thermal_emission = thermal_emission.to(u.ph/(u.s * u.cm**2 * u.AA),equivalencies=u.spectral_density(wvs))

        thermal_emission *= self.get_telescope_emissivity(wvs,band=band)

        return thermal_emission
        


    