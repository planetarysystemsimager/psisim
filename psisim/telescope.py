import numpy as np
import scipy.interpolate as si
import astropy.units as u
import astropy.constants as constants
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from psisim import spectrum

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
    
    def get_atmospheric_transmission(self,wvs,R=1e5):
        '''
        A function that returns the atmospheric transmission for a given set of wavelengths. 

        Later it might be a function of pressure, temperature and humidity

        Inputs: 
        wvs     - A list of wavelengths 
        R       - Spectral resolving power. Doesn't do anything here. 

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
    def __init__(self,airmass = 1.0,water_vapor=  1.0,path="/scr3/dmawet/ETC/"):
        super(TMT, self).__init__(30)

        self.temperature = 276 * u.K
        self.median_seeing = 0.6 * u.arcsec
        self.airmass = airmass
        self.water_vapor = water_vapor

        self.path = path #A path to background, transmission and AO files

    def get_sky_background(self, wvs, R=1e5):
        '''
        A function that returns the sky background for a given set of wavelengths. 

        #Based on Keck sky backgrounds done by Dimitri - Currently super high resolution

        Inputs: 
        wvs     - A list of wavelengths assumed to be microns

        Outputs: 
        backgrounds - A list of backgrounds
        '''

        #Calculate some stuff
        diffraction_limit = (wvs/self.diameter.to(u.micron)*u.radian).to(u.arcsec)
        solidangle = diffraction_limit**2 * 1.13

        #Read in the background file
        sky_background_tmp = np.genfromtxt(self.path+'sky/mk_skybg_zm_'+str(self.water_vapor)+'_'+str(self.airmass)+'_ph.dat', skip_header=0)
        sky_background_MK = sky_background_tmp[:,1]
        sky_background_MK_wave = sky_background_tmp[:,0] * u.nm

        #Interpolate it to the wavelengths we care about
        # sky_background = np.interp(wvs,sky_background_MK_wave.to(u.micron),sky_background_MK)*u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) 
        sky_background = si.interp1d(sky_background_MK_wave.to(u.micron),sky_background_MK,bounds_error=False,fill_value='extrapolate')(wvs)*u.photon/(u.s*u.arcsec**2*u.nm*u.m**2)

        if R < 1e5:
            sky_background = spectrum.downsample_spectrum(sky_background,1e5,R)

        #Multiply by the solid angle
        sky_background *= solidangle

        #Return the function in units that we like. 
        return sky_background.to(u.ph/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wvs))

    def get_atmospheric_transmission(self,wave,R=1e5):
        '''
        A function that computes the sky and telescope throughput 

        Arguments 
        ----------
        wave     - A single wavelength or array of wavelengths [microns]
        '''

        #Read in the sky transmission for the current observing conditions
        sky_trans_tmp = np.genfromtxt(self.path+'sky/mktrans_zm_'+str(self.water_vapor)+'_'+str(self.airmass)+'.dat', skip_header=0)
        sky_trans = sky_trans_tmp[:,1]
        sky_trans_wave = sky_trans_tmp[:,0]*u.micron #* u.nm

        #Interpolate to the wavelengths that we want. 
        # sky_trans_interp = np.interp(wave,sky_trans_wave,sky_trans)
        sky_trans_interp = si.interp1d(sky_trans_wave,sky_trans,bounds_error=False,fill_value='extrapolate')(wave)

        if R < 1e5:
            sky_trans_interp = spectrum.downsample_spectrum(sky_trans_interp,1e5,R)

        return sky_trans_interp

    def get_telescope_throughput(self,wvs,band="TwoMass-J"):
        '''
        Get Telescope throughput for a given observing band.  - These are Keck values

        Currently only Y,TwoMASS-J,TwoMASS-H and TwoMASS-K are supported, otherwise 0.88 is returned. 

        Args:
            band (str): A photometric band. 
        
        '''

        throughput = {"CFHT-Y":0.88,"TwoMASS-J":0.88,"TwoMASS-H":0.88,"TwoMASS-K":0.88}.get(band,0.88)
        
        return throughput*np.ones(np.shape(wvs))
        
class Keck(Telescope):
    '''
    An implementation of the Telescope class
    '''
    def __init__(self,airmass = 1.0,water_vapor=  1.0,path="/scr3/dmawet/ETC/"):
        super(Keck, self).__init__(9.85)

        self.temperature = 276 * u.K
        self.median_seeing = 0.6 * u.arcsec
        self.airmass = airmass
        self.water_vapor = water_vapor

        self.path = path #A path to background, transmission and AO files

    def get_sky_background(self, wvs, R=1e5):
        '''
        A function that returns the sky background for a given set of wavelengths. 

        #Based on Keck sky backgrounds done by Dimitri - Currently super high resolution

        Inputs: 
        wvs     - A list of wavelengths assumed to be microns

        Outputs: 
        backgrounds - A list of backgrounds
        '''

        #Calculate some stuff
        diffraction_limit = (wvs/self.diameter.to(u.micron)*u.radian).to(u.arcsec)
        solidangle = diffraction_limit**2 * 1.13

        #Read in the background file
        sky_background_tmp = np.genfromtxt(self.path+'sky/mk_skybg_zm_'+str(self.water_vapor)+'_'+str(self.airmass)+'_ph.dat', skip_header=0)
        sky_background_MK = sky_background_tmp[:,1]
        sky_background_MK_wave = sky_background_tmp[:,0] * u.nm

        #Interpolate it to the wavelengths we care about
        # sky_background = np.interp(wvs,sky_background_MK_wave.to(u.micron),sky_background_MK)*u.photon/(u.s*u.arcsec**2*u.nm*u.m**2) 
        sky_background = si.interp1d(sky_background_MK_wave.to(u.micron).value,sky_background_MK,bounds_error=False,fill_value='extrapolate')(wvs)*u.photon/(u.s*u.arcsec**2*u.nm*u.m**2)

        if R < 1e5:
            sky_background = spectrum.downsample_spectrum(sky_background,1e5,R)

        #Multiply by the solid angle
        sky_background *= solidangle

        #Return the function in units that we like. 
        return sky_background.to(u.ph/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wvs))

    def get_atmospheric_transmission(self,wave,R=1e5):
        '''
        A function that computes the sky transmission as a function of wavelength 

        Arguments 
        ----------
        wave     - A single wavelength or array of wavelengths [microns]
        '''

        #Read in the sky transmission for the current observing conditions
        sky_trans_tmp = np.genfromtxt(self.path+'sky/mktrans_zm_'+str(self.water_vapor)+'_'+str(self.airmass)+'.dat', skip_header=0)
        sky_trans = sky_trans_tmp[:,1]
        sky_trans_wave = sky_trans_tmp[:,0]*u.micron #* u.nm

        #Interpolate to the wavelengths that we want. 
        # sky_trans_interp = np.interp(wave,sky_trans_wave,sky_trans)
        sky_trans_interp = si.interp1d(sky_trans_wave,sky_trans,bounds_error=False,fill_value='extrapolate')(wave)

        if R < 1e5:
            sky_trans_interp = spectrum.downsample_spectrum(sky_trans_interp,1e5,R)

        return sky_trans_interp

    def get_telescope_throughput(self,wvs,band="TwoMass-J"):
        '''
        Get Telescope throughput for a given observing band. 

        Currently only Y,TwoMASS-J,TwoMASS-H and TwoMASS-K are supported, otherwise 0.88 is returned. 

        Args:
            band (str): A photometric band. 
        
        '''

        throughput = {"CFHT-Y":0.88,"TwoMASS-J":0.88,"TwoMASS-H":0.88,"TwoMASS-K":0.88}.get(band,0.88)
        
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
        


    