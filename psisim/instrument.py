import psisim
import os
import glob
import scipy.interpolate as si
import numpy as np
import astropy.units as u
import astropy.constants as constants
import pysynphot as ps
import warnings
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu



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

    def get_instrument_background(self,wvs):
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

class PSI_Blue(Instrument):
    '''
    An implementation of Instrument for PSI-Blue
    '''
    def __init__(self,telescope=None):
        super(PSI_Blue,self).__init__()

        # The main instrument properties - static
        self.read_noise = 0.
        self.gain = 1. #e-/ADU
        self.dark_current = 0.
        self.qe = 1. 

        self.filters = ['r','i','z','Y','J','H']
        self.ao_filter = ['i']
        self.ao_filter2 = ['H']

        self.IWA = 0.0055 #Inner working angle in arcseconds. Current value based on 1*lambda/D at 800nm
        self.OWA = 1. #Outer working angle in arcseconds

        if telescope is None:
            self.telescope = psisim.telescope.TMT()
        else:
            self.telescope = telescope

        # The current obseving properties - dynamic
        self.exposure_time = None
        self.n_exposures = None
        self.current_filter = None
        self.current_R = None
        self.current_wvs = None
        self.current_dwvs = None

        # By default we assume a standard integrator, but 'lp' is also acceptable
        self.ao_algo = 'si'

    def get_speckle_noise(self,separations,ao_mag,wvs,telescope,ao_mag2=None,
        contrast_dir=None):
        '''
        Returns the contrast for a given list of separations. 
        The default is to use the contrasts provided so far by Jared Males

        The code currently rounds to the nearest I mag. This could be improved. 

        TODO: Write down somewhere the file format expected. 

        Inputs: 
        separations     - A list of separations at which to calculate the speckle noise [float list length n]
        ao_mag          - The magnitude in the ao band, here assumed to be I-band
        wvs          - A list of wavelengths in microns [float length m]

        Outputs: 
        get_speckle_noise - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]

        '''

        integrator=self.ao_algo

        if contrast_dir is None:
            contrast_dir = os.path.dirname(psisim.__file__)+"/data/default_contrast/"

        if integrator not in ['si','lp']:
            raise ValueError("The integrator you've selected is not supported."
                " We currently only support standard integrator of linear predictor"
                " as 'si or 'lp")


        #### HARDCODE integrator to be 'si' ####
        integrator = 'si'

        #Find all the contrast files
        fnames = glob.glob(contrast_dir+"*"+integrator+"_profile.dat")

        #Extract the magnitudes from the filenames
        mags = [float(fname.split("/")[-1].split("_")[1]) for fname in fnames]

        #### Make an array to hold the contrast profiles for each magnitude
        # Assumes that each file has the same number of entries. 

        #Round the host_Imags
        host_mag = np.around(ao_mag)

        #Deal with possible out of bound mags
        if host_mag < np.min(mags):
            host_mag = np.min(mags)

        if host_mag > np.max(mags):
            host_mag = np.max(mags)

        #Get the file index
        magnitude_index = np.where(mags == host_mag)[0][0]
        #Make sure we support it
        if magnitude_index.shape == 0:
            raise ValueError("We don't yet support the ao_mag you input. "
                "We currently support between {} and {}".format(np.min(mags),np.max(mags)))

        #Now read in the correct contrast file
        contrast_file_contents = np.genfromtxt(fnames[magnitude_index])[:,0:2]
        seps = contrast_file_contents[:,0]
        contrasts = contrast_file_contents[:,1]

        #Make an interpolation function
        contrasts_interp = si.interp1d(seps,contrasts,fill_value='extrapolate')
        #Interpolate to the desired separations
        interpolated_contrasts = contrasts_interp(separations)

        ### At this point we scale the contrast to the wavelength that we want. 
        # The contrasts are currently at an I-band 0.8 micron
        # 

        speckle_noise = np.zeros([np.size(separations),np.size(wvs)])

        if isinstance(wvs,float):
            wvs = [wvs]

        for i,wv in enumerate(wvs):
            speckle_noise[:,i] = interpolated_contrasts*(0.8/wv)**2

        if self.ao_algo == 'lp':
            # Olivier said that realistically we can expect a gain of ~5 within the AO Control radius. 
            # Here I'm being lazy and just applying it across the board

            speckle_noise /= 5

        return speckle_noise


    def set_observing_mode(self,exposure_time,n_exposures,sci_filter,R,wvs,dwvs=None):
        '''
        Sets the current observing setup
        '''

        self.exposure_time = exposure_time
        self.n_exposures = n_exposures

        if sci_filter not in self.filters:
            raise ValueError("The filter you selected is not valid for PSF_Blue. Check the self.filters property")
        else:
            self.current_filter = sci_filter

        self.current_R = R

        self.current_wvs = wvs
        if dwvs is None:
            dwvs = np.abs(wvs - np.roll(wvs, 1))
            dwvs[0] = dwvs[1]
        self.current_dwvs = dwvs

    def detect_planets(self,planet_table,snrs,telescope,smallest_iwa_by_wv=True,user_iwas=None):
        '''
        A function that returns a boolean array indicating whether or not a planet was detected
        '''

        if user_iwas is not None:

            if isinstance(user_iwas,float):
                iwas = self.current_wvs*0. + user_iwas
            elif np.size(user_iwas) != np.size(self.current_wvs):
                raise Exception("The input 'user_iwas' array is not the same size as instrument.current_wvs")
            else:
                iwas = user_iwas
        else:
            if smallest_iwa_by_wv:
                iwas = self.current_wvs*1e-6/telescope.diameter*206265 #Lambda/D in arcseconds
            else: 
                iwas = self.current_wvs*0. + self.IWA 

        detected = np.full((len(planet_table),self.current_wvs.size),False,dtype=bool)

        #For each planet, for each wavelength check the separation and the SNR
        for i,planet in enumerate(planet_table):
            sep = planet['AngSep']/1000
            for j,wv in enumerate(self.current_wvs): 
                # if sep < 0.070:
                    # print(sep,snrs[i,j],(sep > iwas[j]))
                if (sep > iwas[j]) & (sep < self.OWA) & (snrs[i,j] > 5):
                    detected[i,j] = True

        return detected
           
class PSI_Red(PSI_Blue):
    '''
    An implementation of Instrument for PSI-Red. Currently slightly hacked to inherit PSI Blue for code reuse
    '''
    def __init__(self, telescope=None):
        super(PSI_Red,self).__init__()

        # The main instrument properties - static
        self.read_noise = 0.
        self.gain = 1. #e-/ADU
        self.dark_current = 0.
        self.qe = 1. 

        self.filters = ['K', 'L', 'M']
        self.ao_filter = ['i']
        self.ao_filter2 = ['H']

        self.IWA = 0.028 #Inner working angle in arcseconds. Current value based on 1*lambda/D at 3 microns
        self.OWA = 3. #Outer working angle in arcseconds

        if telescope is None:
            self.telescope = psisim.telescope.TMT()
        else:
            self.telescope = telescope


    def set_observing_mode(self,exposure_time,n_exposures,sci_filter,R,wvs,dwvs=None):
        '''
        Sets the current observing setup
        '''

        self.exposure_time = exposure_time
        self.n_exposures = n_exposures

        if sci_filter not in self.filters:
            raise ValueError("The filter you selected is not valid for PSF_Red. Check the self.filters property")
        else:
            self.current_filter = sci_filter

        self.current_R = R

        self.current_wvs = wvs
        if dwvs is None:
            dwvs = np.abs(wvs - np.roll(wvs, 1))
            dwvs[0] = dwvs[1]
        self.current_dwvs = dwvs

    def get_instrument_background(self,wvs):
        '''
        Return the instrument background. 

        Let's use the background limits from Skemer et al. 2018. 


        Inputs: 
        wvs - a list of wavelengths in microns

        Outputs: 
        backgrounds - a list of background values at a given wavelength. Unit TBD
        '''


        # First we'll get the point source limit in a 1-hour integration, basedd on the 
        # numbers from Skemer et al. 2018. These numbers likely include both instrument 
        # background and sky background numbers. For now we're assuming that it's all due
        # to instrument background until we hear something else. 

        if self.current_R <= 10: 
            # Assume Imaging
            point_source_limit = {'K':27.4,'L':21.3,'M':18.7}.get(self.current_filter,18.7) #Assume M-band if something goes wrong
        elif (self.current_R > 10) & (self.current_R <= 1000):
            # Low resolution spectroscopy
            point_source_limit = {'K':25.4,'L':19.5,'M':16.7}.get(self.current_filter,16.7) #Assume M-band if something goes wrong
        elif (self.current_R > 1000) & (self.current_R <= 20000):
            # Medium resolution spectrocopy
            point_source_limit = {'K':23.6,'L':17.7,'M':14.9}.get(self.current_filter,14.9) #Assume M-band if something goes wrong
        elif (self.current_R > 20000):
            #High resolution spectroscopy
            point_source_limit = {'K':22.0,'L':16.1,'M':13.3}.get(self.current_filter,14.9) #Assume M-band if something goes wrong

        #Get the central wavelength (in microns) based on Keck filters
        cntr_wv = {'K':2.196,'L':3.776,'M':4.670}.get(self.current_filter,4.670)
        #Now we'll use pysynphot to estimate the number of photons at the given magnitude
        ps.Vega.convert("photlam")
        sp = ps.FlatSpectrum(point_source_limit,fluxunits='vegamag')

        sp.convert('photlam') #Convert to photons/s/cm^2/Angstrom
        limit = sp(np.array([cntr_wv])*1e4) #Get the spectrum at the center wavelength (convert cntr_wv to angstrom)

        if isinstance(wvs,float):
            return limit[0]
        else:
            return np.repeat(limit,len(wvs))

class hispec(Instrument):
    '''
    An implementation of Instrument for Hispec
    '''

    def __init__(self,telescope=None):
        super(hispec,self).__init__()

        try:
            import speclite
        except Exception as e:
            print(e)
            print("You need to install the speclite python package to use hispec, modhis and kpic simulators")


        # Spectrograph Properties
        self.current_R = 1e5
        self.wavelength_sampling = 3 # How oversampled is your spectrum?
        self.spatial_sampling = 3 #How many spatial pixels per wavelength channel

        # The main instrument properties - static
        self.read_noise = 3 *u.electron # * u.photon#e-/pix/fr
        self.dark_current = 0.02*u.electron/u.s #electrons/s/pix
        self.det_welldepth = 1e5 *u.photon
        self.det_linearity = 0.66*self.det_welldepth
        self.qe = 0.95 * u.electron/u.ph
        self.temperature = 276*u.K

        if telescope is None:
            self.telescope = psisim.telescope.Keck()
        else:
            self.telescope = telescope

        self.th_data = np.genfromtxt(self.telescope.path+'/throughput/hispec_throughput_budget.csv',
                                    skip_header=1,usecols=np.arange(5,166),delimiter=',',missing_values='')

        #AO parameters
        self.nactuators = 32. - 2.0 #The number of DM actuators in one direction
        self.fiber_contrast_gain = 3. #The gain in congtrast thanks to the fiber. 
        self.p_law_dh = -2.0 #The some power law constant Dimitri should explain. 
        self.ao_filter = 'bessell-I' #Available AO filters
        self.d_ao = 0.15 * u.m
        self.area_ao = np.pi*(self.d_ao/2)**2


        
        self.name = "Keck-HISPEC"
        
        #Acceptable filters
        self.filters = ['CFHT-Y','TwoMASS-J','TwoMASS-H','TwoMASS-K'] #Available observing filters

        # self.lsf_width = 1.0/2.35 #The linespread function width in pixels (assuming Gaussian for now)
        

        # The current obseving properties - dynamic
        self.exposure_time = None
        self.n_exposures = None
        self.current_filter = None
        self.current_wvs = None
        self.current_dwvs = None
        self.ao_mag = None
        self.mode = None
    
    def set_observing_mode(self,exposure_time,n_exposures,sci_filter,wvs,dwvs=None, mode="off-axis"):
        '''
        Sets the current observing setup
        '''

        self.exposure_time = exposure_time *u.s
        self.n_exposures = n_exposures

        self.set_current_filter(sci_filter)

        if mode.lower() not in ['off-axis', 'on-axis', 'photonic_lantern']:
            raise ValueError("'mode' must be 'off-axis', 'on-axis', or 'photonic_lantern'")

        self.mode = mode.lower()

        if self.mode == 'photonic_lantern':
            self.n_ch = 7 #Number of output channels for the photonic lantern
        else:
            self.n_ch = None

        self.current_wvs = wvs
        if dwvs is None:
            dwvs = np.abs(wvs - np.roll(wvs, 1))
            dwvs[0] = dwvs[1]
        self.current_dwvs = dwvs

        #Set the line spread function with to be the 
        self.lsf_width = self.get_wavelength_bounds(sci_filter)[1]/self.current_R
        
    def set_current_filter(self,filter_name):
        '''
        Sets the current filter
        '''

        if filter_name in self.filters:
            self.current_filter=filter_name
        else:
            raise ValueError("Your filter {} is not in the filter list: {}".format(filter_name,self.filters))
        
    def get_wavelength_bounds(self, filter_name):
        '''
        Return the cut on, center and cut off wavelengths in microns of the different science filters.
        '''

        filter_options = {"CFHT-Y":(0.940*u.micron,1.018*u.micron,1.090*u.micron),
                          "TwoMASS-J":(1.1*u.micron,1.248*u.micron,1.360*u.micron),
                          "TwoMASS-H":(1.480*u.micron,1.633*u.micron,1.820*u.micron),
                          "TwoMASS-K":(1.950*u.micron,2.2*u.micron,2.350*u.micron)}

        return filter_options.get(filter_name)

    def get_wavelength_range(self):
        '''
        A function that returns a wavelength array with the given R and wavelength sampling. 
        '''
        
        #Return the lower, center and upper wavelengths for a given filter

        # filter_options = {"Y":(0.960,1.018,1.070),"TwoMASS-J":(1.1,1.248,1.360),"TwoMASS-H":(1.480,1.633,1.820),"TwoMASS-K":(1.950,2.2,2.350)}

        if self.current_filter is None:
            print("You need to set the current_filter property. \nReturning -1")
        if self.current_filter not in self.filters:
            print("You selected filter is not valid. Please choose from {}\n Returning -1".format(filter_options.keys()))
            return -1

        #Pick the right filter based on the .current_filter property
        filter_on,filter_center,filter_off = self.get_wavelength_bounds(self.current_filter)
        
        #Get the wavelength channel size, assuming the current R and wavelength sampling at the center wavelength
        delta_wv = filter_center/self.current_R/self.wavelength_sampling
        wavelengths = np.arange(filter_on.value,filter_off.value,delta_wv.value) * u.micron

        return wavelengths

    def get_inst_throughput(self,wvs,planet_flag=False,planet_sep=None):
        '''
        Reads instrument throughput from budget file and interpolates to the given wavelengths
        
        Kwargs:
        planet_flag     - Boolean denoting if planet-specific separation losses should be accounted for [default False]
        planet_sep      - [in arcsecond] Float of angular separation at which to determine planet throughput
        '''

        if self.current_filter not in self.filters:
            raise ValueError("Your current filter of {} is not in the available filters: {}".format(self.current_filter,self.filters))

        # By wavelength from throughput budget file
        th_data = self.th_data
        th_wvs = th_data[0] * u.micron

        th_ao = np.interp(wvs, th_wvs, np.prod(th_data[2:13], axis=0)) # AO throughput 
        th_fiu = np.interp(wvs, th_wvs, np.prod(th_data[14:29], axis=0)) # KPIC throughput
        #th_fcd = np.interp(wvs, th_wvs, th_data[30]) # Fiber Dynamic Coupling (need function to scale with Strehl/NGS, currently unused)
        th_fiber = np.interp(wvs, th_wvs, np.prod(th_data[31:38], axis=0)) # Fiber throughput (excluding fcd above)
        th_spec = np.interp(wvs, th_wvs, np.prod(th_data[39:51], axis=0)) # HISPEC - SPEC throughput

        if planet_flag:
            # Get separation-dependent planet throughput
            th_planet = self.get_planet_throughput(planet_sep, wvs)[0]
        else:
            # Set to 1 to ignore separation effects
            th_planet = 1

        SR = self.compute_SR(wvs)

        th_inst = th_ao * th_fiu * th_fiber * th_planet * th_spec * SR

        return th_inst
    
    def get_inst_emissivity(self,wvs):
        '''
        The instrument emissivity
        '''

        # TODO: do we want to use the throughput w/ or w/o the planet losses?
        return (1-self.get_inst_throughput(wvs))

    def get_spec_throughput(self, wvs):
        '''
        The throughput of the spectrograph - different than the throughput of the inst that you get in self.get_inst_throughput. 
        self.get_inst_throughput includes everything, whereas this is just the spectrograph. 
        '''

        # By wavelength from throughput budget file
        th_data = self.th_data
        th_wvs = th_data[0] * u.micron
        th_spec = np.interp(wvs, th_wvs, np.prod(th_data[39:51], axis=0))

        return th_spec

    def get_instrument_background(self,wvs,solidangle):
        '''
        Returns the instrument background at each wavelength in units of photons/s/Angstrom/arcsecond**2
        '''

        inst_therm = blackbody_lambda(wvs, self.temperature)
        inst_therm *= solidangle
        inst_therm = inst_therm.to(u.ph/(u.micron * u.s * u.cm**2),equivalencies=u.spectral_density(wvs)) * self.area_ao.to(u.cm**2)
        inst_therm *= self.get_inst_emissivity(wvs)

        # inst_therm = inst_therm.to(u.ph/u.s/u.arcsec**2/u.Angstrom)

        inst_therm *= self.get_spec_throughput(wvs)

        return inst_therm

    def load_scale_aowfe(self,seeing,airmass,site_median_seeing=0.6):
        '''
        A function that returns ao wavefront errors as a function of rmag

        Args:
        path     -  Path to an ao errorbudget file [str]
        seeing   -  The current seeing conditions in arcseconds  [float]
        airmass  -  The current airmass [float
        '''
        
        path = self.telescope.path

        #Read in the ao_wfe
        ao_wfe=np.genfromtxt(path+'aowfe/hispec_modhis_ao_errorbudgetb.csv', delimiter=',',skip_header=1)
        ao_rmag = ao_wfe[:,0]

        # if isinstance(self, hispec):
        ao_wfe_ngs=ao_wfe[:,2] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        ao_wfe_lgs=ao_wfe[:,3] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        # elif isinstance(self, modhis):
            # ao_wfe_ngs=ao_wfe[:,4] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
            # ao_wfe_lgs=ao_wfe[:,5] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        # elif isinstance(self, kpic_phaseI):
        #     ao_wfe_ngs=ao_wfe[:,1] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        #     ao_wfe_lgs=ao_wfe[:,3] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        # elif isinstance(self, kpic_phaseII):
        #     ao_wfe_ngs=ao_wfe[:,2] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        #     ao_wfe_lgs=ao_wfe[:,3] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        return ao_rmag,ao_wfe_ngs*u.nm,ao_wfe_lgs*u.nm

    def compute_SR(self,wave):
        '''
        Compute the Strehl ratio given the wavelengths, host magnitude and telescope (which contains observing conditions)
        '''

        path = self.telescope.path

        #Get the AO WFE as a function of rmag
        ao_rmag,ao_wfe_ngs,ao_wfe_lgs = self.load_scale_aowfe(self.telescope.seeing,self.telescope.airmass,
                                            site_median_seeing=self.telescope.median_seeing)

        #We take the minimum wavefront error between natural guide star and laser guide star errors
        ao_wfe = np.min([np.interp(self.ao_mag,ao_rmag, ao_wfe_ngs),np.interp(self.ao_mag,ao_rmag, ao_wfe_lgs)]) * u.nm

        #Compute the ratio
        # import pdb; pdb.set_trace()
        SR = np.array(np.exp(-(2*np.pi*ao_wfe.to(u.micron)/wave)**2))

        if self.mode == 'photonic_lantern':
            # Include the photonic lantern gain
            SR_PL_in = np.array([99.99, 92.7, 73.3, 32.3, 19.5, 10.6])/100.0 # From Jovanovic et al. 2017
            SR_PL_out = np.array([0.5508, 0.5322, 0.5001, 0.4175, 0.3831, 0.3360])/0.96**2 /0.5508 * 0.9 # From Jovanovic et al. 2017
            SR_boost = SR_PL_out/SR_PL_in # From Jovanovic et al. 2017
            p = np.polyfit(SR_PL_in[::-1],SR_boost[::-1],4)

            if self.n_ch != None:
                SR = SR * np.polyval(p,SR)

        return SR

    def get_speckle_noise(self,separations,ao_mag,filter,wvs,star_spt,telescope,ao_mag2=None):
        '''
        Returns the contrast for a given list of separations. 

        Inputs: 
        separations     - A list of separations at which to calculate the speckle noise in arcseconds [float list length n]. Assumes these are sorted. 
        ao_mag          - The magnitude in the ao band, here assumed to be I-band
        wvs          - A list of wavelengths in microns [float length m]
        telescope    - A psisim telescope object. 

        Outputs: 
        get_speckle_noise - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]

        '''

        if self.mode == "on-axis":
            return np.ones([np.size(separations),np.size(wvs)])
        
        if np.size(wvs) < 2:
            wvs = np.array(wvs)

        #Get the Strehl Ratio
        SR = self.compute_SR(wvs)

        p_law_kolmogorov = -11./3
        p_law_ao_coro_filter = self.p_law_dh#-p_law_kolmogorov 

        r0 = 0.55e-6/(telescope.seeing.to(u.arcsecond).value/206265) * u.m #Easiest to ditch the seeing unit here. 

        #The AO control radius in units of lambda/D
        cutoff = self.nactuators/2

        contrast = np.zeros([np.size(separations),np.size(wvs)])

        if np.size(separations) < 2:
            separations = np.array([separations.value])*separations.unit

        # #Do this by wavelength
        # for i, wv in enumerate(wvs):
        #     ang_sep_resel_in = separations/206265/u.arcsecond*telescope.diameter/wv.to(u.m) #Convert separations from arcseconds to units of lambda/D

        #     if isinstance(ang_sep_resel_in.value,float):
        #         ang_sep_resel_in = [ang_sep_resel_in]
        #         index_in = 1
        #     else:
        #         index_in = np.max(np.where(ang_sep_resel_in.value < cutoff))

        #     ang_sep_resel_step=0.1
        #     ang_sep_resel=np.arange(0,10000,ang_sep_resel_step)
        #     index=np.squeeze(np.where(ang_sep_resel == cutoff))

        #     #Dimitri to put in references to this math
        #     r0_sc = r0 * (wv/(0.55*u.micron))**(6./5)
        #     w_halo = telescope.diameter / r0_sc

        #     f_halo = np.pi*(1-SR[i])*0.488/w_halo**2 * (1+11./6*(ang_sep_resel/w_halo)**2)**(-11/6.)

        #     contrast[:,i] = np.interp(ang_sep_resel_in,ang_sep_resel,f_halo)

        #     contrast_inside = f_halo[index]*(ang_sep_resel/ang_sep_resel[index])**p_law_ao_coro_filter
        #     contrast[:,i][:index_in] = np.interp(ang_sep_resel_in[:index_in],ang_sep_resel,contrast_inside)

            #Inside the control radius, we modify the raw contrast
            # contrast[:,i][:index] = contrast[:,i][index]
        
        #Dimitri to put in references to this math
        r0_sc = r0 * (wvs/(0.55*u.micron))**(6./5)
        w_halo = telescope.diameter / r0_sc
        
        for i,sep in enumerate(separations):
            ang_sep_resel_in = sep/206265/u.arcsecond*telescope.diameter/wvs.to(u.m) #Convert separations from arcseconds to units of lambda/D

            # import pdb; pdb.set_trace()
            f_halo = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(ang_sep_resel_in/w_halo)**2)**(-11/6.)

            
            contrast_at_cutoff = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(cutoff/w_halo)**2)**(-11/6.)
            #Fill in the contrast array
            contrast[i,:] = f_halo

            biggest_ang_sep = np.abs(ang_sep_resel_in - cutoff) == np.min(np.abs(ang_sep_resel_in - cutoff))

            contrast[i][ang_sep_resel_in < cutoff] = contrast_at_cutoff[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
        # import pdb;pdb.set_trace()
        #Set the contrast inide the AO control radius
        # import pdb;pdb.set_trace()
        # if np.size(separations) < 2:
        #     contrast[ang_sep_resel_in < cutoff] = np.repeat(contrast[biggest_ang_sep,None],contrast.shape[1],axis=1)[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
        # elif np.size(wvs) < 2:
        #     contrast[ang_sep_resel_in < cutoff] = np.repeat(contrast[None,biggest_ang_sep],contrast.shape[0],axis=0)[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
        # else: 
        #     contrast[ang_sep_resel_in < cutoff] = contrast[biggest_ang_sep]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
        
        #Apply the fiber contrast gain
        contrast /= self.fiber_contrast_gain

        #Make sure nothing is greater than 1. 
        contrast[contrast>1] = 1.

        return contrast

    def get_planet_throughput(self,separations,wvs):
        '''
        Returns the planet throughput for a given list of separations. 

        Inputs: 
        separations  - A list of separations at which to calculate the planet throughput in arcseconds [float list length n]. Assumes these are sorted. 
        wvs          - A list of wavelengths in microns [float length m]

        Outputs: 
        get_planet_throughput - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]
        '''

        ## TODO: Add in coro. effects; Currently returns 1 since no sep.-dependent modes are implemented yet
            # See kpic_phaseII for implementation reference
        th_planet = 1 * np.ones([np.size(separations), np.size(wvs)]) 
        
        return th_planet

class modhis(hispec):
    '''
    An implementaion of Instrument for Modhis
    '''
    def __init__(self):
        super(hispec,self).init()
    
        self.temperature = 243*u.K

        print("MODHIS is horribly incomplete at this point")

    #To be continued. 


class kpic_phaseII(Instrument):
    '''
    An implementation of Instrument for KPIC Phase II
    '''

    def __init__(self,telescope=None):
        super(kpic_phaseII,self).__init__()

        try:
            import speclite
        except Exception as e:
            print(e)
            print("You need to install the speclite python package to use hispec, modhis and kpic simulators")


        # Spectrograph Properties
        self.current_R = 35e3
        self.wavelength_sampling = 3 # How oversampled is your spectrum?
        self.spatial_sampling = 3 #How many spatial pixels per wavelength channel

        # The main instrument properties - static
        self.read_noise = 10 *u.electron # * u.photon#e-/pix/fr
        self.dark_current = 0.67*u.electron/u.s #electrons/s/pix
        self.det_welldepth = 1e5 *u.photon
        self.det_linearity = 0.66*self.det_welldepth
        self.qe = 0.95 * u.electron/u.ph
        self.temperature = 281*u.K

        if telescope is None:
            self.telescope = psisim.telescope.Keck()
        else:
            self.telescope = telescope

        self.th_data = np.genfromtxt(self.telescope.path+'/throughput/hispec_throughput_budget.csv',
                                    skip_header=1,usecols=np.arange(5,166),delimiter=',',missing_values='')

        #AO parameters
        self.nactuators = 32. - 2.0 #The number of DM actuators in one direction
        self.fiber_contrast_gain = 10. #The gain in contrast thanks to the fiber. ('off-axis' mode only)
        self.p_law_dh = -2.0 #The some power law constant Dimitri should explain. 
        self.ao_filter = 'TwoMASS-H' #Available AO filters - per Dimitri
        self.d_ao = 0.15 * u.m
        self.area_ao = np.pi*(self.d_ao/2)**2


        
        self.name = "Keck-KPIC-PhaseII"
        
        #Acceptable filters
        #TODO: check filters below (not clearly set for kpic in Dimitri code)
        self.filters = ['CFHT-Y','TwoMASS-J','TwoMASS-H','TwoMASS-K'] #Available observing filters

        # self.lsf_width = 1.0/2.35 #The linespread function width in pixels (assuming Gaussian for now)
        

        # The current obseving properties - dynamic
        self.exposure_time = None
        self.n_exposures = None
        self.current_filter = None
        self.current_wvs = None
        self.current_dwvs = None
        self.ao_mag = None
        self.mode = None
        self.vortex_charge = None      # needed for vfn mode only
        self.host_diameter= 0.       # needed for vfn mode only (default 0 to disable geometric leakage)
    
    def set_observing_mode(self,exposure_time,n_exposures,sci_filter,wvs,dwvs=None, mode="vfn", vortex_charge=None):
        '''
        Sets the current observing setup
        '''

        self.exposure_time = exposure_time *u.s
        self.n_exposures = n_exposures

        self.set_current_filter(sci_filter)

        if mode.lower() not in ["vfn", "off-axis", "on-axis"]:
            raise ValueError("'mode' must be 'vfn', 'off-axis', or 'on-axis'")

        self.mode = mode.lower()    # lower() to remove errors from common VFN capitalization

        # Set vortex charge for vfn mode
        if self.mode == 'vfn' and (vortex_charge not in [1,2]):
            raise ValueError("'vfn' mode requires a 'vortex_charge' of 1 or 2")
        self.vortex_charge = vortex_charge

        self.current_wvs = wvs
        if dwvs is None:
            dwvs = np.abs(wvs - np.roll(wvs, 1))
            dwvs[0] = dwvs[1]
        self.current_dwvs = dwvs

        #Set the line spread function width to be the 
        self.lsf_width = self.get_wavelength_bounds(sci_filter)[1]/self.current_R
        
    def set_current_filter(self,filter_name):
        '''
        Sets the current filter
        '''

        if filter_name in self.filters:
            self.current_filter=filter_name
        else:
            raise ValueError("Your filter {} is not in the filter list: {}".format(filter_name,self.filters))
        
    def get_wavelength_bounds(self, filter_name):
        '''
        Return the cut on, center and cut off wavelengths in microns of the different science filters.
        '''

        filter_options = {"CFHT-Y":(0.940*u.micron,1.018*u.micron,1.090*u.micron),
                          "TwoMASS-J":(1.1*u.micron,1.248*u.micron,1.360*u.micron),
                          "TwoMASS-H":(1.480*u.micron,1.633*u.micron,1.820*u.micron),
                          "TwoMASS-K":(1.950*u.micron,2.2*u.micron,2.45*u.micron)}

        return filter_options.get(filter_name)

    def get_wavelength_range(self):
        '''
        A function that returns a wavelength array with the given R and wavelength sampling. 
        '''
        
        #Return the lower, center and upper wavelengths for a given filter

        # filter_options = {"Y":(0.960,1.018,1.070),"TwoMASS-J":(1.1,1.248,1.360),"TwoMASS-H":(1.480,1.633,1.820),"TwoMASS-K":(1.950,2.2,2.350)}

        if self.current_filter is None:
            print("You need to set the current_filter property. \nReturning -1")
        if self.current_filter not in self.filters:
            print("You selected filter is not valid. Please choose from {}\n Returning -1".format(filter_options.keys()))
            return -1

        #Pick the right filter based on the .current_filter property
        filter_on,filter_center,filter_off = self.get_wavelength_bounds(self.current_filter)
        
        #Get the wavelength channel size, assuming the current R and wavelength sampling at the center wavelength
        delta_wv = filter_center/self.current_R/self.wavelength_sampling
        wavelengths = np.arange(filter_on.value,filter_off.value,delta_wv.value) * u.micron

        return wavelengths

    def get_inst_throughput(self,wvs,planet_flag=False,planet_sep=None):
        '''
        Reads instrument throughput from budget file and interpolates to given wavelengths
        When reading from budget file, accounts for pertinent lines depending on the instrument mode

        Kwargs:
        planet_flag     - Boolean denoting if planet-specific separation losses should be accounted for [default False]
        planet_sep      - [in arcsecond] Float of angular separation at which to determine planet throughput
        '''

        if self.current_filter not in self.filters:
            raise ValueError("Your current filter of {} is not in the available filters: {}".format(self.current_filter,self.filters))

        # Use wavelength-dependent data from throughput budget file
        th_data = self.th_data
        th_wvs = th_data[0] * u.micron

        th_ao = {"CFHT-Y":0.60,"TwoMASS-J":0.63,"TwoMASS-H":0.66,"TwoMASS-K":0.633}.get(self.current_filter) #K-band from Nem's Feb 2020 report
        if self.mode == 'vfn':
            # From Dimitri: th_fiu = {"CFHT-Y":0.66,"TwoMASS-J":0.68,"TwoMASS-H":0.7,"TwoMASS-K":0.6}.get(self.current_filter) #K-band from Dimitri (KPIC OAPS+FM+dichroic+PIAA(95%)+DM Window(90%)+ADC(90%))
            th_fiu = np.interp(wvs, th_wvs, np.prod(th_data[14:19],axis=0)*np.prod(th_data[20:29],axis=0)) # KPIC throughput from budget (omitting coro.)
            #TODO: add in vortex throughput losses (separate from apodizer losses in throughput file)
            # From Dimitri: th_fiber = {"CFHT-Y":0.99,"TwoMASS-J":0.99,"TwoMASS-H":0.99,"TwoMASS-K":0.98}.get(self.current_filter) #from Dimitri code
            th_fiber = np.interp(wvs, th_wvs, th_data[34]*th_data[37]) # Fiber throughput (endfaces only)
            th_fiber = th_fiber * 0.98 # Add in constant propagation loss (0.98) for now

            #TODO: Figure out how thermal effects (in Observation.py) are modulated by fiber in VFN case and implement accordingly
            
        else:
            th_fiu = np.interp(wvs, th_wvs, np.prod(th_data[14:29], axis=0)) # KPIC throughput from budget
            #th_fcd = np.interp(wvs, th_wvs, th_data[30]) # Fiber Dynamic Coupling (need function to scale with Strehl/NGS, currently unused)
            th_fiber = np.interp(wvs, th_wvs, np.prod(th_data[31:35], axis=0)) # Fiber throughput (excluding fcd above. Also exclude prop., breakout, and second endface)
            th_fiber = th_fiber * 0.98 *  np.interp(wvs, th_wvs, th_data[37]) # Add in const. prop. loss (0.98) and second endface
        th_feu = 0.89   #from Dimitri code

        if planet_flag:
            # Get separation-dependent planet throughput
            th_planet = self.get_planet_throughput(planet_sep, wvs)[0]
        else:
            # Set to 1 to ignore separation effects
            th_planet = 1

        #TODO: figure out if SR is needed for VFN (thinking it's not)
        SR = self.compute_SR(wvs)
        if self.mode == 'vfn':
            SR = np.ones(SR.shape)

        th_spec = self.get_spec_throughput(wvs)
        th_inst = th_ao * th_fiu * th_feu * th_fiber * th_planet * th_spec * SR

        return th_inst
    
    def get_inst_emissivity(self,wvs):
        '''
        The instrument emissivity
        '''
        
        # TODO: do we want to use the throughput w/ or w/o the planet losses?
        return (1-self.get_inst_throughput(wvs))

    def get_spec_throughput(self, wvs):
        '''
        The throughput of the spectrograph - different than the throughput of the inst that you get in self.get_inst_throughput. 
        self.get_inst_throughput includes everything, whereas this is just the spectrograph. 
        '''
        # K-band value for NIRSPEC from Dimitri Code
        th_spec = {"CFHT-Y":0.5,"TwoMASS-J":0.5,"TwoMASS-H":0.5,"TwoMASS-K":0.2}.get(self.current_filter,0.5)

        return th_spec*np.ones(np.shape(wvs))

    def get_instrument_background(self,wvs,solidangle):
        '''
        Returns the instrument background at each wavelength in units of photons/s/Angstrom/arcsecond**2
        '''

        inst_therm = blackbody_lambda(wvs, self.temperature)
        inst_therm *= solidangle
        inst_therm = inst_therm.to(u.ph/(u.micron * u.s * u.cm**2),equivalencies=u.spectral_density(wvs)) * self.area_ao.to(u.cm**2)
        inst_therm *= self.get_inst_emissivity(wvs)

        # inst_therm = inst_therm.to(u.ph/u.s/u.arcsec**2/u.Angstrom)

        inst_therm *= self.get_spec_throughput(wvs)

        return inst_therm

    def load_scale_aowfe(self,seeing,airmass,site_median_seeing=0.6):
        '''
        A function that returns ao wavefront errors as a function of rmag

        Args:
        path     -  Path to an ao errorbudget file [str]
        seeing   -  The current seeing conditions in arcseconds  [float]
        airmass  -  The current airmass [float]
        '''
        
        path = self.telescope.path

        #TODO: check file and indices below; I updated it to what Dimitri's code has for KPIC
        #Read in the ao_wfe
        ao_wfe=np.genfromtxt(path+'aowfe/hispec_modhis_ao_errorbudget_v3.csv', delimiter=',',skip_header=1)
        ao_rmag = ao_wfe[:,0]

        # indexes for ao_wfe from Dimitri Code
        ao_wfe_ngs=ao_wfe[:,4] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        ao_wfe_lgs=ao_wfe[:,5] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))

        return ao_rmag,ao_wfe_ngs*u.nm,ao_wfe_lgs*u.nm

    def compute_SR(self,wave):
        '''
        Compute the Strehl ratio given the wavelengths, host magnitude and telescope (which contains observing conditions)
        '''

        path = self.telescope.path

        #Get the AO WFE as a function of rmag
        ao_rmag,ao_wfe_ngs,ao_wfe_lgs = self.load_scale_aowfe(self.telescope.seeing,self.telescope.airmass,
                                            site_median_seeing=self.telescope.median_seeing)

        #We take the minimum wavefront error between natural guide star and laser guide star errors
        ao_wfe = np.min([np.interp(self.ao_mag,ao_rmag, ao_wfe_ngs),np.interp(self.ao_mag,ao_rmag, ao_wfe_lgs)]) * u.nm

        #Compute the strehl ratio
        # import pdb; pdb.set_trace()
        SR = np.array(np.exp(-(2*np.pi*ao_wfe.to(u.micron)/wave)**2))
        return SR

    def get_speckle_noise(self,separations,ao_mag,filter,wvs,star_spt,telescope,ao_mag2=None):
        '''
        Returns the contrast for a given list of separations. 

        Inputs: 
        separations     - A list of separations at which to calculate the speckle noise in arcseconds [float list length n]. Assumes these are sorted. 
        ao_mag          - The magnitude in the ao band, here assumed to be I-band
        wvs          - A list of wavelengths in microns [float length m]
        telescope    - A psisim telescope object. 

        Outputs: 
        get_speckle_noise - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]

        '''

        #TODO: decide if PIAA will be optional via flag or permanent
        #TODO: add ADC residuals effect
        #TODO: @Max, why feed "filter", "star_spt" if not used. Why feed "telescope" if already available from self.telescope?

        if self.mode != 'vfn':
            print("Warning: only 'vfn' mode has been confirmed")
        
        if self.mode == "on-axis":
           return np.ones([np.size(separations),np.size(wvs)])
        
        if np.size(wvs) < 2:
            wvs = np.array(wvs)

        if self.mode == "off-axis":
            #-- Deal with nominal KPIC mode (fiber centered on planet)
            #TODO: this was copied from HISPEC instrument. Check if any mods are needed for KPIC

            #Get the Strehl Ratio
            SR = self.compute_SR(wvs)

            p_law_kolmogorov = -11./3
            p_law_ao_coro_filter = self.p_law_dh#-p_law_kolmogorov 

            r0 = 0.55e-6/(telescope.seeing.to(u.arcsecond).value/206265) * u.m #Easiest to ditch the seeing unit here. 

            #The AO control radius in units of lambda/D
            cutoff = self.nactuators/2

            contrast = np.zeros([np.size(separations),np.size(wvs)])

            if np.size(separations) < 2:
                separations = np.array([separations.value])*separations.unit

            # #Do this by wavelength
            # for i, wv in enumerate(wvs):
            #     ang_sep_resel_in = separations/206265/u.arcsecond*telescope.diameter/wv.to(u.m) #Convert separations from arcseconds to units of lambda/D

            #     if isinstance(ang_sep_resel_in.value,float):
            #         ang_sep_resel_in = [ang_sep_resel_in]
            #         index_in = 1
            #     else:
            #         index_in = np.max(np.where(ang_sep_resel_in.value < cutoff))

            #     ang_sep_resel_step=0.1
            #     ang_sep_resel=np.arange(0,10000,ang_sep_resel_step)
            #     index=np.squeeze(np.where(ang_sep_resel == cutoff))

            #     #Dimitri to put in references to this math
            #     r0_sc = r0 * (wv/(0.55*u.micron))**(6./5)
            #     w_halo = telescope.diameter / r0_sc

            #     f_halo = np.pi*(1-SR[i])*0.488/w_halo**2 * (1+11./6*(ang_sep_resel/w_halo)**2)**(-11/6.)

            #     contrast[:,i] = np.interp(ang_sep_resel_in,ang_sep_resel,f_halo)

            #     contrast_inside = f_halo[index]*(ang_sep_resel/ang_sep_resel[index])**p_law_ao_coro_filter
            #     contrast[:,i][:index_in] = np.interp(ang_sep_resel_in[:index_in],ang_sep_resel,contrast_inside)

                #Inside the control radius, we modify the raw contrast
                # contrast[:,i][:index] = contrast[:,i][index]
            
            #Dimitri to put in references to this math
            r0_sc = r0 * (wvs/(0.55*u.micron))**(6./5)
            w_halo = telescope.diameter / r0_sc
            
            for i,sep in enumerate(separations):
                ang_sep_resel_in = sep/206265/u.arcsecond*telescope.diameter/wvs.to(u.m) #Convert separations from arcseconds to units of lambda/D

                # import pdb; pdb.set_trace()
                f_halo = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(ang_sep_resel_in/w_halo)**2)**(-11/6.)

                
                contrast_at_cutoff = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(cutoff/w_halo)**2)**(-11/6.)
                #Fill in the contrast array
                contrast[i,:] = f_halo

                biggest_ang_sep = np.abs(ang_sep_resel_in - cutoff) == np.min(np.abs(ang_sep_resel_in - cutoff))

                contrast[i][ang_sep_resel_in < cutoff] = contrast_at_cutoff[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
            # import pdb;pdb.set_trace()
            #Set the contrast inide the AO control radius
            # import pdb;pdb.set_trace()
            # if np.size(separations) < 2:
            #     contrast[ang_sep_resel_in < cutoff] = np.repeat(contrast[biggest_ang_sep,None],contrast.shape[1],axis=1)[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
            # elif np.size(wvs) < 2:
            #     contrast[ang_sep_resel_in < cutoff] = np.repeat(contrast[None,biggest_ang_sep],contrast.shape[0],axis=0)[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
            # else: 
            #     contrast[ang_sep_resel_in < cutoff] = contrast[biggest_ang_sep]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
            
            #Apply the fiber contrast gain
            contrast /= self.fiber_contrast_gain

            #Make sure nothing is greater than 1. 
            contrast[contrast>1] = 1.
            
            return contrast

        elif self.mode == "vfn":
            #-- Deal with VFN KPIC mode

            #-- Determine WFE
            #Get the AO WFE as a function of rmag
            ao_rmag,ao_wfe_ngs,ao_wfe_lgs = self.load_scale_aowfe(telescope.seeing,telescope.airmass,
                                                site_median_seeing=telescope.median_seeing)

            #We take the minimum wavefront error between natural guide star and laser guide star errors
            ao_wfe = np.min([np.interp(ao_mag,ao_rmag, ao_wfe_ngs),np.interp(ao_mag,ao_rmag, ao_wfe_lgs)]) * u.nm

            #-- Get Stellar leakage due to WFE
            #Pick the WFE coefficient based on the vortex charge. Coeff values emprically determined in simulation
            if self.vortex_charge == 1:
                wfe_coeff = 0.840       # Updated on 1/11/21 based on 6/17/19 pyWFS data
            elif self.vortex_charge == 2:
                wfe_coeff = 1.650       # Updated on 1/11/21 based on 6/17/19 pyWFS data

            #Approximate contrast from WFE
            contrast = (wfe_coeff * ao_wfe.to(u.micron) / wvs)**(2.) # * self.vortex_charge)
            
            #-- Get Stellar leakage due to Tip/Tilt Jitter
            #TODO: Use AO_mag to determine T/T residuals 
            # For now: Assume constant 2.5mas (RMS) T/T Jitter based on current PyWFS data
            ttarcsec = 2.5 * 1e-3 # convert to arcsec
            # Convert to lam/D
            ttlamD = ttarcsec / (wvs.to(u.m)/telescope.diameter * 206265)
            
            # Use leakage approx. from Ruane et. al 2019 
                # https://arxiv.org/pdf/1908.09780.pdf      Eq. 3
            ttnull = (ttlamD)**(2*self.vortex_charge)
            
            # Add to total contrast
            contrast += ttnull
                
            #-- Get Stellar leakage due to finite sized star (Geometric leakage)
              # Assumes user has already set host diameter with set_vfn_host_diameter()
              # Equation and coefficients are from Ruante et. al 2019
                # https://arxiv.org/pdf/1908.09780.pdf     fig 7c
            # Convert host_diameter to units of lambda/D
            host_diam_LoD = self.host_diameter / (wvs.to(u.m)/telescope.diameter * 206265)
            
            # Define Coefficients for geometric leakage equation
            if self.vortex_charge == 1:
                geo_coeff = 3.5
            elif self.vortex_charge == 2:
                geo_coeff = 4.2
            # Compute leakage
            geonull = (self.host_diameter / geo_coeff)**(2*self.vortex_charge)
            
            # Add to total contrast
            contrast += geonull
                
            #-- Make sure contrast has the expected output format
            #Null is independent of the planet separation; use np.tile to replicate results at all seps
            contrast = np.tile(contrast, (np.size(separations),1))

            #convert to ndarray for consistency with contrast returned by other modes
            contrast = np.array(contrast)
            
            #Make sure nothing is greater than 1. 
            contrast[contrast>1] = 1.
            
            return contrast

        else:
            raise ValueError("'%s' is a not a supported 'mode'" % (self.mode))


    def get_planet_throughput(self,separations,wvs):
        '''
        Returns the planet throughput for a given list of separations. 

        Inputs: 
        separations  - A list of separations at which to calculate the planet throughput in arcseconds [float list length n]. Assumes these are sorted. 
        wvs          - A list of wavelengths in microns [float length m]

        Outputs: 
        get_planet_throughput - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]
        '''

        #TODO: Implement PIAA improvement as flag for VFN mode
        #TODO: add in ADC residual effects
        if self.mode == 'vfn':
            path = self.telescope.path

            # load ideal VFN coupling curves
            th_vfn_ideal = np.genfromtxt(path+'VFN/Charge%d_Ideal.txt'%(self.vortex_charge), delimiter=',', skip_header=0)

            if np.size(separations) < 2:
                separations = np.array([separations.value])*separations.unit

            th_planet = np.zeros([np.size(separations),np.size(wvs)])
            for i,sep in enumerate(separations):
                ang_sep_resel_in = sep.value/(wvs.to(u.m)/self.telescope.diameter * 206265) #Convert separations from arcseconds to units of lambda/D
                th_planet[i,:] = np.interp(ang_sep_resel_in, th_vfn_ideal[:,0], th_vfn_ideal[:,1])

        elif self.mode in ['on-axis', 'off-axis']:
            # Account for planet-specific injection efficiency into fiber
                # eg. if coronagraph has separation-dependent coupling, apply here
            th_planet = 1 * np.ones([np.size(separations), np.size(wvs)]) # x1 for now since no coro.
        
        else:
            raise ValueError("'%s' is a not a supported 'mode'" % (self.mode))

        return th_planet
    def set_vfn_host_diameter(self,host_size_in_mas):
        '''
        Sets the host_diameter instance variable in units of arcsec
        
        Inputs:
        host_size_in_mas - Angular diameter of the host star in units of milliarcseconds
        '''
        self.host_diameter = host_size_in_mas * 1e-3 # convert to arcsec
