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

    def __init__(self):
        pass

    def get_inst_throughput(self, wvs):
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
    def __init__(self):
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
    def __init__(self):
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

    def __init__(self,telescope):
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
        self.dark_current = 0.02*u.electron/u.s * self.spatial_sampling #electrons/s/wavelength channel
        self.det_welldepth = 1e5 *u.photon
        self.det_linearity = 0.66*self.det_welldepth
        self.qe = 0.95 * u.electron/u.ph
        self.temperature = 276*u.K

        self.telescope = telescope

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

        if mode != "off-axis" or mode != "on-axis":
            raise ValueError("'mode' must be 'off-axis' or 'on-axis'")

        self.mode = mode

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

    def get_inst_throughput(self,wvs):
        '''
        To be filled in
        '''

        if self.current_filter not in self.filters:
            raise ValueError("Your current filter of {} is not in the available filters: {}".format(self.current_filter,self.filters))

        #Will do this by band for now. 
        th_ao = {"CFHT-Y":0.60,"TwoMASS-J":0.63,"TwoMASS-H":0.66,"TwoMASS-K":0.73}.get(self.current_filter) #Ao throughput measured by P. Wizinowich
        th_fiu = {"CFHT-Y":0.66,"TwoMASS-J":0.68,"TwoMASS-H":0.7,"TwoMASS-K":0.72}.get(self.current_filter) #KPIC OAPS+FM+dichroic
        th_fiu_insertion = 0.87 * 0.99**4 * 0.98**2 #assumes PIAA optics and AR coating on PIAA and fiber end - Not yet wavelength dependent
        th_feu = 1.0 
        th_fiber = {"CFHT-Y":0.99,"TwoMASS-J":0.99,"TwoMASS-H":0.99,"TwoMASS-K":0.9}.get(self.current_filter) #standard or ZBLAN 50-meter fibers 

        SR = self.compute_SR(wvs)

        th_spec = self.get_spec_throughput(wvs)
        th_inst = th_ao * th_fiu * th_fiu_insertion * th_feu * th_fiber * th_spec * np.ones(wvs.shape) * SR

        return th_inst
    
    def get_inst_emissivity(self,wvs):
        '''
        The instrument emissivity
        '''

        return (1-self.get_inst_throughput(wvs))

    def get_spec_throughput(self, wvs):
        '''
        The throughput of the spectrograph - different than the throughput of the inst that you get in self.get_inst_throughput. 
        self.get_inst_throughput includes everything, whereas this is just the spectrograph. 
        '''

        th_spec = {"CFHT-Y":0.5,"TwoMASS-J":0.5,"TwoMASS-H":0.5,"TwoMASS-K":0.5}.get(self.current_filter,0.5)

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

        if self.mode == "on-axis"
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

class modhis(hispec):
    '''
    An implementaion of Instrument for Modhis
    '''
    def __init__(self):
        super(hispec,self).init()
    
        self.temperature = 243*u.K

        print("MODHIS is horribly incomplete at this point")

    #To be continued. 

