import os
import glob
import scipy.interpolate as si
import numpy as np
import astropy.units as u
import astropy.constants as constants
import astropy.io.ascii
import pysynphot as ps
import warnings
from astropy.modeling.models import BlackBody

from psisim import datadir
import psisim.instrument
from psisim.instruments.template import Instrument
import psisim.nair as nair

class kpic_phaseII(Instrument):
    '''
    An implementation of Instrument for KPIC Phase II
    '''

    def __init__(self, telescope=None, use_adc=True):
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
        self.dark_current = 0.8*u.electron/u.s#0.67*u.electron/u.s #electrons/s/pix
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
        self.nirspec_th_data = np.loadtxt(os.path.join(os.path.dirname(psisim.__file__),'data','throughput','nirspec_only_throughput.csv'),delimiter=",",skiprows=1)

        # load in fiber coupling efficiency as a function of misalignment
        fiber_data_filename = os.path.join(datadir, "smf", "keck_pupil_charge0.csv")
        self.fiber_coupling = astropy.io.ascii.read(fiber_data_filename, names=["sep", "eta"])
        self.fiber_coupling['eta'] /= self.fiber_coupling['eta'][0] # normalize to peak, since this is just the relative term

        #AO parameters
        self.nactuators = 32. - 2.0 #The number of DM actuators in one direction
        self.fiber_contrast_gain = 2. #The gain in contrast thanks to the fiber. ('off-axis' mode only)
        self.p_law_dh = -2.0 #The some power law constant Dimitri should explain. 
        self.ao_filter = 'TwoMASS-H' #Available AO filters - per Dimitri
        self.d_ao = 9.85*u.m#0.15 * u.m
        self.area_ao = np.pi*(self.d_ao/2)**2
        
        self.name = "Keck-KPIC-PhaseII"
        
        #Acceptable filters
        #TODO: check filters below (not clearly set for kpic in Dimitri code)
        self.filters = ['CFHT-Y','TwoMASS-J','TwoMASS-H','TwoMASS-K'] #Available observing filters

        # self.lsf_width = 1.0/2.35 #The linespread function width in pixels (assuming Gaussian for now)
        
        self.use_adc = use_adc

        # The current obseving properties - dynamic
        self.exposure_time = None
        self.n_exposures = None
        self.current_filter = None
        self.current_wvs = None
        self.current_dwvs = None
        self.ao_mag = None
        self.mode = None
        self.vortex_charge = None      # for vfn only
        self.host_diameter= 0.*u.mas   # for vfn only (default 0 disables geometric leak.)
        self.zenith = None # in degrees
    
    def set_observing_mode(self,exposure_time,n_exposures,sci_filter,wvs,dwvs=None, mode="vfn", vortex_charge=None, zenith=0):
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

        self.zenith = zenith
        
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
            print("Your filter {} is not in the filter list: {}\n Returning -1".format(self.current_filter,self.filters))
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

        # th_ao = {"CFHT-Y":0.60,"TwoMASS-J":0.63,"TwoMASS-H":0.66,"TwoMASS-K":0.633}.get(self.current_filter) #K-band from Nem's Feb 2020 report
        th_ao = np.interp(wvs, th_wvs, np.prod(th_data[2:13],axis=0))
        if self.mode == 'vfn':
            #th_fiu = {"CFHT-Y":0.66,"TwoMASS-J":0.68,"TwoMASS-H":0.7,"TwoMASS-K":0.6}.get(self.current_filter) #K from Dimitri (KPIC OAPS+FM+dichroic+PIAA(95%)+DM Window(90%)+ADC(90%))
            th_fiu = np.interp(wvs, th_wvs, np.prod(th_data[14:19],axis=0)*np.prod(th_data[20:29],axis=0)) # (omit coro.)
            
            #TODO: add in vortex throughput losses (separate from apodizer losses in throughput file)
            
            #th_fiber = {"CFHT-Y":0.99,"TwoMASS-J":0.99,"TwoMASS-H":0.99,"TwoMASS-K":0.98}.get(self.current_filter) #from Dimitri code
            th_fiber = np.interp(wvs, th_wvs, th_data[34]*th_data[37]) # (endfaces only)
            th_fiber = th_fiber * 0.98 # Add in constant propagation loss (0.98) for now

            #TODO: Figure out how thermal effects (in Observation.py) are modulated by fiber in VFN case and implement accordingly
            
        else:
            th_fiu = np.interp(wvs, th_wvs, np.prod(th_data[14:29], axis=0)) # KPIC throughput from budget
            #th_fcd = np.interp(wvs, th_wvs, th_data[30]) # Fiber Dynamic Coupling (need function to scale with Strehl/NGS, currently unused)
            th_fiber = np.interp(wvs, th_wvs, np.prod(th_data[31:35], axis=0)) # Fiber throughput (excluding fcd above. Also exclude prop., breakout, and second endface)
            th_fiber = th_fiber * 0.98 *  np.interp(wvs, th_wvs, th_data[37]) # Add in const. prop. loss (0.98) and second endface
            th_fiber *= self.get_dar_coupling_throughput(self, wvs)

        th_feu = 0.73 #measured for phase 1
        # th_feu = 0.89   #from Dimitri code

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

        # import matplotlib.pyplot as plt
        # # print("th_ao",th_ao)
        # print("th_feu",th_feu)
        # print("th_planet",th_planet)
        # plt.plot(wvs,th_ao,label="th_ao")
        # plt.plot(wvs,th_fiu,label="th_fiu")
        # # plt.plot(wvs,th_feu,label="th_feu")
        # plt.plot(wvs,th_fiber,label="th_fiber")
        # # plt.plot(wvs,th_planet,label="th_planet")
        # plt.plot(wvs,th_spec,label="th_spec")
        # plt.plot(wvs,SR,label="SR")
        # plt.legend()
        # plt.show()

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
        if self.current_filter == "TwoMASS-K":
            th_spec = 0.2*si.interp1d(self.nirspec_th_data[:,0], self.nirspec_th_data[:,1],bounds_error=False,fill_value=0)(wvs)
        else:
            th_spec = {"CFHT-Y":0.5,"TwoMASS-J":0.5,"TwoMASS-H":0.5,"TwoMASS-K":0.2}.get(self.current_filter,0.5)
            th_spec = th_spec*np.ones(np.shape(wvs))

        return th_spec

    def get_instrument_background(self,wvs,solidangle):
        '''
        Returns the instrument background at each wavelength in units of photons/s/Angstrom/arcsecond**2
        '''
        bb_lam = BlackBody(self.temperature,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        inst_therm = bb_lam(wvs)
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

        #Read in the ao_wfe
        ao_wfe=np.genfromtxt(path+'aowfe/hispec_modhis_ao_errorbudget_v3.csv', delimiter=',',skip_header=1)
        ao_rmag = ao_wfe[:,0]
        
        if self.mode == 'vfn':
            # For VFN+PyWFS, rescale WFE to use telemetry values from the PyWFS
            # The default table includes some errors that VFN doesn't care about
              # Based on 11/2021 telemetry, PyWFS has hit 85nm RMS WF residuals so
              # let's set that as the best value for now and then scale up from there
            ao_wfe[:,4] = ao_wfe[:,4] * 85/ao_wfe[0,4]

        # indexes for ao_wfe from Dimitri Code
        # ao_wfe_ngs=ao_wfe[:,4] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.)) # KPIC phase 2
        ao_wfe_ngs=ao_wfe[:,3] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.)) # KPIC Phase 1
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
        ao_wfe = np.min([np.interp(self.ao_mag,ao_rmag, ao_wfe_ngs).value,np.interp(self.ao_mag,ao_rmag, ao_wfe_lgs).value]) * u.nm

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

            #Dimitri to put in references to this math
            r0_sc = r0 * (wvs/(0.55*u.micron))**(6./5)
            w_halo = telescope.diameter / r0_sc
            
            for i,sep in enumerate(separations):
                ang_sep_resel_in = sep.to(u.rad).value * telescope.diameter.to(u.m)/wvs.to(u.m) #Convert separations from arcseconds to units of lambda/D

                # import pdb; pdb.set_trace()
                f_halo = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(ang_sep_resel_in/w_halo)**2)**(-11/6.)

                
                contrast_at_cutoff = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(cutoff/w_halo)**2)**(-11/6.)
                #Fill in the contrast array
                contrast[i,:] = f_halo

                biggest_ang_sep = np.abs(ang_sep_resel_in - cutoff) == np.min(np.abs(ang_sep_resel_in - cutoff))

                contrast[i][ang_sep_resel_in < cutoff] = contrast_at_cutoff[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
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
            ao_wfe = np.min([np.interp(ao_mag,ao_rmag, ao_wfe_ngs).value,np.interp(ao_mag,ao_rmag, ao_wfe_lgs).value]) * u.nm

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
            host_diam_LoD = self.host_diameter.value / (wvs.to(u.m)/telescope.diameter * 206265)
            
            # Define Coefficients for geometric leakage equation
            if self.vortex_charge == 1:
                geo_coeff = 3.5
            elif self.vortex_charge == 2:
                geo_coeff = 4.2
            # Compute leakage
            geonull = (host_diam_LoD / geo_coeff)**(2*self.vortex_charge)
            
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
        host_size_in_mas - (Astropy Quantity - u.mas) Angular diameter of the host star
        '''
        self.host_diameter = host_size_in_mas.to(u.arcsec)

    
    def get_dar_coupling_throughput(self, wvs, wvs0=None):
        """
        Gets the relative loss in fiber coupling due to DAR (normalized to 1)

        Args:
            wvs (np.array of float): wavelengths to consider
            wvs0 (float, optiona): the reference wavelength where fiber coupling is maximized. 
                                    Assumed to be the mean of the input wavelengths if not passed in

        Returns:
            np.array of float: the relative loss in throughput in fiber coupling due to DAR (normalized to 1)
        """
        if wvs0 is None:
            wvs0 = np.mean(wvs)

        # check whether we are using the ADC
        if not self.use_adc:
            n = self.telescope.get_nair(wvs)
            n0 = self.telescope.get_nair(wvs0)

            dar = np.abs(nair.compute_dar(n, n0, np.radians(self.zenith)))

            lam_d = (wvs * u.um) / (self.telescope.diameter.to(u.um)) * 206265 * 1000

            coupling_th = np.interp(dar/lam_d, self.fiber_coupling['sep'], self.fiber_coupling['eta'], right=0)
        else:
            coupling_th = 1.00 # assume perfect ADC

        return coupling_th