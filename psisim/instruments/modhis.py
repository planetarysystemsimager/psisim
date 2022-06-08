import os
import glob
import scipy.interpolate as si
import numpy as np
import astropy.units as u
import astropy.constants as constants
import pysynphot as ps
import warnings
from astropy.modeling.models import BlackBody

import psisim
from psisim.instruments.hispec import hispec


class modhis(hispec):
    '''
    An implementaion of Instrument for Modhis
    '''
    def __init__(self,telescope=None):
        super(modhis,self).__init__()
    
        print("MODHIS is mostly untested at this point")
    
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
        self.temperature = 243*u.K

        if telescope is None:
            self.telescope = psisim.telescope.TMT()
        else:
            self.telescope = telescope

        # TODO: Create a budget file for MODHIS and load that here
        #self.th_data = np.genfromtxt(self.telescope.path+'/throughput/hispec_throughput_budget.csv',
        #                            skip_header=1,usecols=np.arange(5,166),delimiter=',',missing_values='')

        #AO parameters
        self.nactuators = 60. - 2.0 #The number of DM actuators in one direction
        self.fiber_contrast_gain = 10. #Contrast gain due to fiber ('off-axis' mode only)
        self.p_law_dh = -2.0 #The some power law constant Dimitri should explain. 
        self.ao_filter = 'bessell-I'
        self.d_ao = 0.3 * u.m
        self.area_ao = np.pi*(self.d_ao/2)**2
        
        self.name = "TMT-MODHIS"
        
        #Acceptable observing filters
        # TODO: add whatever remaining filters MODHIS will have
        self.filters = ["CFHT-Y","TwoMASS-J",'TwoMASS-H','TwoMASS-K']         

        # The current obseving properties - dynamic
        self.exposure_time = None
        self.n_exposures = None
        self.current_filter = None
        self.current_wvs = None
        self.current_dwvs = None
        self.ao_mag = None
        self.mode = None
        self.vortex_charge = None      # for vfn only
        self.host_diameter= 0.0*u.mas  # for vfn only (default 0 disables geometric leak.)
        self.ttarcsec = (2.0*u.mas).to(u.arcsec)   # for vfn only (assume 2mas jitter for MODHIS by default)
        
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
        
        # TODO: test other modes and remove this warning
        if mode != 'vfn':
            warnings.warn("Modes other than 'vfn' are not tested at the moment")

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

        filter_options = {"CFHT-Y":(0.940*u.micron,0.965*u.micron,1.090*u.micron),
                          "TwoMASS-J":(1.1*u.micron,1.248*u.micron,1.360*u.micron),
                          "TwoMASS-H":(1.480*u.micron,1.633*u.micron,1.820*u.micron),
                          "TwoMASS-K":(1.950*u.micron,2.2*u.micron,2.350*u.micron)}

        return filter_options.get(filter_name)

    def get_wavelength_range(self):
        '''
        A function that returns a wavelength array with the given R and wavelength sampling. 
        '''
        
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

        # TODO: Create a proper budget file for MODHIS and use that instead of constants
  
        th_ao = {"CFHT-Y":0.8,"TwoMASS-J":0.8,"TwoMASS-H":0.8,"TwoMASS-K":0.8}.get(self.current_filter)  #From Dimitri code
                  
        if self.mode == 'vfn':
            th_fiu = {"CFHT-Y":0.66*0.96,"TwoMASS-J":0.68*0.96,"TwoMASS-H":0.7*0.96,"TwoMASS-K":0.72*0.96}.get(self.current_filter) #From Dimitri Code(KPIC OAPS+FM+dichroic + PIAA)
            
            #TODO: Add in coro. Check about ADC and DM window.
            
            th_fiber = {"CFHT-Y":0.99*0.96,"TwoMASS-J":0.99*0.96,"TwoMASS-H":0.99*0.96,"TwoMASS-K":0.9*0.96}.get(self.current_filter) #From Dimitri code(prop loss, 98% per endface)

        else:
            th_fiu = {"CFHT-Y":0.66*0.96,"TwoMASS-J":0.68*0.96,"TwoMASS-H":0.7*0.96,"TwoMASS-K":0.72*0.96}.get(self.current_filter) #From Dimitri Code(KPIC OAPS+FM+dichroic + PIAA)
            
            #TODO: Check about ADC and DM window.
                  
            th_fiber = {"CFHT-Y":0.87*0.99*0.96,"TwoMASS-J":0.87*0.99*0.96,"TwoMASS-H":0.87*0.99*0.96,"TwoMASS-K":0.87*0.9*0.96}.get(self.current_filter) #From Dimitri code(87% insert. assuming PIAA, prop loss, 98% per endface)

        th_feu = 1.0   #from Dimitri code

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

        th_spec = {"CFHT-Y":0.5,"TwoMASS-J":0.5,"TwoMASS-H":0.5,"TwoMASS-K":0.5}.get(self.current_filter,0.5) #From Dimitri code

        return th_spec*np.ones(np.shape(wvs))

    def get_instrument_background(self,wvs,solidangle):
        '''
        Returns the instrument background at each wavelength in units of photons/s/Angstrom/arcsecond**2
        '''
        bb_lam = BlackBody(self.temperature,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        inst_therm = bb_lam(wvs)
        inst_therm *= solidangle
        inst_therm = inst_therm.to(u.ph/(u.micron * u.s * u.cm**2),equivalencies=u.spectral_density(wvs)) * self.area_ao.to(u.cm**2)
        inst_therm *= self.get_inst_emissivity(wvs)
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
            # For VFN, rescale WFE to use telemetry values from the PyWFS
            # The default table includes some errors that VFN doesn't care about
              # Based on 11/2021 telemetry, PyWFS has hit 85nm RMS WF residuals so
              # let's set that as the best value for now and then scale up from there
            ao_wfe[:,6] = ao_wfe[:,6] * 85/ao_wfe[0,6]

        # indexes for ao_wfe from Dimitri Code
        ao_wfe_ngs=ao_wfe[:,6] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
        ao_wfe_lgs=ao_wfe[:,7] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))

        return ao_rmag,ao_wfe_ngs*u.nm,ao_wfe_lgs*u.nm

    def compute_SR(self,wave):
        '''
        Compute the Strehl ratio given the wavelengths, host magnitude and telescope (which contains observing conditions)
        '''

        path = self.telescope.path

        #Get the AO WFE as a function of rmag
        ao_rmag,ao_wfe_ngs,ao_wfe_lgs = self.load_scale_aowfe(self.telescope.seeing,self.telescope.airmass,
                                            site_median_seeing=self.telescope.median_seeing)

        # Take minimum wavefront error between natural guide star and laser guide star
        ao_wfe = np.min([np.interp(self.ao_mag,ao_rmag, ao_wfe_ngs).value,np.interp(self.ao_mag,ao_rmag, ao_wfe_lgs).value]) * u.nm

        #Compute the strehl ratio
        SR = np.array(np.exp(-(2*np.pi*ao_wfe.to(u.micron)/wave)**2))
        return SR

    def get_speckle_noise(self,separations,ao_mag,filter,wvs,star_spt,telescope,ao_mag2=None):
        '''
        Returns the contrast for a given list of separations. 

        Inputs: 
        separations  - A list of separations at which to calculate the speckle noise in arcseconds [float list length n]. Assumes these are sorted. 
        ao_mag       - The magnitude in the ao band, here assumed to be I-band
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
            #-- Deal with nominal MODHIS mode (fiber centered on planet)
            #TODO: this was copied from KPIC instrument. Check if any mods are needed for MODHIS

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
                ang_sep_resel_in = sep/206265/u.arcsecond*telescope.diameter/wvs.to(u.m) #Convert separtiona from arcsec to units of lam/D

                f_halo = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(ang_sep_resel_in/w_halo)**2)**(-11/6.)

                contrast_at_cutoff = np.pi*(1-SR)*0.488/w_halo**2 * (1+11./6*(cutoff/w_halo)**2)**(-11/6.)
                #Fill in the contrast array
                contrast[i,:] = f_halo

                biggest_ang_sep = np.abs(ang_sep_resel_in - cutoff) == np.min(np.abs(ang_sep_resel_in - cutoff))

                contrast[i][ang_sep_resel_in < cutoff] = contrast_at_cutoff[ang_sep_resel_in < cutoff]*(ang_sep_resel_in[ang_sep_resel_in < cutoff]/cutoff)**p_law_ao_coro_filter
            
            #Apply the fiber contrast gain
            contrast /= self.fiber_contrast_gain

            #Make sure nothing is greater than 1. 
            contrast[contrast>1] = 1.
            
            return contrast

        elif self.mode == "vfn":
            #-- Deal with VFN mode

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
            # Convert jitter to lam/D
            ttlamD = self.ttarcsec.value / (wvs.to(u.m)/telescope.diameter * 206265)
            
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
                ang_sep_resel_in = sep.value/(wvs.to(u.m)/self.telescope.diameter * 206265) #Convert seps from arcsec to units of lam/D
                th_planet[i,:] = np.interp(ang_sep_resel_in, th_vfn_ideal[:,0], th_vfn_ideal[:,1])

        elif self.mode in ['on-axis', 'off-axis']:
            # TODO: for on-axis mode, can use VFN charge-0 coupling curve
            
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
    
    def set_vfn_tt_jitter(self,tt_jitter_in_mas):
        '''
        Sets the ttarcsec instance variable in units of arcsec. This jitter is used
          to determine the VFN null depth tip/tilt leakage term.
        
        Inputs:
        tt_jitter_in_mas - (Astropy Quantity - u.mas) RMS TT jitter for the system
        '''
        self.ttarcsec = tt_jitter_in_mas.to(u.arcsec)
