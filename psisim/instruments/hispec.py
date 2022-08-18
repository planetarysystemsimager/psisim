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
import psisim.telescope
from psisim.instruments.template import Instrument
import psisim.nair as nair

filter_options = {"CFHT-Y":(0.940*u.micron,1.018*u.micron,1.090*u.micron),
                    "TwoMASS-J":(1.1*u.micron,1.248*u.micron,1.360*u.micron),
                    "TwoMASS-H":(1.480*u.micron,1.633*u.micron,1.820*u.micron),
                    "TwoMASS-K":(1.950*u.micron,2.2*u.micron,2.350*u.micron)}
                    
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
        self.qe = 0.9 * u.electron/u.ph
        self.temperature_fei = 278*u.K
        self.temperature_spec = 77*u.K
        self.temperature_fiber = 278*u.K

        if telescope is None:
            self.telescope = psisim.telescope.Keck()
        else:
            self.telescope = telescope

        # self.th_data = np.genfromtxt(datadir+'/throughput/hispec_throughput_budget.csv',
        #                             skip_header=1,usecols=np.arange(5,1566),delimiter=',',missing_values='')

        # load in fiber coupling efficiency as a function of misalignment
        fiber_data_filename = os.path.join(datadir, "smf", "keck_pupil_charge0.csv")
        self.fiber_coupling = astropy.io.ascii.read(fiber_data_filename, names=["sep", "eta"])
        self.fiber_coupling['eta'] /= self.fiber_coupling['eta'][0] # normalize to peak, since this is just the relative term

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
        self.zenith = None # in degrees
    
    def set_observing_mode(self,exposure_time,n_exposures,sci_filter,wvs,dwvs=None, mode="off-axis",zenith=0):
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

        wave = wvs.to(u.um).value

        oxydized_al_data = np.genfromtxt(datadir + '/throughput/oxydized_al.csv', delimiter=',', skip_header=1)
        FSS99_ag_data = np.genfromtxt(datadir + '/throughput/FSS99_ag.csv', delimiter=',', skip_header=1)
        opticoat_ag_data = np.genfromtxt(datadir + '/throughput/opticoat_ag.csv', delimiter=',', skip_header=1)
        kao_shwfsdichroic_data = np.genfromtxt(datadir + '/throughput/kao_shwfsdichroic.csv', delimiter=',', skip_header=1)

        dust_th = 0.97
        K1_th = np.interp(wave,oxydized_al_data[:,0],oxydized_al_data[:,1]) * dust_th
        K2_th = np.interp(wave,FSS99_ag_data[:,0],FSS99_ag_data[:,1]) * dust_th
        K3_th = K1_th
        TTM_th = np.interp(wave,opticoat_ag_data[:,0],opticoat_ag_data[:,1]) * dust_th
        OAP1_th = TTM_th
        DM_th = K2_th
        OAP2_th = TTM_th
        SHWFSDICH_th = np.interp(wave,kao_shwfsdichroic_data[:,0],kao_shwfsdichroic_data[:,1]) * dust_th
        PICKOFF_th = TTM_th
        KAO_th = K1_th * K2_th * K3_th * TTM_th * OAP1_th * DM_th * OAP2_th * SHWFSDICH_th * PICKOFF_th
        KAO_em = 1-KAO_th

        protected_au_data = np.genfromtxt(datadir + '/throughput/protected_au.csv', delimiter=',', skip_header=1)

        PWFSDICH_th = 0.94
        FM1_th = np.interp(wave,protected_au_data[:,0],protected_au_data[:,1])
        FOAP1_th = FM1_th
        ADC_th = 0.9
        TRACDICH_th = 0.94
        redbluedich_th = 0.94
        PIAA_th = 0.99**4
        injectionlens_th = 0.955
        HISPECFEI_th = PWFSDICH_th * FM1_th * FOAP1_th * ADC_th * TRACDICH_th * redbluedich_th * PIAA_th*injectionlens_th
        HISPECFEI_em = 1.0 - HISPECFEI_th

        hispec_fiber_data = np.genfromtxt(datadir + '/throughput/hispec_yjhkfiber.csv', delimiter=',', skip_header=1)

        fiberin_th = 0.99
        fiberprop_th = np.interp(wave, hispec_fiber_data[:,0],hispec_fiber_data[:,1])
        fiber_break = 0.98**3
        fiberout_th = 0.99
        fiber_th = fiberin_th * fiberprop_th * fiber_break * fiberout_th

        hispec_th = self.get_spec_throughput(wvs)

        if planet_flag:
            # Get separation-dependent planet throughput
            th_planet = self.get_planet_throughput(planet_sep, wvs)[0]
        else:
            # Set to 1 to ignore separation effects
            th_planet = 1

        strehl = self.compute_SR(wvs)
        diff_keck = 0.6 * 1.4
        coupling = strehl * diff_keck

        th_inst = KAO_th * HISPECFEI_th * fiber_th * hispec_th * coupling * th_planet
        # th_inst = th_ao * th_fiu * th_fiber * th_planet * th_spec * SR

        return th_inst
    
    # def get_inst_emissivity(self,wvs):
    #     '''
    #     The instrument emissivity
    #     '''
    #
    #     # TODO: do we want to use the throughput w/ or w/o the planet losses?
    #     return (1-self.get_inst_throughput(wvs))

    def get_spec_throughput(self, wvs):
        '''
        The throughput of the spectrograph - different than the throughput of the inst that you get in self.get_inst_throughput. 
        self.get_inst_throughput includes everything, whereas this is just the spectrograph. 
        '''

        wave = wvs.to(u.um).value

        protected_au_data = np.genfromtxt(datadir + '/throughput/protected_au.csv', delimiter=',', skip_header=1)
        echelle_data = np.genfromtxt(datadir + '/throughput/echelle.csv', delimiter=',', skip_header=1)
        cx_data = np.genfromtxt(datadir + '/throughput/cx.csv', delimiter=',', skip_header=1)

        TMA1_th = (np.interp(wave, protected_au_data[:,0],protected_au_data[:,1]))**3
        cold_stop = 0.94
        echelle_th = 0.75 * np.interp(wave,echelle_data[:,0],echelle_data[:,1])#80% max efficiency
        cx_th = np.interp(wave,cx_data[:,0],cx_data[:,1])
        FM1_th = np.interp(wave, protected_au_data[:,0],protected_au_data[:,1])
        TMA2_th = TMA1_th
        hispec_th = TMA1_th* cold_stop * echelle_th * cx_th *FM1_th * TMA2_th

        return hispec_th

    def get_instrument_background(self,wvs,solidangle):
        '''
        Returns the instrument background at each wavelength in units of photons/s/Angstrom/arcsecond**2
        '''

        wave = wvs.to(u.um).value

        oxydized_al_data = np.genfromtxt(datadir + '/throughput/oxydized_al.csv', delimiter=',', skip_header=1)
        FSS99_ag_data = np.genfromtxt(datadir + '/throughput/FSS99_ag.csv', delimiter=',', skip_header=1)
        opticoat_ag_data = np.genfromtxt(datadir + '/throughput/opticoat_ag.csv', delimiter=',', skip_header=1)
        kao_shwfsdichroic_data = np.genfromtxt(datadir + '/throughput/kao_shwfsdichroic.csv', delimiter=',', skip_header=1)

        dust_th = 0.97
        K1_th = np.interp(wave,oxydized_al_data[:,0],oxydized_al_data[:,1]) * dust_th
        K2_th = np.interp(wave,FSS99_ag_data[:,0],FSS99_ag_data[:,1]) * dust_th
        K3_th = K1_th
        TTM_th = np.interp(wave,opticoat_ag_data[:,0],opticoat_ag_data[:,1]) * dust_th
        OAP1_th = TTM_th
        DM_th = K2_th
        OAP2_th = TTM_th
        SHWFSDICH_th = np.interp(wave,kao_shwfsdichroic_data[:,0],kao_shwfsdichroic_data[:,1]) * dust_th
        PICKOFF_th = TTM_th
        KAO_th = K1_th * K2_th * K3_th * TTM_th * OAP1_th * DM_th * OAP2_th * SHWFSDICH_th * PICKOFF_th
        KAO_em = 1-KAO_th

        protected_au_data = np.genfromtxt(datadir + '/throughput/protected_au.csv', delimiter=',', skip_header=1)

        PWFSDICH_th = 0.94
        FM1_th = np.interp(wave,protected_au_data[:,0],protected_au_data[:,1])
        FOAP1_th = FM1_th
        ADC_th = 0.9
        TRACDICH_th = 0.94
        redbluedich_th = 0.94
        PIAA_th = 0.99**4
        injectionlens_th = 0.955
        HISPECFEI_th = PWFSDICH_th * FM1_th * FOAP1_th * ADC_th * TRACDICH_th * redbluedich_th * PIAA_th*injectionlens_th

        hispec_fiber_data = np.genfromtxt(datadir + '/throughput/hispec_yjhkfiber.csv', delimiter=',', skip_header=1)

        fiberin_th = 0.99
        fiberprop_th = np.interp(wave, hispec_fiber_data[:,0],hispec_fiber_data[:,1])
        fiber_break = 0.98**3
        fiberout_th = 0.99
        fiber_th = fiberin_th * fiberprop_th * fiber_break * fiberout_th

        hispec_th = self.get_spec_throughput(wvs)

        bb_lam_fei = BlackBody(self.temperature_fei,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        fei_therm = bb_lam_fei(wvs)
        fei_therm *= solidangle
        fei_therm = fei_therm.to(u.ph/(u.micron * u.s * u.cm**2),equivalencies=u.spectral_density(wvs)) * self.telescope.collecting_area
        fei_therm *= 1-KAO_em*HISPECFEI_th
        fei_therm *= fiber_th
        fei_therm *= hispec_th

        bb_lam_fiber = BlackBody(self.temperature_fiber, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
        fiber_therm = bb_lam_fiber(wvs)
        fiber_therm *= solidangle
        fiber_therm = fiber_therm.to(u.ph / (u.micron * u.s * u.cm ** 2),
                                     equivalencies=u.spectral_density(wvs)) * self.telescope.collecting_area
        fiber_therm *= 1-fiber_th
        fiber_therm *= hispec_th

        bb_lam_spec = BlackBody(self.temperature_spec,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        spec_therm = bb_lam_spec(wvs)
        spec_therm *= solidangle
        spec_therm = spec_therm.to(u.ph / (u.micron * u.s * u.cm ** 2),
                                   equivalencies=u.spectral_density(wvs)) * self.telescope.collecting_area
        spec_therm *= 1-hispec_th

        return fei_therm + fiber_therm + spec_therm

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
        ao_wfe = np.min([np.interp(self.ao_mag,ao_rmag, ao_wfe_ngs).value,np.interp(self.ao_mag,ao_rmag, ao_wfe_lgs).value]) * u.nm

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
            ang_sep_resel_in = sep.to(u.rad).value * telescope.diameter.to(u.m)/wvs.to(u.m) #Convert separations from arcseconds to units of lambda/D

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

        n = self.telescope.get_nair(wvs)
        n0 = self.telescope.get_nair(wvs0)

        dar = np.abs(nair.compute_dar(n, n0, np.radians(self.zenith)))

        lam_d = (wvs * u.um) / (self.telescope.diameter.to(u.um)) * 206265 * 1000

        coupling_th = np.interp(dar/lam_d, self.fiber_coupling['sep'], self.fiber_coupling['eta'], right=0)

        return coupling_th
