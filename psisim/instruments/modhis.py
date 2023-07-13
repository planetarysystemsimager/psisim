import os
import glob
import scipy.interpolate as si
from scipy import interpolate
import numpy as np
import astropy.units as u
import astropy.constants as constants
import pysynphot as ps
import warnings
from astropy.modeling.models import BlackBody
import pandas as pd
from scipy.integrate import trapz
from astropy.io import fits

import psisim
from psisim.instruments.hispec import hispec
from psisim import datadir


# +
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
        self.qe = 0.9 * u.electron/u.ph
        self.temperature_fei = 243*u.K
        self.temperature_spec = 77*u.K
        self.temperature_fiber = 276*u.K

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
        self.area_tel = 655 * u.m**2
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
        self.ao_ho_wfe = None
        self.ao_tt_dynamic = None
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
                  


####

    def get_inst_throughput(self,wvs,planet_flag=False,planet_sep=None):
        '''
        Reads instrument throughput from budget file and interpolates to given wavelengths
        When reading from budget file, accounts for pertinent lines depending on the instrument mode

        Kwargs:
        planet_flag     - Boolean denoting if planet-specific separation losses should be accounted for [default False]
        planet_sep      - [in arcsecond] Float of angular separation at which to determine planet throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        Based on function "instrument" in specsim (https://github.com/ashbake/specsim/blob/main/utils/load_inputs.py)
        (Currently, track camera is not supported, so "so.ao.pywfs_dichroic" is not needed)
        Based on function "get_base_throughput" in specsim (https://github.com/ashbake/specsim/blob/main/utils/throughput_tools.py)
        
        '''

        if self.current_filter not in self.filters:
            raise ValueError("Your current filter of {} is not in the available filters: {}".format(self.current_filter,self.filters))

        # TODO: Create a proper budget file for MODHIS and use that instead of constants

        #th_ao = {"CFHT-Y":0.8,"TwoMASS-J":0.8,"TwoMASS-H":0.8,"TwoMASS-K":0.8}.get(self.current_filter)  #From Dimitri code

        '''          
        if self.mode == 'vfn':
            th_fiu = {"CFHT-Y":0.66*0.96,"TwoMASS-J":0.68*0.96,"TwoMASS-H":0.7*0.96,"TwoMASS-K":0.72*0.96}.get(self.current_filter) #From Dimitri Code(KPIC OAPS+FM+dichroic + PIAA)
            
            #TODO: Add in coro. Check about ADC and DM window.
            
            th_fiber = {"CFHT-Y":0.99*0.96,"TwoMASS-J":0.99*0.96,"TwoMASS-H":0.99*0.96,"TwoMASS-K":0.9*0.96}.get(self.current_filter) #From Dimitri code(prop loss, 98% per endface)
        '''
        #th_fei = self.get_fei_throughput(wvs)
        #th_fiu = {"CFHT-Y":0.66*0.96,"TwoMASS-J":0.68*0.96,"TwoMASS-H":0.7*0.96,"TwoMASS-K":0.72*0.96}.get(self.current_filter) #From Dimitri Code(KPIC OAPS+FM+dichroic + PIAA)

        #TODO: Check about ADC and DM window.
        #th_fiber = self.get_fiber_throughput(wvs)
        #th_fiber = {"CFHT-Y":0.87*0.99*0.96,"TwoMASS-J":0.87*0.99*0.96,"TwoMASS-H":0.87*0.99*0.96,"TwoMASS-K":0.87*0.9*0.96}.get(self.current_filter) #From Dimitri code(87% insert. assuming PIAA, prop loss, 98% per endface)
        th_ao = self.get_ao_throughput(wvs)
        th_feicom = self.get_feicom_throughput(wvs)
        th_feired = self.get_feired_throughput(wvs)
        th_feiblue = self.get_feiblue_throughput(wvs)
        th_fibred = self.get_fibred_throughput(wvs)
        th_fibblue = self.get_fibblue_throughput(wvs)
        th_rspec = self.get_rspec_throughput(wvs)
        th_bspec = self.get_bspec_throughput(wvs)
        
        base_throughput_red = th_ao * th_feicom * th_feired * th_fibred * th_rspec 
        base_throughput_blue = th_ao * th_feicom * th_feiblue * th_fibblue * th_bspec 
        red_range_index = np.where(wvs.value>1.4)
        blue_range_index = np.where((wvs.value<=1.4))
        
        base_throughput = np.concatenate([base_throughput_blue[blue_range_index],base_throughput_red[red_range_index]])
        coupling = self.pick_coupling(w=wvs)
        if self.mode == 'vfn':
            coupling = np.ones(coupling.shape)
        if planet_flag:
            # Get separation-dependent planet throughput
            th_planet = self.get_planet_throughput(planet_sep, wvs)[0]
        else:
            # Set to 1 to ignore separation effects
            th_planet = 1

        #TODO: figure out if SR is needed for VFN (thinking it's not)

        th_inst =base_throughput * th_planet * coupling

        return th_inst
    
    def get_ao_throughput(self,wvs):
        """
        input: wvs = wavelength in um
    
        output: AO throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS ao throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 

        """
        ao_data =np.genfromtxt(datadir + '/throughput/ao_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_ao=interpolate.interp1d(ao_data[:, 0], ao_data[:, 1], bounds_error=False,fill_value=0)
        ao_th = f_ao(wvs.value)
        
        return ao_th

    def get_feicom_throughput(self,wvs):
    
        """
        input: wvs = wavelength in um
    
        output: FEI common throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS FEI common throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        feicom_data =np.genfromtxt(datadir + '/throughput/feicom_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_feicom=interpolate.interp1d(feicom_data[:, 0], feicom_data[:, 1], bounds_error=False,fill_value=0)
        feicom_th = f_feicom(wvs.value)
        
        return feicom_th
    
    def get_feired_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: FEI red throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS FEI red throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        """
        feired_data =np.genfromtxt(datadir + '/throughput/feired_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_feired=interpolate.interp1d(feired_data[:, 0], feired_data[:, 1], bounds_error=False,fill_value=0)
        feired_th = f_feired(wvs.value)
        
        return feired_th
    
    def get_feiblue_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: FEI blue throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS FEI blue throuhhput, source: Garreth Ruane, date: Jun 29, 2023.      
        
        """
        feiblue_data =np.genfromtxt(datadir + '/throughput/feiblue_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_feiblue=interpolate.interp1d(feiblue_data[:, 0], feiblue_data[:, 1], bounds_error=False,fill_value=0)
        feiblue_th = f_feiblue(wvs.value)
        
        return feiblue_th
    
    def get_fibred_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: FIB red throughput

        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS FIB red throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        fibred_data =np.genfromtxt(datadir + '/throughput/fibred_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_fibred=interpolate.interp1d(fibred_data[:, 0], fibred_data[:, 1], bounds_error=False,fill_value=0)
        fibred_th = f_fibred(wvs.value)
        
        return fibred_th
    
    def get_fibblue_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: FIB blue throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS FIB red throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        fibblue_data =np.genfromtxt(datadir + '/throughput/fibblue_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_fibblue=interpolate.interp1d(fibblue_data[:, 0], fibblue_data[:, 1], bounds_error=False,fill_value=0)
        fibblue_th = f_fibblue(wvs.value)
        
        return fibblue_th
    
    def get_bspec_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: BSPEC throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS BSPEC throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        bspec_data =np.genfromtxt(datadir + '/throughput/bspec_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_bspec=interpolate.interp1d(bspec_data[:, 0], bspec_data[:, 1], bounds_error=False,fill_value=0)
        bspec_th = f_bspec(wvs.value)
        
        return bspec_th
    
    def get_rspec_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: RSPEC throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official MODHIS RSPEC throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        rspec_data =np.genfromtxt(datadir + '/throughput/rspec_throughput_modhis.csv', delimiter=',', skip_header=1)
        f_rspec=interpolate.interp1d(rspec_data[:, 0], rspec_data[:, 1], bounds_error=False,fill_value=0)
        rspec_th = f_rspec(wvs.value)
        
        return rspec_th
    
    def pick_coupling(self,w,ttStatic=0,LO=10,PLon=1,piaa_boost=1.3,points=None,values=None,atm=0,adc=0):
        """
        select correct coupling file
        to do:implement interpolation of coupling files instead of rounding variables
        
        Huihao Zhang (zhang.12043@osu.edu)
        
        Based on function "pick_coupling" in specsim (https://github.com/ashbake/specsim/blob/main/utils/throughput_tools.py)

        """
        out = self.grid_interp_coupling(int(PLon),atm=int(atm),adc=int(adc))
        grid_points, grid_values = out[0],out[1:] #if PL, three values
        points=grid_points
        values=grid_values
        dynwfe= self.ao_ho_wfe
        ttDynamic= self.ao_tt_dynamic

        PLon = int(PLon)
        waves = (w.value).copy()
        if np.min(waves) > 10:
            waves/=1000 # convert nm to um

        # check range of each variable
        if ttStatic > 10 or ttStatic < 0:
            raise ValueError('ttStatic is out of range, 0-10')
        if ttDynamic > 20 or ttDynamic < 0:
            raise ValueError('ttDynamic is out of range, 0-10')
        if LO > 100 or LO < 0:
            raise ValueError('LO is out of range,0-100')
        if PLon >1:
            raise ValueError('PL is out of range')

        if PLon:
            values_1,values_2,values_3 = values
            point = (LO,ttStatic,ttDynamic,waves)
            mode1 = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0) # see example https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
            mode2 = interpolate.interpn(points, values_2, point,bounds_error=False,fill_value=0) 
            mode3 = interpolate.interpn(points, values_3, point,bounds_error=False,fill_value=0) 

            #PLwav,PLdat = load_photonic_lantern() #transfer matrices input mode--> each SMF
            #mat = PLdat[10] # use middle one for now
            #test1 = mode1 * mat[0,0]  + mode2*mat[1,0] + mode3*mat[2,0]
            #test2 = mode1 * mat[0,1]  + mode2*mat[1,1] + mode3*mat[2,1]
            #test3 = mode1 * mat[2,2]  + mode2*mat[1,2] + mode3*mat[2,2]
            # apply only to YJ or make matrix diagonal for HK..map onto same wavelength grid somehow
            # get coupling
            losses = np.ones_like(mode1) # due to PL imperfection
            losses[np.where(waves< 1.400)[0]] = 0.95 # only apply to y band
            raw_coupling = losses*(mode1+mode2+mode3) # do dumb things for now #0.95 is a recombination loss term 
        else:
            values_1= values
            points, values_1 = grid_interp_coupling(PLon)
            point = (LO,ttStatic,ttDynamic,waves)
            raw_coupling = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0)

        if np.max(waves) < 10:
            waves*=1000 # nm to match dynwfe

        ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
        coupling  = raw_coupling * piaa_boost * ho_strehl

        return coupling
 
    def grid_interp_coupling(self,PLon=1,atm=1,adc=1):
        """
        interpolate coupling files over their various parameters
        PLon: 0 or 1, whether PL is on or not
        path: data path to coupling files
        atm: 0 or 1 - whether gary at atm turned on in sims
        adc: 0 or 1 - whether gary had adc included in sims
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        Based on function "grid_interp_coupling" in specsim (https://github.com/ashbake/specsim/blob/main/utils/throughput_tools.py)
        
        """
        path= datadir+'/coupling/'
        LOs = np.arange(0,125,25)
        ttStatics = np.arange(11)
        ttDynamics = np.arange(0,20.5,0.5)

        filename_skeleton = 'couplingEff_atm%s_adc%s_PL%s_defoc25nmRMS_LO%snmRMS_ttStatic%smas_ttDynamic%smasRMS.csv'

        # to dfine values, must open up each file. not sure if can deal w/ wavelength
        values_1 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
        values_2 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))
        values_3 = np.zeros((len(LOs),len(ttStatics),len(ttDynamics),100))  
        for i,LO in enumerate(LOs):
            for j,ttStatic in enumerate(ttStatics):
                for k,ttDynamic in enumerate(ttDynamics):
                    if round(ttDynamic)==ttDynamic: ttDynamic=round(ttDynamic)
                    f = pd.read_csv(path+filename_skeleton%(atm,adc,PLon,LO,ttStatic,ttDynamic))
                    if PLon:
                        values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?
                        values_2[i,j,k,:]=f['coupling_eff_mode2'] #what to fill here?
                        values_3[i,j,k,:]=f['coupling_eff_mode3'] #what to fill here?
                    else:
                        values_1[i,j,k,:]=f['coupling_eff_mode1'] #what to fill here?

                    #values_hk[i,j,k]=f['coupling_eff_mode1'][50] #what to fill here?

        points = (LOs, ttStatics, ttDynamics,f['wavelength_um'].values)

        if PLon:
            return points,values_1,values_2,values_3
        else:
            return points,values_1
        


    
    def integrate(self,x,y):
        """
        Integrate y wrt x
        
        Huihao Zhang (zhang.12043@osu.edu)
        
        Based on function "integrate" in specsim (https://github.com/ashbake/specsim/blob/main/utils/functions.py)
        
        """
        return trapz(y,x=x)
    
    def load_phoenix(self,stelname,wav_start=750,wav_end=780):
        """
        load fits file stelname with stellar spectrum from phoenix 
        http://phoenix.astro.physik.uni-goettingen.de/?page_id=15

        return subarray 

        wav_start, wav_end specified in nm

        convert s from egs/s/cm2/cm to phot/cm2/s/nm using
        https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        Based on function "load_phoenix" in specsim (https://github.com/ashbake/specsim/blob/main/utils/load_inputs.py)
        
        """

        # conversion factor

        f = fits.open(stelname)
        spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
        f.close()

        path = stelname.split('/')
        wave_file = '/' + os.path.join(*stelname.split('/')[0:-1]) + '/' + \
                        'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits' #assume wave in same folder
        f = fits.open(wave_file)
        lam = f[0].data # angstroms
        f.close()

        # Convert
        conversion_factor = 5.03*10**7 * lam #lam in angstrom here
        spec *= conversion_factor # phot/cm2/s/angstrom

        # Take subarray requested
        isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

        # Convert 
        return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm
    
    def load_sonora(self,stelname,wav_start=750,wav_end=780):
        """
        load sonora model file

        return subarray 

        wav_start, wav_end specified in nm

        convert s from erg/cm2/s/Hz to phot/cm2/s/nm using
        https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf

        wavelenght loaded is microns high to low
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        Based on function "load_sonora" in specsim (https://github.com/ashbake/specsim/blob/main/utils/load_inputs.py)
        
        """
        f = np.loadtxt(stelname,skiprows=2)

        lam  = 10000* f[:,0][::-1] #microns to angstroms, needed for conversiosn
        spec = f[:,1][::-1] # erg/cm2/s/Hz

        spec *= 3e18/(lam**2)# convert spec to erg/cm2/s/angstrom

        conversion_factor = 5.03*10**7 * lam #lam in angstrom here
        spec *= conversion_factor # phot/cm2/s/angstrom

        isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

        return lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm (my fave)
    
    def get_inst_emissivity(self,wvs):
        '''
        The instrument emissivity
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        '''
        # SR should not be included in emissivity, so will divide from throughput
        coupling = self.pick_coupling(w=wvs)
        if self.mode == 'vfn':
            coupling = np.ones(coupling.shape)

        # TODO: do we want to use the throughput w/ or w/o the planet losses?
        return (1-self.get_inst_throughput_newao(wvs) / coupling)

    def get_fei_emissivity(self,wvs):
        
        """
        fei emissivity
        
        Note that for PSIsim, the FEI throughput included AO throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        """
        th_ao = self.get_ao_throughput(wvs)
        th_feicom = self.get_feicom_throughput(wvs)
        th_feired = self.get_feired_throughput(wvs)
        th_feiblue = self.get_feiblue_throughput(wvs)
        
        fei_tot_throughput_red = th_feicom * th_feired 
        fei_tot_throughput_blue = th_feicom * th_feiblue 
        
        red_range_index = np.where(wvs.value>1.4)
        blue_range_index = np.where((wvs.value<=1.4))
        
        fei_tot_throughput = np.concatenate([fei_tot_throughput_blue[blue_range_index],fei_tot_throughput_red[red_range_index]])
        return (1 - fei_tot_throughput * th_ao )

    def get_fiber_emissivity(self, wvs):
        """
        fiber emissivity
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        """
        th_fibred = self.get_fibred_throughput(wvs)
        th_fibblue = self.get_fibblue_throughput(wvs)
        
        red_range_index = np.where(wvs.value>1.4)
        blue_range_index = np.where((wvs.value<=1.4))
        
        fiber_tot_base_throughput = np.concatenate([th_fibblue[blue_range_index],th_fibred[red_range_index]])
    
        return (1 - fiber_tot_base_throughput)
    
    def get_spec_emissivity(self, wvs):
        """
        spec emissivity
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        """
                
        th_rspec = self.get_rspec_throughput(wvs)
        th_bspec = self.get_bspec_throughput(wvs)
        
        red_range_index = np.where(wvs.value>1.4)
        blue_range_index = np.where((wvs.value<=1.4))
        
        spec_tot_throughput = np.concatenate([th_bspec[blue_range_index],th_rspec[red_range_index]])
    
        return (1 - spec_tot_throughput)
    
    def compute_SR(self,w,ttStatic=0,LO=10,PLon=1,piaa_boost=1.3,points=None,values=None,atm=0,adc=0):
        """
        Strehl ratio
        select correct coupling file
        to do:implement interpolation of coupling files instead of rounding variables

        date of the change: Jul 12, 2023
        Huihao Zhang (zhang.12043@osu.edu)
        
        Based on function "pick_coupling" in specsim (https://github.com/ashbake/specsim/blob/main/utils/throughput_tools.py)

        """
        out = self.grid_interp_coupling(int(PLon),atm=int(atm),adc=int(adc))
        grid_points, grid_values = out[0],out[1:] #if PL, three values
        points=grid_points
        values=grid_values
        dynwfe= self.ao_ho_wfe
        ttDynamic= self.ao_tt_dynamic

        PLon = int(PLon)
        waves = (w.value).copy()
        if np.min(waves) > 10:
            waves/=1000 # convert nm to um

        # check range of each variable
        if ttStatic > 10 or ttStatic < 0:
            raise ValueError('ttStatic is out of range, 0-10')
        if ttDynamic > 20 or ttDynamic < 0:
            raise ValueError('ttDynamic is out of range, 0-10')
        if LO > 100 or LO < 0:
            raise ValueError('LO is out of range,0-100')
        if PLon >1:
            raise ValueError('PL is out of range')

        if PLon:
            values_1,values_2,values_3 = values
            point = (LO,ttStatic,ttDynamic,waves)
            mode1 = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0) # see example https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
            mode2 = interpolate.interpn(points, values_2, point,bounds_error=False,fill_value=0) 
            mode3 = interpolate.interpn(points, values_3, point,bounds_error=False,fill_value=0) 

            #PLwav,PLdat = load_photonic_lantern() #transfer matrices input mode--> each SMF
            #mat = PLdat[10] # use middle one for now
            #test1 = mode1 * mat[0,0]  + mode2*mat[1,0] + mode3*mat[2,0]
            #test2 = mode1 * mat[0,1]  + mode2*mat[1,1] + mode3*mat[2,1]
            #test3 = mode1 * mat[2,2]  + mode2*mat[1,2] + mode3*mat[2,2]
            # apply only to YJ or make matrix diagonal for HK..map onto same wavelength grid somehow
            # get coupling
            losses = np.ones_like(mode1) # due to PL imperfection
            losses[np.where(waves< 1.400)[0]] = 0.95 # only apply to y band
            raw_coupling = losses*(mode1+mode2+mode3) # do dumb things for now #0.95 is a recombination loss term 
        else:
            values_1= values
            points, values_1 = grid_interp_coupling(PLon)
            point = (LO,ttStatic,ttDynamic,waves)
            raw_coupling = interpolate.interpn(points, values_1, point,bounds_error=False,fill_value=0)

        if np.max(waves) < 10:
            waves*=1000 # nm to match dynwfe

        ho_strehl =  np.exp(-(2*np.pi*dynwfe/waves)**2) # computed per wavelength as grid
        return ho_strehl
    
    def get_fiber_throughput(self, wvs):
        """
        fib total throughput
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        """
        th_fibred = self.get_fibred_throughput(wvs)
        th_fibblue = self.get_fibblue_throughput(wvs)
        
        red_range_index = np.where(wvs.value>1.4)
        blue_range_index = np.where((wvs.value<=1.4))
        
        fiber_tot_base_throughput = np.concatenate([th_fibblue[blue_range_index],th_fibred[red_range_index]])
    
        return fiber_tot_base_throughput
    
    def get_spec_throughput(self, wvs):
        """
        spec total throughput
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        """
                
        th_rspec = self.get_rspec_throughput(wvs)
        th_bspec = self.get_bspec_throughput(wvs)
        
        red_range_index = np.where(wvs.value>1.4)
        blue_range_index = np.where((wvs.value<=1.4))
        
        spec_tot_throughput = np.concatenate([th_bspec[blue_range_index],th_rspec[red_range_index]])
    
        return spec_tot_throughput
    

####


    def get_instrument_background(self,wvs,solidangle):
        '''
        Returns the instrument background at each wavelength in units of photons/s/Angstrom/arcsecond**2
        '''
        bb_lam_fei = BlackBody(self.temperature_fei,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        fei_therm = bb_lam_fei(wvs)
        fei_therm *= solidangle
        fei_therm = fei_therm.to(u.ph/(u.micron * u.s * u.cm**2),equivalencies=u.spectral_density(wvs)) * self.area_tel.to(u.cm**2)
        fei_therm *= self.get_fei_emissivity(wvs)
        fei_therm *= self.get_fiber_throughput(wvs)
        fei_therm *= self.get_spec_throughput(wvs)



        bb_lam_fiber = BlackBody(self.temperature_fiber, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
        fiber_therm = bb_lam_fiber(wvs)
        fiber_therm *= solidangle
        fiber_therm = fiber_therm.to(u.ph / (u.micron * u.s * u.cm ** 2),
                                     equivalencies=u.spectral_density(wvs)) * self.area_tel.to(u.cm ** 2)
        fiber_therm *= self.get_fiber_emissivity(wvs)
        fiber_therm *= self.get_spec_throughput(wvs)



        bb_lam_spec = BlackBody(self.temperature_spec,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        spec_therm = bb_lam_spec(wvs)
        spec_therm *= solidangle
        spec_therm = spec_therm.to(u.ph / (u.micron * u.s * u.cm ** 2),
                                   equivalencies=u.spectral_density(wvs)) * self.area_tel.to(u.cm ** 2)
        spec_therm *= self.get_spec_emissivity(wvs)
        #spec_therm *= self.qe

        return fei_therm + fiber_therm + spec_therm
    

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

        #if self.mode != 'vfn':
        #    print("Warning: only 'vfn' mode has been confirmed")

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
