import os
import glob
import scipy.interpolate as si
import numpy as np
from scipy import interpolate
import pandas as pd
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

# +
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
        self.ao_ho_wfe = None
        self.ao_tt_dynamic = None
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
        th_fib = self.get_fib_throughput(wvs)
        th_rspec = self.get_rspec_throughput(wvs)
        th_bspec = self.get_bspec_throughput(wvs)
        
        base_throughput_red = th_ao * th_feicom * th_feired * th_fib * th_rspec 
        base_throughput_blue = th_ao * th_feicom * th_feiblue * th_fib * th_bspec 
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

        file: official HISPEC ao throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 

        """
        ao_data =np.genfromtxt(datadir + '/throughput/ao_throughput_hispec.csv', delimiter=',', skip_header=1)
        f_ao=interpolate.interp1d(ao_data[:, 0], ao_data[:, 1], bounds_error=False,fill_value=0)
        ao_th = f_ao(wvs.value)
        
        return ao_th

    def get_feicom_throughput(self,wvs):
    
        """
        input: wvs = wavelength in um
    
        output: FEI common throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official HISPEC FEI common throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        feicom_data =np.genfromtxt(datadir + '/throughput/feicom_throughput_hispec.csv', delimiter=',', skip_header=1)
        f_feicom=interpolate.interp1d(feicom_data[:, 0], feicom_data[:, 1], bounds_error=False,fill_value=0)
        feicom_th = f_feicom(wvs.value)
        
        return feicom_th
    def get_feired_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: FEI red throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official HISPEC FEI red throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        feired_data =np.genfromtxt(datadir + '/throughput/feired_throughput_hispec.csv', delimiter=',', skip_header=1)
        f_feired=interpolate.interp1d(feired_data[:, 0], feired_data[:, 1], bounds_error=False,fill_value=0)
        feired_th = f_feired(wvs.value)
        
        return feired_th
    def get_feiblue_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: FEI blue throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official HISPEC FEI blue throuhhput, source: Garreth Ruane, date: Jun 29, 2023.         

        """
        feiblue_data =np.genfromtxt(datadir + '/throughput/feiblue_throughput_hispec.csv', delimiter=',', skip_header=1)
        f_feiblue=interpolate.interp1d(feiblue_data[:, 0], feiblue_data[:, 1], bounds_error=False,fill_value=0)
        feiblue_th = f_feiblue(wvs.value)
        
        return feiblue_th

    def get_fib_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: FIB total throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official HISPEC FIB total throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        fib_data =np.genfromtxt(datadir + '/throughput/fib_throughput_hispec.csv', delimiter=',', skip_header=1)
        f_fib=interpolate.interp1d(fib_data[:, 0], fib_data[:, 1], bounds_error=False,fill_value=0)
        fib_th = f_fib(wvs.value)
        
        return fib_th
    def get_bspec_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: BSPEC throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official HISPEC BSPEC throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        
        """
        bspec_data =np.genfromtxt(datadir + '/throughput/bspec_throughput_hispec.csv', delimiter=',', skip_header=1)
        f_bspec=interpolate.interp1d(bspec_data[:, 0], bspec_data[:, 1], bounds_error=False,fill_value=0)
        bspec_th = f_bspec(wvs.value)
        
        return bspec_th
    def get_rspec_throughput(self,wvs):
        

        """
        input: wvs = wavelength in um
    
        output: RSPEC throughput
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)

        file: official HISPEC RSPEC throuhhput, source: Garreth Ruane, date: Jun 29, 2023. 
        """
        rspec_data =np.genfromtxt(datadir + '/throughput/rspec_throughput_hispec.csv', delimiter=',', skip_header=1)
        f_rspec=interpolate.interp1d(rspec_data[:, 0], rspec_data[:, 1], bounds_error=False,fill_value=0)
        rspec_th = f_rspec(wvs.value)
        
        return rspec_th
    
    def pick_coupling(self,w,ttStatic=0,LO=30,PLon=1,piaa_boost=1.3,points=None,values=None,atm=1,adc=1):
        """
        input: wvs = wavelength
    
        output: instrument coupling
        
        
        date of the change: Jun 29, 2023

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
        
        date of the change: Jun 29, 2023

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
        return (1-self.get_inst_throughput(wvs) / coupling)

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

        fib emissivity
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        
        """

        th_fib =self.get_fib_throughput(wvs)
        

        
        return (1 - th_fib)
    
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
    
    def compute_SR(self,w,ttStatic=0,LO=30,PLon=1,piaa_boost=1.3,points=None,values=None,atm=1,adc=1):
        """
        Strehl ratio
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
        return ho_strehl
    
    # def get_inst_emissivity(self,wvs):
    #     '''
    #     The instrument emissivity
    #     '''
    #
    #     # TODO: do we want to use the throughput w/ or w/o the planet losses?
    #     return (1-self.get_inst_throughput(wvs))

    def get_instrument_background(self,wvs,solidangle):
        '''

        Returns the instrument background at each wavelength in units of photons/s/Angstrom/arcsecond**2
        
        date of the change: Jun 29, 2023

        Huihao Zhang (zhang.12043@osu.edu)
        '''

        bb_lam_fei = BlackBody(self.temperature_fei,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        fei_therm = bb_lam_fei(wvs)
        fei_therm *= solidangle
        fei_therm = fei_therm.to(u.ph/(u.micron * u.s * u.cm**2),equivalencies=u.spectral_density(wvs)) * self.telescope.collecting_area
        fei_therm *= self.get_fei_emissivity(wvs)
        fei_therm *= self.get_fib_throughput(wvs)
        fei_therm *= self.get_spec_throughput(wvs)

        bb_lam_fiber = BlackBody(self.temperature_fiber, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
        fiber_therm = bb_lam_fiber(wvs)
        fiber_therm *= solidangle
        fiber_therm = fiber_therm.to(u.ph / (u.micron * u.s * u.cm ** 2),
                                     equivalencies=u.spectral_density(wvs)) * self.telescope.collecting_area
        fiber_therm *= self.get_fiber_emissivity(wvs)
        fiber_therm *= self.get_spec_throughput(wvs)

        bb_lam_spec = BlackBody(self.temperature_spec,scale=1.0*u.erg/(u.cm**2*u.AA*u.s*u.sr))
        spec_therm = bb_lam_spec(wvs)
        spec_therm *= solidangle
        spec_therm = spec_therm.to(u.ph / (u.micron * u.s * u.cm ** 2),
                                   equivalencies=u.spectral_density(wvs)) * self.telescope.collecting_area
        spec_therm *= self.get_spec_emissivity(wvs)

        return fei_therm + fiber_therm + spec_therm
####
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

#    def get_dar_coupling_throughput(self, wvs, wvs0=None):
#        """
#        Gets the relative loss in fiber coupling due to DAR (normalized to 1)
#
#        Args:
#            wvs (np.array of float): wavelengths to consider
#            wvs0 (float, optiona): the reference wavelength where fiber coupling is maximized. 
#                                    Assumed to be the mean of the input wavelengths if not passed in
#
#        Returns:
#            np.array of float: the relative loss in throughput in fiber coupling due to DAR (normalized to 1)
#        """
#        if wvs0 is None:
#            wvs0 = np.mean(wvs)
#
#        n = self.telescope.get_nair(wvs)
#        n0 = self.telescope.get_nair(wvs0)
#
#        dar = np.abs(nair.compute_dar(n, n0, np.radians(self.zenith)))
#
#        lam_d = (wvs * u.um) / (self.telescope.diameter.to(u.um)) * 206265 * 1000
#
#        coupling_th = np.interp(dar/lam_d, self.fiber_coupling['sep'], self.fiber_coupling['eta'], right=0)
#
#        return coupling_th
