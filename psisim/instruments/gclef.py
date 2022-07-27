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
from psisim.instruments.template import Instrument

class gclef(Instrument):
    '''
     An implementaion of Instrument for Modhis
    '''
    def __init__(self,telescope=psisim.telescope.GMT):
        super(gclef,self).__init__()

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
        self.read_noise = 2 *u.electron # * u.photon#e-/pix/fr
        self.dark_current = 0.02*u.electron/u.s #electrons/s/pix
        self.qe = 1 * u.electron/u.ph
        self.temperature = 276*u.K

        if telescope is None:
            self.telescope = psisim.telescope.Keck()
        else:
            self.telescope = telescope

        # self.th_data = np.genfromtxt(datadir+'/throughput/hispec_throughput_budget.csv',
        #                             skip_header=1,usecols=np.arange(5,1566),delimiter=',',missing_values='')

        # # load in fiber coupling efficiency as a function of misalignment
        # fiber_data_filename = os.path.join(datadir, "smf", "keck_pupil_charge0.csv")
        # self.fiber_coupling = astropy.io.ascii.read(fiber_data_filename, names=["sep", "eta"])
        # self.fiber_coupling['eta'] /= self.fiber_coupling['eta'][0] # normalize to peak, since this is just the relative term

        # TODO: GET AO parameters from Jared
        # self.nactuators = 32. - 2.0 # The number of DM actuators in one direction
        # self.fiber_contrast_gain = 3. #The gain in contrast thanks to the fiber. 
        # self.p_law_dh = -2.0 # The some power law constant Dimitri should explain. 
        # self.ao_filter = 'bessell-I' # Available AO filters
        # self.d_ao = 0.15 * u.m
        # self.area_ao = np.pi*(self.d_ao/2)**2
        
        self.name = "GMT-GCLEF"
        
        #TODO: Acceptable filters
        # self.filters = ['CFHT-Y','TwoMASS-J','TwoMASS-H','TwoMASS-K'] #Available observing filters

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
    
    






