import os
import glob
import scipy.interpolate as si
import numpy as np
import astropy.units as u
import astropy.constants as constants
import pysynphot as ps
import warnings
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
from astropy.modeling.models import BlackBody

import psisim
from psisim.instruments.hispec import *


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
        # self.th_data = np.genfromtxt(self.telescope.path+'/throughput/hispec_throughput_budget.csv',
                                #    skip_header=1,usecols=np.arange(5,166),delimiter=',',missing_values='')

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
        # self.filters = ["CFHT-Y","TwoMASS-J",'TwoMASS-H','TwoMASS-K']         


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

    # def load_scale_aowfe(self,seeing,airmass,site_median_seeing=0.6):
    #     '''
    #     A function that returns ao wavefront errors as a function of rmag

    #     Args:
    #     path     -  Path to an ao errorbudget file [str]
    #     seeing   -  The current seeing conditions in arcseconds  [float]
    #     airmass  -  The current airmass [float]
    #     '''
        
    #     path = self.telescope.path

    #     #Read in the ao_wfe
    #     ao_wfe=np.genfromtxt(path+'aowfe/hispec_modhis_ao_errorbudget_v3.csv', delimiter=',',skip_header=1)
    #     ao_rmag = ao_wfe[:,0]
        
    #     if self.mode == 'vfn':
    #         # For VFN, rescale WFE to use telemetry values from the PyWFS
    #         # The default table includes some errors that VFN doesn't care about
    #           # Based on 11/2021 telemetry, PyWFS has hit 85nm RMS WF residuals so
    #           # let's set that as the best value for now and then scale up from there
    #         ao_wfe[:,6] = ao_wfe[:,6] * 85/ao_wfe[0,6]

    #     # indexes for ao_wfe from Dimitri Code
    #     ao_wfe_ngs=ao_wfe[:,6] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))
    #     ao_wfe_lgs=ao_wfe[:,7] * np.sqrt((seeing/site_median_seeing * airmass**0.6)**(5./3.))

    #     return ao_rmag,ao_wfe_ngs*u.nm,ao_wfe_lgs*u.nm
