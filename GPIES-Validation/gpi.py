import os
import glob
import math
import scipy.interpolate as si
import numpy as np
import astropy.units as u
import astropy.constants as constants
import pysynphot as ps
from random import randint
import psisim
from psisim.instruments.template import Instrument

class GPI(Instrument):
    '''
    An implementation of Instrument for GPI
    '''
    def __init__(self,telescope=None):
        super(GPI,self).__init__()

        
        # Source 1: Patrick Ingraham https://arxiv.org/pdf/1407.2302.pdf
        

        # The main instrument properties - static
        self.read_noise = 17. * u.electron # Source 1
        self.gain = 3.04 #e-/ADU # Source 1
        self.dark_current = 0.006 *u.electron/u.s # Source 1
        self.qe = .85 *u.electron/u.ph # Source 1; found as range 0.79-0.92; can refine later
        self.spatial_sampling = 3 #this is just data cubes right? so 3d?
        
        self.filters = ['H'] #'Y','J','H','K1','K2' only H band implemented for now
        self.ao_filter = ['i']

        self.IWA = 0.123*u.arcsec #Inner working angle in arcseconds based on H-band focal plane mask diameter; can add other bands later
        self.OWA = 1.9*u.arcsec #Outer working angle in arcseconds based on diagonal field of view

        if telescope is None:
            self.telescope = psisim.telescope.GeminiSouth()
        else:
            self.telescope = telescope

        # The current observing properties - dynamic
        self.exposure_time = None
        self.n_exposures = None
        self.current_filter = None
        self.current_R = None
        self.current_wvs = None
        self.current_dwvs = None

    def get_speckle_noise(self, separations, ao_mag, ao_filter, wvs=None, star_spt=None, telescope=None, ao_mag2=None, contrast_dir=None):
        '''
        Source 2: Vanessa Bailey https://arxiv.org/pdf/1609.08689.pdf
        
        The code currently uses contrast data from Vanessa Bailey's paper. The contrast at separations larger than 0.8" is assumed to be the same as at 0.8".  The contrast at separations smaller than 0.25" is extrapolated.
        
        Inputs: 
        separations     - A list of separations at which to calculate the speckle noise [float list length n]
        ao_mag          - The magnitude in the ao band, here assumed to be I-band

        Outputs: 
        get_speckle_noise - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]

        '''
        #Read in the separation v contrast file        
        contrasts_table = np.loadtxt("../psisim/data/SepvCon.csv", dtype = float, skiprows = 2) # Source 2
        separation_data = contrasts_table[0:3,0]
        contrast_data = contrasts_table[:,1]
        get_speckle_noise = []
        
        separations *= 1/u.arcsec
        separations = [separations]

       #Make an interpolation function to find the contrast at other points
        for i in separations:
            if i > 0.8:
                i = 0.8 #We assume that at separations greater than 0.8", the contrast is the same as at 0.8"
            if ao_mag < 2.0:
                contrasts = contrast_data[0:3]
            elif ao_mag >= 2.0 and ao_mag <= 3.0:
                contrasts = contrast_data[0:3]
            elif ao_mag > 3.0 and ao_mag <= 4.0:
                contrasts = contrast_data[3:6]
            elif ao_mag > 4.0 and ao_mag <= 5.0:
                contrasts = contrast_data[6:9]
            elif ao_mag > 5.0 and ao_mag <= 6.0:
                contrasts = contrast_data[9:12]
            elif ao_mag > 6.0 and ao_mag <= 7.0:
                contrasts = contrast_data[12:15]
            elif ao_mag > 7.0 and ao_mag <= 8.0:
                contrasts = contrast_data[15:18]
            elif ao_mag > 8.0 and ao_mag <= 9.0:
                contrasts = contrast_data[18:21]
            elif ao_mag > 9.0 and ao_mag <= 10.0:
                contrasts = contrast_data[21:24]
            elif ao_mag > 10.0:
                contrasts = contrast_data[21:24]
#         if ao_mag < 2.0:
#             ao_mag = 2.0
#         elif ao_mag >= 10.0:
#             ao_mag = 9.0
            f = si.interp1d(separation_data, contrasts, fill_value = "extrapolate") #We extrapolate contrasts at separations smaller than 0.25"
            interpolated_contrast = f(i)/5 # Dividing by 5 because we used 5-sigma noise values and we want 1-sigma
            get_speckle_noise = np.append(get_speckle_noise, interpolated_contrast)
        return get_speckle_noise

    def set_observing_mode(self,exposure_time,n_exposures,sci_filter,R,wvs,dwvs=None):
        '''
        Sets the current observing setup
        
        Exposure time is in seconds
        '''

        self.exposure_time = exposure_time*u.s
        self.n_exposures = n_exposures

        if sci_filter not in self.filters:
            raise ValueError("The filter you selected is not valid. Only H band implemented so far.")
        else:
            self.current_filter = sci_filter
            #To-do: change inner working angle according to filter 
#             fpm_diam_y   0.156
#             fpm_diam_j   0.184
#             fpm_diam_h   0.246
#             fpm_diam_k1   0.306
#             fpm_diam_k2   0.306

        self.current_R = R

        self.current_wvs = wvs 
        if dwvs is None:
            dwvs = np.abs(wvs - np.roll(wvs, 1))
            dwvs[0] = dwvs[1]
        self.current_dwvs = dwvs

    def detect_planets(self,planet_table,snrs,smallest_iwa_by_wv=True,user_iwas=None):
        '''
        A function that returns a boolean array indicating whether or not a planet was detected
        
        user_iwas should be in arcseconds with astropy units
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
                iwas=[self.current_wvs.to(u.m)/self.telescope.diameter*206265*u.arcsec] #Lambda/D in arcseconds
            else: 
                iwas=[self.current_wvs*0. + self.IWA]

        detected = np.full((len(planet_table),self.current_wvs.size),False,dtype=bool)
        #For each planet, for each wavelength check the separation and the SNR
        for i,planet in enumerate(planet_table):
            sep = planet['AngSep'].to(u.arcsec)
            for j,wv in enumerate([self.current_wvs]): 
                # if sep < 0.070:
                    # print(sep,snrs[i,j],(sep > iwas[j]))
                if (sep > iwas[j]) & (sep < self.OWA) & (snrs[i,j].value > 5):
                    detected[i,j] = True

        return detected
           
    def get_inst_throughput(self, wvs=None, planet_flag=False, planet_sep=None):
        '''
        Jerome Maire https://arxiv.org/pdf/1407.2306.pdf
        
        Throughput value is an estimated value based on data from Jerome Maire paper; It is currently a rough estimate.
        
        We assume that this throughput includes the throughput from the adaptive optics filter and quantum efficiency.
        
        A function that returns the instrument throughput in the H band.
        '''
        return 0.05
    
    
    def get_filter_transmission(self,wvs,filter_name):
        '''
        A function to get the transmission of a given filter at a given set of wavelengths
         
        Transmission data from SVO filter profile services:
        http://svo2.cab.inta-csic.es/theory/fps/index.php?id=Gemini/GPI.H&&mode=browse&gname=Gemini&gname2=GPI#filter

        User inputs:
        filter_name - A string corresponding to a filter in the filter database
        '''

        if filter_name == 'H':
            wavelength_transmission = np.loadtxt("../psisim/data/transmission.dat", dtype = float, skiprows = 3)
            transmissions = wavelength_transmission[:,1]
            av_transmission = sum(transmissions)/len(transmissions)
        else:
            raise ValueError("Only H band implemented so far")
            
        return av_transmission
    
    def get_effective_wavelength(self,sci_filter):
        if sci_filter not in self.filters:
            raise ValueError("The filter you selected is not valid. Only H band implemented so far.")
        else:
            wvs = (16322.60*u.Angstrom).to(u.m) #Effective wavelength of GPI H-band from http://svo2.cab.inta-csic.es/theory/fps/
            
            #dwvs is the full width half max
            wavelength_transmission = np.loadtxt("../psisim/data/transmission.dat", dtype = float, skiprows = 3)
            f = si.interp1d(wavelength_transmission[0:948,1],wavelength_transmission[0:948,0])
            g = si.interp1d(wavelength_transmission[600:1895,1],wavelength_transmission[600:1895,0])
            right = f(0.480413525)
            left = g(0.480413525)
            dwvs = ((left-right)*u.Angstrom).to(u.m)
        return wvs, dwvs
    def get_post_processing_gain(self,AngSep):
        '''
        Function to return the post processing gain based on the stellar I-band magnitude
        
        Source 4: Vanessa Bailey https://arxiv.org/pdf/1609.08689.pdf; Interpolation based on initial contrast to final contrast ratios from Figure 4  
        
        Takes angular separation in arcseconds as input
        '''
        interp_post_processing = si.interp1d([.25,.4,.8],[2.62,2.49,2.55],bounds_error = False, fill_value = ([2.62],[2.49]))
        interp_post_processing_std = si.interp1d([.25,.4,.8],[.54,.48,.46],bounds_error = False, fill_value = ([.54],[.46]))
        post_processing_gain_mean = interp_post_processing(AngSep.value)
        post_processing_gain_std = interp_post_processing_std(AngSep.value)
        post_processing_gain = np.random.lognormal(post_processing_gain_mean,post_processing_gain_std)
        return post_processing_gain
