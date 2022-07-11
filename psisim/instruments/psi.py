import os
import glob
import scipy.interpolate as si
import numpy as np
import astropy.units as u
import astropy.constants as constants
import pysynphot as ps

import psisim
from psisim.instruments.template import Instrument


class PSI_Blue(Instrument):
    '''
    An implementation of Instrument for PSI-Blue
    '''
    def __init__(self,telescope=None):
        super(PSI_Blue,self).__init__()

        # The main instrument properties - static
        self.read_noise = 0. * u.electron
        self.gain = 1. #e-/ADU
        self.dark_current = 0. *u.electron/u.s
        self.qe = 1. *u.electron/u.ph
        self.spatial_sampling = 2
        
        self.filters = ['r','i','z','Y','J','H']
        self.ao_filter = ['i']
        self.ao_filter2 = ['H']

        self.IWA = 0.0055*u.arcsec #Inner working angle in arcseconds. Current value based on 1*lambda/D at 800nm
        self.OWA = 1.*u.arcsec #Outer working angle in arcseconds

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

    def get_speckle_noise(self,separations,ao_mag,ao_filter,wvs,star_spt,telescope,ao_mag2=None,
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
            sep = planet['AngSep'].to(u.arcsec).value
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
        self.read_noise = 0.*u.electron
        self.gain = 1. #e-/ADU
        self.dark_current = 0.*u.electron/u.s
        self.qe = 1.*u.electron/u.ph

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

    def get_instrument_background(self,wvs,solidangle):
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
            return limit[0]*u.ph/u.s/u.AA
        else:
            return np.repeat(limit,len(wvs))*u.ph/u.s/u.AA
