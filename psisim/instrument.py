import psisim
import os
import glob
import scipy.interpolate as si
import numpy as np

class Instrument():
	'''
	A class that defines a general instrument

	The main properties will be: 
	read_noise	- The read noise in e-
	filters - A list of strings of filter names
	ao_filter - A string that is the filter name for the AO mag
	gain 	- The detector gain in e-/ADU
	dark_current - The dark current in e-/sec/pixel
	qe - quantum efficiency, as a fraction
	pixel_scale - the size of a pixel in angular units

	There will also be a set of "current_setup" properties:
	exposure_time  - The exposure time in seconds [float]
	n_exposures    - The number of exposures [int]
	current_filter - The current filter [string]
	current_R 		- The current resolving power (float)
	More to come!

	Later we might also have ao_filter2

	The main functions will be: 
	get_inst_throughput()	 - A function that returns the total instrument throughput as a function of wavelength

	'''

	def __init__(self):
		pass

	def get_inst_throughput(self, wvs):
		'''
		A function that returns the instrument throughput at a given set of wavelengths
		'''

		if isinstance(wvs,float):
			return 1
		else:
			return np.ones(len(wvs))

	def get_filter_transmission(self,wvs,filter_name):
		'''
		A function to get the transmission of a given filter at a given set of wavelengths

		User inputs:
		wvs 	- A list of desired wavelengths [um]
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
		ao_mag 	- The magnitude that the AO system sees [float]
		sci_mag - The magnitude in the science band [float]
		sci_filter - The science band filter name [string]
		SpT 	- The spetral type (or maybe the effective temperature, TBD) [string or float]
		
		Keyword Arguments:
		ao_mag2 - PSI blue might have a visible and NIR WFS, so we need two ao_mags

		Outputs:
		speckle_noise -	the speckle_noise in contrast units
		'''
		pass

	def get_instrument_background(self,wvs):
		'''
		Return the instrument background at a given set of wavelengths

		Inputs: 
		wvs - a list of wavelengths in microns

		Outputs: 
		backgrounds - a list of background values at a given wavelength. Unit TBD
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


		# The current obseving properties - dynamic
		self.exposure_time = None
		self.n_exposures = None
		self.current_filter = None
		self.current_R = None

	def get_speckle_noise(self,separations,ao_mag,sci_mag,wvs,SpT,ao_mag2=None,
		contrast_dir=None, integrator='si'):
		'''
		Returns the contrast for a given list of separations. 
		The default is to use the contrasts provided so far by Jared Males

		The code currently rounds to the nearest I mag. This could be improved. 

		TODO: Write down somewhere the file format expected. 

		Inputs: 
		separations	 - A list of separations at which to calculate the speckle noise [float list length n]
		ao_mag 		 - The magnitude in the ao band, here assumed to be I-band
		sci_mag 	 - The magnitude in the science band (do we actually need this?)
		wvs 		 - A list of wavelengths in microns [float length m]

		Outputs: 
		get_speckle_noise - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]

		'''

		if contrast_dir is None:
			contrast_dir = os.path.dirname(psisim.__file__)+"/data/default_contrast/"

		if integrator not in ['si','lp']:
			raise ValueException("The integrator you've selected is not supported."
				" We currently only support standard integrator of linear predictor"
				" as 'si or 'lp")


		#Find all the contrast files
		fnames = glob.glob(contrast_dir+"*"+integrator+"_profile.dat")

		#Extract the magnitudes from the filenames
		mags = [float(fname.split("/")[-1].split("_")[1]) for fname in fnames]

		#### Make an array to hold the contrast profiles for each magnitude
		# Assumes that each file has the same number of entries. 

		#Round the host_Imags
		host_mag = np.around(ao_mag)
		#Get the file index
		magnitude_index = np.where(mags == host_mag)[0][0]
		#Make sure we support it
		if magnitude_index.shape == 0:
			raise ValueException("We don't yet support the ao_mag you input. "
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

		return speckle_noise


	def set_observing_mode(exposure_time,n_exposures,sci_filter,R):
		'''
		Sets the current observing setup
		'''

		self.exposure_time = exposure_time
		self.n_exposures = n_exposures

		if sci_filter not in self.filters:
			raise ValueException("The filter you selected is not valid for PSF_Blue. Check the self.filters property")
		else:
			self.current_filter = sci_filter

		self.R = R

