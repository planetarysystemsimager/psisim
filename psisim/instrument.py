


class Instrument():
	'''
	A class that defines a general instrument

	The main properties will be: 
	read_noise	- The read noise in e-
	filters - A list of strings of filter names
	ao_filter - A string that is the filter name for the AO mag

	There will also be a set of "current_setup" properties:
	self.exposure_time  - The exposure time in seconds [float]
	self.n_exposures    - The number of exposures [int]
	self.current_filter - The current filter [string]
	self.current_R 		- The current resolving power (float)
	More to come!

	Later we might also have ao_filter2

	The main functions will be: 
	get_inst_throughput()	 - A function that returns the total instrument throughput as a function of wavelength

	'''

	def __init__(self):


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
		self.read_noise = 1
		self.filters = ['r','i','z','Y','J','H']
		self.ao_filter = ['i']
		self.ao_filter2 = ['H']

		# The current obseving properties - dynamic
		self.exposure_time = None
		self.n_exposures = None
		self.current_filter = None
		self.current_R = None

	def get_speckle_noise(self,separations,ao_mag,sci_mag,sci_filter,SpT,ao_mag2=None):
		'''
		MAX TO FILL IN
		'''


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

