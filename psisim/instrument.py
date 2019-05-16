


class Instrument():
	'''
	A class that defines a general instrument

	The main properties will be: 
	read_noise	- The read noise in e-
	filters - A list of strings of filter names
	ao_filter - A string that is the filter name for the AO mag

	Later we might also have ao_filter2

	The main functions will be: 
	get_inst_throughput()	 - A function that returns the total instrument throughput as a function of wavelength

	'''

	def __init__(self):


	def get_inst_throughput(self, wvs):
		'''
		A function that returns the instrument throughput at a given set of wavelengths
		'''

		pass

	def get_filter_transmission(self,wvs,filter_name):
		'''
		A function to get the transmission of a given filter at a given set of wavelengths

		User inputs:
		wvs 	- A list of desired wavelengths [um]
		filter_name - A string corresponding to a filter in the filter database
		'''

		# if filter_name not in self.filters:
			# ERROR
		pass

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
	


