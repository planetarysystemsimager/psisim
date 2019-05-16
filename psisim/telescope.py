import numpy as np


class Telescope():
	'''
	A general class that describes a telescope and site

	The main properties will be: 
	diameter	- The telescope diameter in meters
	collecting_area	- The telescope collecting area in meters
	
	Properties to consider adding later may be seeing, R0, Tau, etc. 

	The main functions will be: 
	get_sky_background()	- A function to get the sky background
	get_atmospheric_transmission()	- A function to get the atmospheric transmission

	'''
	def __init__(self,diameter, collecting_area=None):
		'''
		Constructor

		'''

		self.diameter = diameter

		#If no collecting area is passed, be naive and assume it's a full circular mirror
		if collecting_area is not None:
			self.collecting_area = collecting_area
		else: 
			self.collecting_area = np.pi*(self.diameter/2)**2





	def get_sky_background(self,wvs):
		'''
		A function that returns the sky background for a given set of wavelengths. 

		Later it might be a function of pressure, temperature and humidity

		Inputs: 
		wvs 	- A list of wavelengths 

		Outputs: 
		backgrounds - A list of backgrounds
		'''


		pass #Placeholder
		#TODO: Raise not implemented exception
		# return backgrounds


	def get_atmospheric_transmission(self,wvs):
		'''
		A function that returns the atmospheric transmission for a given set of wavelengths. 

		Later it might be a function of pressure, temperature and humidity

		Inputs: 
		wvs 	- A list of wavelengths 

		Outputs: 
		transmissions - A list of atmospheric transmissions
		'''


		pass #Placeholder
		#TODO: Raise not implemented exception
		# return transmissions




