#import astropy tables or something


class Universe():
	'''
	A universe class that includes

	Inherited from EXOSIMS? TBD

	Properties:
	planets	- A planet table that holds all the planet properties [Astropy table]

	'''

	def __init__(self):
		'''
		'''
		pass

	def save_planet_table(self,filename):
		'''
		Save the planet table to an astropy fits tables
		'''
		pass


	def load_planet_table(self,filename):
		'''
		Load a planet table that was saved by save_planet_table
		'''
		pass

class ExoSims_Universe(Universe):
	'''
	A child class of Universe that is adapted specfically to work with the outputs of EXOSIMS
	'''

	def __init__(self,exosims_config_filename):
		super(ExoSims_Universe, self).__init__()
		
		self.filename = exosims_config_filename

	def simulate_EXOSIMS_Universe(self):
		'''
		A function that runs EXOSIMS and takes the output to populate the planets table
		'''

		pass


