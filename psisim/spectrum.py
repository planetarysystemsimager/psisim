import picaso


def generate_picaso_inputs(planet_table_entry,planet_type):
	'''
	A function that returns the required inputs for picaso, 
	given a row from a universe planet table

	Inputs:
	planet_table_entry - a single row, corresponding to a single planet
						 from a universe planet table [astropy table (or maybe astropy row)]
	planet_type - either "Terrestrial", "Ice" or "Giant" [string]
	
	'''

	pass

def simulate_spectrum(planet_table_entry,wvs,R,atmospheric_parameters,package="picaso"):
	'''
	Simuluate a spectrum from a given package

	Inputs: 
	planet_table_entry - a single row, corresponding to a single planet
						 from a universe planet table [astropy table (or maybe astropy row)]
	wvs				   - a list of wavelengths to consider
	R				   - the resolving power
	atmospheric parameters - To be defined

	Outputs:
	F_lambda
	'''

	if isinstance(wvs,float):
		return 1
	else:
		return np.ones(len(wvs))

	pass

def downsample_spectrum(spectrum,R_in, R_out):
	'''
	Downsample a spectrum from one resolving power to another

	Inputs: 
	spectrum - F_lambda that has a resolving power of R_in
	R_in 	 - The resolving power of the input spectrum
	R_out	 - The desired resolving power of the output spectrum

	Outputs:
	new_spectrum - The original spectrum, but now downsampled
	'''
	pass