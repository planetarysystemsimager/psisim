


def simulate_observation(telescope,instrument,planet_table_entry,planet_spectrum,wvs,
	inject_noise=True):
	'''
	A function that simulates an observation

	Inputs:
	Telescope	 - A Telescope object
	Instrument	 - An Instrument object
	planet_table_entry - an entry/row from a Universe planet table
	planet_spectrum - A planet spectrum from simulate spectrum (assumed to be at 10pc)
	observing_configs - To be defined

	Outputs: 
	F_lambda, F_lambda_error
	'''


	##### ALL UNITS NEED TO BE PROPERLY EXAMINED #####

	#Some relevant planet properties
	separation = planet_table_entry['AngSep']
	star_imag = planet_table_entry['StarImag']
	star_spt = "Dummy" #planet_table_entry['SpT']

	#Scale the planet spectrum for distance
	distance = planet_table_entry['Distance'] #Assumed in parcsec
	scaled_spectrum = planet_spectrum*(distance/10)**2

	#Multiply by instrument throughputs
	detector_spectrum = scaled_spectrum*instrument.get_inst_throughput(wvs)
	detector_spectrum *= instrument.get_filter_transmission(wvs,instrument.current_filter)

	# TODO: Convert to photons/s. Presumably the specutrum is in W/m^2 or something 
	# and needs to be converted

	#Multiply by the gain to get into e- and then multiply by the quantum efficiency
	detector_spectrum *= instrument.gain*instrument.qe

	#Multiply by the exposure time
	detector_spectrum *= instrument.exposure_time #The detector spectrum is now in e-

	#Multiply by the number of exposures
	detector_spectrum *= instrument.n_exposures


	# Now get the various noise sources: 
	speckle_noise = instrument.get_speckle_noise(separations,star_imag,
		instrument.current_filter,wvs,star_spt) #### ALMOST CERTAINLY NOT IN THE CORRECT UNITS. CURRENTLY CONTRAST UNITS NEEDS TO GET TO E-. 

	# Multiply the read noise by sqrt(n_exposures)
	read_noise = np.sqrt(instrument.n_exposures)*instrument.read_noise
	
	#Add the dark_current to the spectrum and calculate dark noise. NEVERMIND NOT ADDING TO SPECTRUM RIGHT NOW
	dark_current = instrument.dark_current*instrument.exposure_time
	# detector_spectrum += dark_current
	dark_noise = np.sqrt(dark_current)

	#Photon noise. Detector_spectrum should be in total of e- now.
	photon_noise = np.sqrt(detector_spectrum) 

	#For now I'm going to add the speckle_noise to all the other sources of noise in quadrature
	total_noise = np.sqrt(speckle_noise**2 + read_noise**2 + dark_noise**2 + photon_noise**2)

	#TODO: Add background noise

	# Inject noise into spectrum
	if inject_noise:
		# For each point in the spectrum, draw from a normal distribution,
		# with a mean centered on the spectrum and the standard deviation
		# equal to the noise
		for i,noise in enumerate(total_noise):
			detector_spectrum[i] = np.random.normal(detector_spectrum[i],noise)

	#TODO: Currently everything is in e-. We likely want it in a different unit at the end. 

	return detector_spectrum, total_noise

def simulate_observation_set(telescope, instrument, planet_table,planet_spectra_list,wvs):
	'''
	Simulates observations of multiple planets, with the same observing configs
	
	Inputs:
	Telescope	 - A Telescope object
	Instrument	 - An Instrument object
	planet_table - a Universe planet table
	planet_spectra_list - A list of planet spectra. One for each entry in the planet table

	Outputs: 
	F_lambdas, F_lambda_errors
	'''

	n_planets = np.size(planet_table) #Not sure this will work

	F_lambdas = []
	F_lambda_errors = []

	for i,planet in enumerate(planet_table):
		new_F_lambda,new_F_lambda_errors = simulate_observation(telescope,instrument,
			planet,planet_spectra[i],wvs)
		F_lambdas.append(new_F_lambda)
		F_lambda_errors.append(new_F_lambda_errors)

	F_lambdas = np.array(F_lambdas)
	F_lambda_errors = np.array(F_lambda_errors)

	return F_lambdas,F_lambda_errors




