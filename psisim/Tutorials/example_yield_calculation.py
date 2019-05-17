from psisim import telescope,instrument,observation,spectrum,universe,plots

wvs = [1.1,1.6] #Choose some wavelengths

tmt = telescope.TMT()
psi_blue = instrument.psi_blue()
psi_blue.set_observing_mode(60,40,'z',10) #60s, 40 exposures,z-band, R of 10

exosims_config_filename = "/" #Some filename here
uni = universe.ExoSims_Universe(exosims_config_filename)
uni.simulate_EXOSIMS_Universe()

planet_table = uni.planet_table
n_planets = len(planet_table)

planet_types = []
planet_spectra = []

for planet in planet_table:

	#INSERT PLANET SELECTION RULES HERE
	planet_type = "Terrestrial"
	planet_types.append(planet_type)

	#Generate the spectrum and downsample
	atmospheric_parameters = generate_picaso_inputs(planet,planet_type)
	planet_spectrum = spectrum.simulate_spectrum(planet,wvs,psi_blue.R,atmospheric_parameters)
	# planet_spectrum = spectrum.downsample_spectrum(planet_spectrum,10000,psi_blue.R)
	planet_spectra.append(planet_spectrum)

sim_F_lambda, sim_F_lambda_errs = observation.simulate_observation_set(tmt,psi_blue,
	planet_table,planet_spectra,wvs)