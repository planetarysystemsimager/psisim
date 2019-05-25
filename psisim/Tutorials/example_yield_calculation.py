from psisim import telescope,instrument,observation,spectrum,universe,plots

wvs = [0.6,0.8,1.0,1.2,1.6] #Choose some wavelengths

tmt = telescope.TMT()
psi_blue = instrument.PSI_Blue()
psi_blue.set_observing_mode(60,40,'z',10) #60s, 40 exposures,z-band, R of 10

exosims_config_filename = "forBruceandDimitri_EXOCAT1.json" #Some filename here
uni = universe.ExoSims_Universe(exosims_config_filename)
uni.simulate_EXOSIMS_Universe()

planet_table = uni.planets
n_planets = len(planet_table)

planet_types = []
planet_spectra = []

n_planets_now = 5

for planet in planet_table[:n_planets_now]:

	#INSERT PLANET SELECTION RULES HERE
	planet_type = "Terrestrial"
	planet_types.append(planet_type)

	#Generate the spectrum and downsample
	atmospheric_parameters = spectrum.generate_picaso_inputs(planet,planet_type)
	planet_spectrum = spectrum.simulate_spectrum(planet,wvs,psi_blue.current_R,atmospheric_parameters)
	planet_spectrum = spectrum.downsample_spectrum(planet_spectrum,10000,psi_blue.current_R)
	planet_spectra.append(planet_spectrum)

sim_F_lambda, sim_F_lambda_errs = observation.simulate_observation_set(tmt,psi_blue,
	planet_table[:n_planets_now],planet_spectra,wvs,inject_noise=False,post_processing_gain=1000)

snrs = sim_F_lambda/sim_F_lambda_errs