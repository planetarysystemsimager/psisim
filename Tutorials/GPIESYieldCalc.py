from psisim import telescope,instrument,observation,spectrum,universe,plots
import numpy as np
import copy
import time
from astropy.io import fits
import astropy.units as u
import pysynphot as ps

# Set up the telescope and instrument
gemini = telescope.GeminiSouth()
gpi = instrument.GPI()
wvs,dwvs = gpi.get_effective_wavelength('H')
R = wvs/dwvs
gpi.set_observing_mode(60,38,'H',R,wvs,dwvs)

# Simulate GPIES stars and populate with planets
uni = universe.GPIES_Universe()
planet_table = uni.simulate_GPIES_Universe()
#This step expects wvs in
min_iwa = .12# in arcseconds
owa = 1.9 # in arcseconds
generated_planets = len(planet_table)
planet_table = planet_table[planet_table['AngSep'].to(u.arcsec).value > min_iwa]
planet_table = planet_table[planet_table['AngSep'].to(u.arcsec).value < owa]
detectable_planets = len(planet_table)
n_planets = len(planet_table)
if n_planets > 0:   
    planet_types = []
    planet_spectra = []

    n_planets_now = len(planet_table) # how many 

    # Generate spectra
    for planet in planet_table:
        #INSERT PLANET SELECTION RULES HERE
        planet_type = "Gas"
        planet_types.append(planet_type)

        
        ## get_stellar_spectrum expects wvs in microns
        planet_spectrum = spectrum.get_stellar_spectrum(planet,[wvs.value]*wvs.unit,R,model='COND+BTSettle_Broadband',verbose=False,user_params = None,doppler_shift=False,broaden=False,delta_wv=None)
        planet_spectra.append(planet_spectrum)
        

    # Simulate observations and calculate the SNR
    planet_spectra = np.array(planet_spectra)
    planet_table['StarAOmag'] = planet_table['StarHmag']
    post_processing_gain = []
    for i in range(len(planet_table)):
        post_processing = gpi.get_post_processing_gain(planet_table['AngSep'][i])
        post_processing_gain += [post_processing]
    sim_F_lambda, sim_F_lambda_errs, sim_F_lambda_stellar, noise_components = observation.simulate_observation_set(gemini,gpi, planet_table, planet_spectra, [wvs.value]*wvs.unit, R, inject_noise=False,post_processing_gain=post_processing_gain,return_noise_components=True)

    speckle_noises = noise_components[:,0]
    photon_noises = noise_components[:,2]

    flux_ratios = np.divide(sim_F_lambda,sim_F_lambda_stellar)
    flux_ratios
    detection_limits = np.divide(sim_F_lambda_errs,sim_F_lambda_stellar)

    snrs = np.divide(sim_F_lambda,sim_F_lambda_errs)
    snrs = snrs[:,None]

    detected = gpi.detect_planets(planet_table,snrs)
    detect = []
    for i in detected:
        detect += [i[0]]
    planet_table['Detected'] = detect
    planet_table['Contrast'] = flux_ratios
    planet_table['DetLim'] = detection_limits*5
    for i in planet_table:
        for j in range(len(i)):
            if j == 1 or j == 3 or j == 4 or j == 5 or j == 8 or j == 9:
                value = str(round(i[j].value,3))
            else:
                value = str(i[j])  
            with open('PlanetTables2.txt', 'a') as file:
                file.write('{0}\t'.format(value))
        with open('PlanetTables2.txt', 'a') as file:
            file.write('\n')
    detected_planets = 0
    for i in detected:
        if i == True:
            detected_planets += 1
else:
    detected_planets = 0

with open('GPIESYields2.txt', 'a') as file:
    file.write('\t{0:.0f}'.format(generated_planets) + '\t{0:.0f}'.format(detectable_planets) + '\t{0:.0f}'.format(detected_planets) + '\n')