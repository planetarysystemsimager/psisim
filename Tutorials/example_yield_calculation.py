from psisim import telescope,instrument,observation,spectrum,universe,plots
import multiprocessing as mp
import numpy as np
import matplotlib.pylab as plt
import copy
import time
from astropy.io import fits
import astropy.units as u


### Set up the telescope and observing mode ###
tmt = telescope.TMT()
psi_blue = instrument.PSI_Blue()
psi_blue.set_observing_mode(3600*u.s,10,'z',50, np.linspace(0.60,0.85,40)*u.micron) #60s, 40 exposures,z-band, R of 10

### Generate the universe using EXOSIMS ###
exosims_config_filename = "default_PSISIM_EXOSIMS_universe.json" #Some filename here
uni = universe.ExoSims_Universe(exosims_config_filename)
uni.simulate_EXOSIMS_Universe()
planet_table = uni.planets
full_planet_table = copy.deepcopy(uni.planets)

### Down select the planets whose separations are less than lambda/D ###
min_iwa = np.min(psi_blue.current_wvs).to(u.m)/tmt.diameter*u.rad
planet_table = planet_table[planet_table['AngSep'] > min_iwa.to(u.mas)]
planet_table = planet_table[planet_table['Flux Ratio'] > 1e-10]
n_planets = len(planet_table)


### Model the spectra! ###
planet_types = []
planet_spectra = []

n_planets_now = 2
#We'll pick random planets, since many systems are multi-planet systems and they show up 
#sequentially in EXOSIMS
rand_planets = np.random.randint(0, n_planets, n_planets_now)

## Model spectrum wavelength choice based on instrument setup ##
# We're going to generate a model spectrum at a resolution twice the 
# requested resolution
intermediate_R = psi_blue.current_R*2
#Choose the model wavelength range to be just a little bigger than 
#the observation wavelengths
model_wv_low = 0.9*np.min(psi_blue.current_wvs) 
model_wv_high = 1.1*np.max(psi_blue.current_wvs)

#Figure out a good wavelength spacing for the model
wv_c = 0.5*(model_wv_low+model_wv_high) #Central wavelength of the model
dwv_c = wv_c/intermediate_R #The delta_lambda at the central wavelength
#The number of wavelengths to generate. Divide by two for nyquist in the d_wv. 
#Multiply the final number by 2 just to be safe.
n_model_wv = int((model_wv_high-model_wv_low)/(dwv_c/2))*2
#Generate the model wavelenths
model_wvs = np.linspace(model_wv_low, model_wv_high, n_model_wv) #Choose some wavelengths

print("\n Starting to generate planet spectra")

#Load in the PICASO opacities
opacity_file = None
opacities = spectrum.load_picaso_opacity(dbname=opacity_file)

def generate_spectrum(planet):
    """
    Function to loop over to generate spectra. 
    """
    #INSERT PLANET SELECTION RULES HERE
    if planet["PlanetMass"] <= 17*u.earthMass: #Planet masses in Earth Masses
        #If Neptune or less, we'll just use the Exosims Flux Ratio for now at all wavelengths. 
        #Later this will also use PICASO once we learn how to use it better
        planet_type = "Exosims"
        planet_spectrum = np.repeat(planet['Flux Ratio'],len(model_wvs))
        
    else:
        planet_type = "Gas"
        time1 = time.time()
    	#Generate the spectrum and downsample to intermediate resolution
        atmospheric_parameters = spectrum.generate_picaso_inputs(planet,planet_type, opacities,clouds=True)
        planet_spectrum = spectrum.simulate_spectrum(planet, model_wvs, intermediate_R, atmospheric_parameters)
        
        time2 = time.time()
        print('Spectrum took {0:.3f} s'.format((time2-time1)))

    return planet_type, planet_spectrum

### Non parallel Code 
for planet in planet_table[rand_planets]:
    planet_type, planet_spectrum = generate_spectrum(planet,)
    planet_types.append(planet_type)
    planet_spectra.append(planet_spectrum)

### Parallel Code
# pool = mp.Pool(processes=2) # pick the number of processes you want to use
# outputs = pool.map(generate_spectrum, planet_table[rand_planets])

# for planet_type, planet_spectrum in outputs:
#     planet_types.append(planet_type)
#     planet_spectra.append(planet_spectrum)


print("Done generating planet spectra")
print("\n Starting to simulate observations")


######################## Simulate the Spectra ######################
#Here we'll assume that the AO system is depends on v-band magnitude
planet_table['StarAOmag'] = planet_table['StarVmag']

planet_spectra = np.array(planet_spectra)
post_processing_gain=100
sim_F_lambda, sim_F_lambda_errs,sim_F_lambda_stellar, noise_components = observation.simulate_observation_set(tmt, psi_blue,
	planet_table[rand_planets], planet_spectra, model_wvs, intermediate_R, inject_noise=False,
	post_processing_gain=post_processing_gain,return_noise_components=True)

# import pdb; pdb.set_trace()
noise_components = np.array(noise_components)
speckle_noises = noise_components[:,0,:]
photon_noises = noise_components[:,3,:]

flux_ratios = sim_F_lambda/sim_F_lambda_stellar
detection_limits = sim_F_lambda_errs/sim_F_lambda_stellar
snrs = sim_F_lambda/sim_F_lambda_errs


######################## Plot Yield ######################
detected = psi_blue.detect_planets(planet_table[rand_planets],snrs,tmt)

#Choose which wavelength you want to plot the detections at:
wv_index = 20
fig, ax = plots.plot_detected_planet_contrasts(planet_table[rand_planets],wv_index,
	detected,flux_ratios,psi_blue,tmt,ymin=1e-13,alt_label=r"5-$\sigma$ Detection Limits", show=False)

#The user can now adjust the plot as they see fit. 
#e.g. Annotate the plot
# ax.text(2e-2,1e-5,"Planets detected: {}".format(len(np.where(detected[:,wv_index])[0])),color='k')
# ax.text(2e-2,0.5e-5,"Planets not detected: {}".format(len(np.where(~detected[:,wv_index])[0])),color='k')
# ax.text(2e-2,0.25e-5,"Post-processing gain: {}".format(post_processing_gain),color='k')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(4e-3,3)
ax.set_ylim(5e-12,5e-6)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()

######################## Save things ######################
planet_table[rand_planets].write("planet_table.csv")
ps_hdu = fits.PrimaryHDU(planet_spectra)
ps_hdu.writeto("planet_spectra.fits",overwrite=True)
flux_hdu = fits.PrimaryHDU([sim_F_lambda, sim_F_lambda_errs,np.array(sim_F_lambda_stellar)])
flux_hdu.writeto("Observation_set.fits",overwrite=True)
noise_components_hdu = fits.PrimaryHDU(noise_components)
noise_components_hdu.writeto("noise_components.fits",overwrite=True)

# #HOWTO: Read things back in
# from astropy.io import ascii
# planet_table = ascii.read("planet_table.csv")
# planet_spectra = fits.open("planet_spectra.fits")[0].data
# tmp = fits.open("Observation_set.fits")[0].data
# sim_F_lambda = tmp[0]
# sim_F_lambda_errs = tmp[1]
# sim_F_lambda_stellar = tmp[2]
# noise_components = fits.open("noise_components.fits")[0].data
######################## Plot Cloud vs. Clear ######################


# get the best SNR Planet
avg_snrs = np.mean(snrs, axis=1)
print(avg_snrs)
argsort_snrs = np.argsort(np.abs(avg_snrs - 6))
bestsnr = argsort_snrs[0] #np.argmax(avg_snrs)

# Generate the cloudy spectrum of this planet
planet = planet_table[rand_planets[bestsnr]]
atmospheric_parameters_clear = spectrum.generate_picaso_inputs(planet, planet_type, clouds=False)
planet_spectrum_clear = spectrum.simulate_spectrum(planet, model_wvs, intermediate_R, atmospheric_parameters_clear)

# Generate noisy spectra for cloudy and clear
clear_F_lambda, clear_F_lambda_errs, _ = observation.simulate_observation(tmt, psi_blue,
   planet_table[rand_planets[bestsnr]], planet_spectrum_clear, model_wvs, intermediate_R, inject_noise=True,post_processing_gain=post_processing_gain)
cloudy_F_lambda, cloudy_F_lambda_errs, _ = observation.simulate_observation(tmt, psi_blue,
   planet_table[rand_planets[bestsnr]], planet_spectra[bestsnr], model_wvs, intermediate_R, inject_noise=True,post_processing_gain=post_processing_gain)

fig = plt.figure()

plt.errorbar(psi_blue.current_wvs*1000, clear_F_lambda, yerr=clear_F_lambda_errs, color='Blue', marker='o', linestyle='none', label="Clear", zorder=1)
plt.errorbar(psi_blue.current_wvs*1000, cloudy_F_lambda, yerr=cloudy_F_lambda_errs, color='Gray', marker='o', linestyle='none', label="Cloudy", zorder=1)


plt.grid()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Flux (counts)")
plt.legend()
plt.tight_layout()

plt.show()
