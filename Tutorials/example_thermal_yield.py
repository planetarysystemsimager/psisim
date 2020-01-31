from psisim import telescope,instrument,observation,spectrum,universe,plots
import numpy as np
import matplotlib.pylab as plt
import copy
import time
import astropy.units as u
import os
import psisim

psisim_path = os.path.dirname(psisim.__file__)

tmt = telescope.TMT()
psi_red = instrument.PSI_Red()
psi_red.set_observing_mode(3600,2,'M',10, np.linspace(4.2,4.8,3)*u.micron) #3600s, 2 exposures,M-band, R of 10

######################################
######## Generate the universe #######
######################################

exosims_config_filename = psisim_path+"/../Tutorials/forBruceandDimitri_EXOCAT1.json" #Some filename here
uni = universe.ExoSims_Universe(exosims_config_filename)
uni.simulate_EXOSIMS_Universe()

##############################
######## Lots of setup #######
##############################

min_iwa = np.min(psi_red.current_wvs).to(u.m)/tmt.diameter*u.rad
planet_table = uni.planets
# planet_table = planet_table[np.where(planet_table['PlanetMass'] > 10)]
planet_table = planet_table[planet_table['AngSep'] > min_iwa.to(u.mas)]
planet_table = planet_table[planet_table['Flux Ratio'] > 1e-10]

n_planets = len(planet_table)

planet_types = []
planet_spectra = [] #The spectrum from the cooling tracks
planet_eq_spectra = [] #The spectrum from the equilibrium thermal emission
planet_ages = []

n_planets_now = n_planets
rand_planets = np.random.randint(0, n_planets, n_planets_now)

########### Model spectrum wavelength choice #############
# We're going to generate a model spectrum at a resolution twice the 
# requested resolution
intermediate_R = psi_red.current_R*2
#Choose the model wavelength range to be just a little bigger than 
#the observation wavelengths
model_wv_low = 0.9*np.min(psi_red.current_wvs) 
model_wv_high = 1.1*np.max(psi_red.current_wvs)

#Figure out a good wavelength spacing for the model
wv_c = 0.5*(model_wv_low+model_wv_high) #Central wavelength of the model
dwv_c = wv_c/intermediate_R #The delta_lambda at the central wavelength
#The number of wavelengths to generate. Divide by two for nyquist in the d_wv. 
#Multiply the final number by 2 just to be safe.
n_model_wv = int((model_wv_high-model_wv_low)/(dwv_c/2))*2
#Generate the model wavelenths
model_wvs = np.linspace(model_wv_low, model_wv_high, n_model_wv) #Choose some wavelengths

#Let's assume we're doing the AO sensing in I-band: 
planet_table['StarAOmag'] = planet_table['StarImag']

#####################################
######## Generate the Spectra #######
#####################################

print("\n Starting to generate planet spectra")
for planet in planet_table[rand_planets]:
    #INSERT PLANET SELECTION RULES HERE

    if planet['PlanetMass'] < 10:
        #If the planet is < 10 M_Earth, we don't trust bex. So we'll be pessimistic and just report its thermal equilibrium. 
        planet_type = "blackbody"
        planet_types.append(planet_type)

        #The bond albedo
        atmospheric_parameters = 0.5
        planet_spectrum = spectrum.simulate_spectrum(planet, model_wvs, intermediate_R, atmospheric_parameters, package='blackbody')
        planet_spectra.append(planet_spectrum)
        planet_eq_spectra.append(planet_spectrum)

    else:
        planet_type = "Gas"
        planet_types.append(planet_type)

        age = np.random.random() * 5e9 # between 0 and 5 Gyr
        planet_ages.append(age)

        time1 = time.time()

        ### Here we're going to generate the spectrum as the addition of cooling models and a blackbody (i.e. equilibrium Temperature)
    	## Generate the spectrum from cooling models and downsample to intermediate resolution
        atmospheric_parameters = age, 'M', True
        planet_spectrum = spectrum.simulate_spectrum(planet, model_wvs, intermediate_R, atmospheric_parameters, package='bex-cooling')

        ## Generate the spectrum from a blackbody
        atmospheric_parameters = 0.5#The bond albedo
        planet_eq_spectrum = np.array(spectrum.simulate_spectrum(planet, model_wvs, intermediate_R, atmospheric_parameters, package='blackbody'))
     
        planet_spectra.append(planet_spectrum)
        planet_eq_spectra.append(planet_eq_spectrum)
        
        time2 = time.time()
        print('Spectrum took {0:.3f} s'.format((time2-time1)))

print("Done generating planet spectra")
print("\n Starting to simulate observations")

#Here we take the biggest of either the planet cooling spectrum or the planet equilibrium spectrum
#Kind of hacky, but is a good start
final_spectra = np.array(copy.deepcopy(planet_spectra))
planet_eq_spectra = np.array(planet_eq_spectra)
planet_spectra = np.array(planet_spectra)
final_spectra[planet_eq_spectra > planet_spectra] = planet_eq_spectra[planet_eq_spectra > planet_spectra]


##########################################
######## Simulate the observations #######
##########################################

post_processing_gain=10
sim_F_lambda, sim_F_lambda_errs,sim_F_lambda_stellar, noise_components = observation.simulate_observation_set(tmt, psi_red,
	planet_table[rand_planets], final_spectra, model_wvs, intermediate_R, inject_noise=False,
	post_processing_gain=post_processing_gain,return_noise_components=True)

speckle_noises = np.array([s[0] for s in noise_components])
photon_noises = np.array([s[3] for s in noise_components])

flux_ratios = sim_F_lambda/sim_F_lambda_stellar
detection_limits = sim_F_lambda_errs/sim_F_lambda_stellar
snrs = sim_F_lambda/sim_F_lambda_errs

detected = psi_red.detect_planets(planet_table[rand_planets],snrs,tmt)


########################################
######## Make the contrast Plot ########
########################################

#Choose which wavelength you want to plot the detections at:
wv_index = 1

fig, ax = plots.plot_detected_planet_contrasts(planet_table[rand_planets],wv_index,
    detected,flux_ratios,psi_red,tmt,ymin=1e-8, ymax=1e-1,show=False)

#The user can now adjust the plot as they see fit. 
#e.g. Annotate the plot
# ax.text(4e-2,1e-5,"Planets detected: {}".format(len(np.where(detected[:,wv_index])[0])),color='k')
# ax.text(4e-2,0.5e-5,"Planets not detected: {}".format(len(np.where(~detected[:,wv_index])[0])),color='k')
# ax.text(4e-2,0.25e-5,"Post-processing gain: {}".format(post_processing_gain),color='k')
print("Planets detected: {}".format(len(np.where(detected[:,wv_index])[0])))
print("Planets not detected: {}".format(len(np.where(~detected[:,wv_index])[0])))
print("Post-processing gain: {}".format(post_processing_gain))
plt.show()

########################################
######## Make the magnitude Plot #######
########################################

## Choose which wavelength you want to plot the detections at:
# wv_index = 1

# fig, ax = plots.plot_detected_planet_magnitudes(planet_table[rand_planets],wv_index,
#     detected,flux_ratios,psi_red,tmt,show=False)

# #The user can now adjust the plot as they see fit. 
# #e.g. Annotate the plot
# # ax.text(4e-2,1e-5,"Planets detected: {}".format(len(np.where(detected[:,wv_index])[0])),color='k')
# # ax.text(4e-2,0.5e-5,"Planets not detected: {}".format(len(np.where(~detected[:,wv_index])[0])),color='k')
# # ax.text(4e-2,0.25e-5,"Post-processing gain: {}".format(post_processing_gain),color='k')
# print("Planets detected: {}".format(len(np.where(detected[:,wv_index])[0])))
# print("Planets not detected: {}".format(len(np.where(~detected[:,wv_index])[0])))
# print("Post-processing gain: {}".format(post_processing_gain))
# plt.show()

###########################################
######## Recalculate the magnitudes #######
###########################################


# dMags = -2.5*np.log10(flux_ratios[:,wv_index])    

# band = psi_red.current_filter
# if band == 'R':
#     bexlabel = 'CousinsR'
#     starlabel = 'StarRmag'
# elif band == 'I':
#     bexlabel = 'CousinsI'
#     starlabel = 'StarImag'
# elif band == 'J':
#     bexlabel = 'SPHEREJ'
#     starlabel = 'StarJmag'
# elif band == 'H':
#     bexlabel = 'SPHEREH'
#     starlabel = 'StarHmag'
# elif band == 'K':
#     bexlabel = 'SPHEREKs'
#     starlabel = 'StarKmag'
# elif band == 'L':
#     bexlabel = 'NACOLp'
#     starlabel = 'StarKmag'
# elif band == 'M':
#     bexlabel = 'NACOMp'
#     starlabel = 'StarKmag'
# else:
#     raise ValueError("Band needs to be 'R', 'I', 'J', 'H', 'K', 'L', 'M'. Got {0}.".format(band))

# stellar_mags = planet_table[starlabel]
# stellar_mags = np.array(stellar_mags)

# planet_mag = stellar_mags+dMags

# plt.figure()
# plt.plot(planet_mag,np.log10(masses),'o',alpha=0.3)   
# plt.xlabel("M-band Magnitude")
# plt.ylabel("Log10(Mass)")
# plt.show()

# plt.figure()
# plt.plot(np.log10(flux_ratios[:,wv_index]),np.log10(masses),'o',alpha=0.3) 
# plt.xlabel("log10(Flux Ratios")
# plt.ylabel("Log10(Mass)")
# plt.show()

# sim_F_lambda

# plt.figure()
# plt.plot((sim_F_lambda[:,wv_index]),np.log10(masses),'o',alpha=0.3) 
# plt.xlabel("log10(sim_F_lambda)")
# plt.ylabel("Log10(Mass)")
# plt.xlim(0,1e8)
# plt.show()


############################################################
######## Histogram of detections and non-detections ########
############################################################

### And now we'll make a simple histogram of detected vs. generated planet mass. 
masses = [planet['PlanetMass'] for planet in planet_table]
detected_masses = [planet['PlanetMass'] for planet in planet_table[detected[:,1]]] #Picking 4.5 micron

bins = np.logspace(np.log10(1),np.log10(1000),20)
fig,axes = plt.subplots(2,1)
hist_detected = axes[0].hist(detected_masses,bins=bins,color='darkturquoise',histtype='step',label="Detected",linewidth=3.5)
hist_all = axes[0].hist(masses,color='k',bins=bins,histtype='step',label="Full Sample",linewidth=3.5)
axes[0].set_xscale("log")
axes[0].set_ylabel("Number of planets")
axes[0].set_xlabel(r"Planet Mass [$M_{\oplus}$]")
axes[0].legend()

efficiency = hist_detected[0]/hist_all[0]
efficiency[efficiency!=efficiency]=0
axes[1].step(np.append(hist_detected[1][0],hist_detected[1]),np.append(0,np.append(efficiency,0)),where='post',linewidth=3)
axes[1].set_ylabel("Detection Efficiency")
#Stupid hack
axes[1].set_xscale("log")
axes[1].set_xlabel(r"Planet Mass [$M_{\oplus}$]")
fig.suptitle(r"Thermal Emission Detections at {:.1f}$\mu m$".format(psi_red.current_wvs[1]))
plt.show()

##################################
######## Save the results ########
##################################

from astropy.io import fits
planet_table[rand_planets].write("thermal_planet_table.csv")
ps_hdu = fits.PrimaryHDU(planet_spectra)
ps_hdu.writeto("thermal_planet_spectra.fits",overwrite=True)
flux_hdu = fits.PrimaryHDU([sim_F_lambda, sim_F_lambda_errs,np.array(sim_F_lambda_stellar)])
flux_hdu.writeto("thermal_Observation_set.fits",overwrite=True)
noise_components_hdu = fits.PrimaryHDU(noise_components)
noise_components_hdu.writeto("thermal_noise_components.fits",overwrite=True)

