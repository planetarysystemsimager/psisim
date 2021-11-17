from psisim import telescope,instrument,observation,spectrum,universe,plots,signal
import time
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from importlib import reload
import speclite.filters
from scipy.interpolate import interp1d, RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import median_filter
from scipy.signal import medfilt, correlate
from numpy.random import poisson, randn
import copy 

#Do you want to make plots?
plot = False

## Some initial setup
filters = spectrum.load_filters()
settings_file = "hispec_settings.txt"
settings = {}
with open(settings_file) as f:
    for line in f:
        line = str(line)
        if line[0] == "#":
            continue
        (key, val) = line.split(" ",2)[:2]
        val = val.strip("\n")
        if key == "#":
            pass
        else: 
            settings[key] = val

## TODO: Add checks to make sure the minimum required settings are present

### Telescope Setup
keck = telescope.Keck(path=settings['path'])
if settings.has_key('airmass'): keck.airmass = float(settings['airmass'])
if settings.has_key('water_vapor'): keck.water_vapor = float(settings['water_vapor'])
if settings.has_key('seeing'):
    keck.seeing = float(settings['seeing'])
else: 
    keck.seeing = keck.median_seeing

### Instrument Setup
hispec = instrument.hispec(telescope=keck)
hispec.set_current_filter(settings['filter'])

#Get the set of wavelengths based on the current instrument setup
wavelengths = hispec.get_wavelength_range()

# Set the observing mode: Exposure time (per exposure), Number of Exposures,filter name, wavelength array
hispec.set_observing_mode(float(settings['itime']),int(settings['nint']),
                          settings['filter'], wavelengths,mode=settings['mode'])

#First set the host properties for a Phoenix model. 
host_properties = {"StarLogg":float(settings['starlogg'])*u.dex(u.cm/ u.s**2),
                   "StarTeff":int(settings['starteff'])*u.K,
                   "StarZ":settings['starZ'],
                   "StarAlpha":settings['starAlpha'],
                   "StarRadialVelocity":float(settings['starRV'])*u.km/u.s,
                   "StarVsini":float(settings['starvsini'])*u.km/u.s,
                   "StarLimbDarkening":float(settings['starlimbdarkening'])}


#Now setup the user parameters that a Phoenix model needs: (path, object_filter, magnitude_in_object_filter,
# filters_object,current_filter). 
host_user_params = (settings['path'],settings['filter'],settings['starmag'],
                    filters,hispec.current_filter)

#Generate the spectrum! (Here we apply a doppler shift and rotationally broaden)
host_spectrum = spectrum.get_stellar_spectrum(host_properties,wavelengths,
                                              hispec.current_R,model="Phoenix",
                                              user_params=host_user_params,
                                              doppler_shift=True,broaden=True,
                                              delta_wv=hispec.current_dwvs)

if plot: 
    plt.figure(figsize=(20,10))

    plt.semilogy(host_spectrum.wvs,host_spectrum.spectrum)

    plt.xlabel("Wavelength [{}]".format(host_spectrum.wvs.unit))
    plt.ylabel("Spectrum [{}]".format(host_spectrum.spectrum.unit))


## Generate the companion spectrum now: 
obj_properties = {"StarLogg":float(settings['logg'])*u.dex(u.cm/ u.s**2),
                  "StarTeff":int(settings['teff'])*u.K,
                  "StarRadialVelocity":float(settings['rv'])*u.km/u.s,
                  "StarVsini":float(settings['vsini'])*u.km/u.s,
                  "StarLimbDarkening":float(settings['limbdarkening'])}

obj_user_params = (path,'TwoMASS-K',float(settings['mag']),filters,hispec.current_filter)

obj_spectrum = spectrum.get_stellar_spectrum(obj_properties,wavelengths,
                                             hispec.current_R,
                                             model="Sonora",
                                             user_params=obj_user_params,
                                             doppler_shift=True,broaden=True,
                                             delta_wv=hispec.current_dwvs)

#Plot the object spectrum
if plot: 
    plt.figure(figsize=(30,10))

    plt.semilogy(obj_spectrum.wvs,obj_spectrum.spectrum)

    plt.xlabel("Wavelength [{}]".format(obj_spectrum.wvs.unit))
    plt.ylabel("Spectrum [{}]".format(obj_spectrum.spectrum.unit))
    plt.ylim(1e-10,1e-4)

#PSISIM wants the object spectrum in contrast units
obj_spectrum.spectrum /= host_spectrum.spectrum

## Simulate the Observation
# The angular separation of the companion, in milliarcsecond
host_properties['AngSep'] = float(settings['angsep']) *u.mas

#Get the host star magnitude in the AO filter
host_properties["StarAOmag"] = spectrum.get_model_ABmags(host_properties,[hispec.ao_filter], model='Phoenix',
                                                         verbose=False,user_params = host_user_params)
hispec.ao_mag = host_properties["StarAOmag"]

#Hispec doesn't care about the spectral type, but we need to include the paramter
host_properties['StarSpT'] = None


###############################################################
######## Cycle through all the filters and simulate ###########
###############################################################
all_wavelengths = []
full_host_spectrum = []
full_obj_spectrum = []
full_obj_spectrum_nosky = []
full_total_noise = []
full_thermal_spec = []
full_noise_components = []
full_speckle_noise = []


for hispec_filter in hispec.filters:
    
    #Setup the instrument
    hispec.set_current_filter(hispec_filter)
    wavelengths = hispec.get_wavelength_range()
    hispec.set_observing_mode(3600,1,hispec_filter, wavelengths) 
    
    
    host_user_params = (path,'TwoMASS-J',5.0,filters,hispec.current_filter)
    host_spectrum = spectrum.get_stellar_spectrum(host_properties,wavelengths,hispec.current_R,
                                                  model="Phoenix",user_params=host_user_params,
                                                  doppler_shift=True,broaden=True,delta_wv=hispec.current_dwvs)
    
    obj_user_params = (path,'TwoMASS-K',20,filters,hispec.current_filter)
    obj_spectrum = spectrum.get_stellar_spectrum(obj_properties,wavelengths,hispec.current_R,model="Sonora",
                                                 user_params=obj_user_params,doppler_shift=True,broaden=True,
                                                 delta_wv=hispec.current_dwvs)
    
    obj_spectrum.spectrum /= host_spectrum.spectrum
    
    obj_spec,total_noise,stellar_spec,thermal_spec,noise_components= observation.simulate_observation(keck,hispec,
                                                                                      host_properties,
                                                                                      obj_spectrum.spectrum,wavelengths,1e5,
                                                                                      inject_noise=False,verbose=True,
                                                                                      post_processing_gain = np.inf,
                                                                                      return_noise_components=True,
                                                                                      stellar_spec=host_spectrum.spectrum,
                                                                                      apply_lsf=True,
                                                                                      integrate_delta_wv=False,
                                                                                      plot=False,
                                                                                      sky_on=True)
    obj_spec_no_sky,_,_,_ = observation.simulate_observation(keck,hispec,
                                                                                      host_properties,
                                                                                      obj_spectrum.spectrum,wavelengths,1e5,
                                                                                      inject_noise=False,verbose=True,
                                                                                      post_processing_gain = np.inf,
                                                                                      return_noise_components=False,
                                                                                      stellar_spec=host_spectrum.spectrum,
                                                                                      apply_lsf=True,
                                                                                      integrate_delta_wv=False,
                                                                                      plot=False,
                                                                                      sky_on=False)

    full_speckle_noise.append(hispec.get_speckle_noise(0.4*u.arcsecond,host_properties["StarAOmag"],hispec.current_filter,
                                             wavelengths,host_properties['StarSpT'],keck)[0])
    all_wavelengths.append(wavelengths)
    full_host_spectrum.append(stellar_spec)
    full_obj_spectrum.append(obj_spec)
    full_obj_spectrum_nosky.append(obj_spec_no_sky)
    full_total_noise.append(total_noise)
    full_thermal_spec.append(thermal_spec)
    full_noise_components.append(noise_components)

all_wavelengths = np.hstack(all_wavelengths).value*wavelengths.unit
full_host_spectrum = np.hstack(full_host_spectrum).value*stellar_spec.unit
full_obj_spectrum = np.hstack(full_obj_spectrum).value*obj_spec.unit
full_obj_spectrum_nosky = np.hstack(full_obj_spectrum_nosky).value*obj_spec_no_sky.unit
full_total_noise = np.hstack(full_total_noise).value*total_noise.unit
full_noise_components = np.hstack(full_noise_components)*obj_spec.unit
full_thermal_spec = np.hstack(full_thermal_spec).value*thermal_spec.unit
full_speckle_noise = np.hstack(full_speckle_noise)


if plot: 
    plt.figure(figsize=(30,10))

    plt.plot(all_wavelengths,full_obj_spectrum,label="Spectrum")
    plt.plot(all_wavelengths,full_total_noise,label="Total Stastistical Noise Level")

    plt.legend()

    plt.xlabel("Wavelength [{}]".format(all_wavelengths.unit))
    plt.ylabel("Spectrum [{}]".format(full_obj_spectrum.unit))
    plt.title(r"HISPEC Observation of Sonora grid $T_{{eff}}$ = {}K,logg = {}, Exp. Time = {}, $N_{{exp}}$= {}".format(
    obj_properties["StarTeff"].value,obj_properties["StarLogg"].value,hispec.exposure_time,hispec.n_exposures))
    plt.ylim(1e-4,1e2)
    plt.grid()