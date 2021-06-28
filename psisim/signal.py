import numpy as np
from scipy.signal import medfilt, correlate
from numpy.random import poisson, randn
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator as rgi
import astropy.units as u
import warnings

#Compute CCF SNR based on match filter
def compute_ccf_snr_matchedfilter(signal, model, total_noise, sky_trans, systematics_residuals=0.01,kernel_size=501,norm_cutoff=0.8):
    '''
    Calculate the Cross-correlation function signal to noise ration with a matched filter

    Inputs:
    signal      - Your observed spectrum
    model       - Your model spectrum
    total_noise - Your total noise estimate in the same units as your signal
    sky_trans   - The sky transmission
    systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
    kernel_size  - The default high-pass filter size.
    norm_cutoff  - A cutoff below which we don't calculate the ccf-snr
    '''
    #Get the noise variance
    total_noise_var = total_noise**2 
    bad_noise = np.isnan(total_noise_var)
    total_noise_var[bad_noise]=np.inf

    #Calculate some normalization factor
    #Dimitri to explain this better. 
    norm = ((1-systematics_residuals)*sky_trans)
    
    #Get a median-filtered version of your model spectrum
    model_medfilt = medfilt(model,kernel_size=kernel_size)
    #Subtract the median version from the original model, effectively high-pass filtering the model
    model_filt = model.value-model_medfilt
    model_filt[np.isnan(model_filt)] = 0.
    model_filt[norm<norm_cutoff] = 0.
    model_filt[bad_noise] = 0.

    #Divide out the sky transmision
    normed_signal = signal/norm
    #High-pass filter like with the model
    signal_medfilt = medfilt(normed_signal,kernel_size=kernel_size)
    signal_filt = normed_signal.value-signal_medfilt
    signal_filt[np.isnan(signal_filt)] = 0.
    signal_filt[norm<norm_cutoff] = 0.
    signal_filt[bad_noise] = 0.
    
    #Now the actual ccf_snr
    ccf_snr = np.sqrt((np.sum(signal_filt * model_filt/total_noise_var))**2 / np.sum(model_filt * model_filt/total_noise_var))

    return ccf_snr

# Compute exposure time required to achieve a specific CCF SNR
def compute_exp_time_to_ccf_snr_matchedfilter(signal, model, photon_noise, read_noise, systematics, sky_trans, instrument, goal_ccf, systematics_residuals=0.01, kernel_size=501, norm_cutoff=0.8):
    '''
    Calculate the time required to achieve a desired CCF SNR with a matched filter

    Inputs:
    signal      - Your observed spectrum
    model       - Your model spectrum
    photon_noise - photon noise as returned by simulate_observation()
    read_noise  - read noise as returned by simulate_observation()
    systematics - systematic noise: (cal*(host_flux_at_obj+thermal_spec))**2
    sky_trans   - The sky transmission
    instrument  - a psisim instrument object
    goal_ccf    - CCF SNR for which exposure time will be computed
    systematics_residuals - A multiplicative factor that estimates the residual level of the host star spectrum and telluric lines in your signal (Default of 1%)
    kernel_size  - The default high-pass filter size.
    norm_cutoff  - A cutoff below which we don't calculate the ccf-snr
    '''
    # TODO: This function does not account for read_noise or systematics at the moment
      # To account for read_noise, we need to change how the number of frames is done in PSISIM
      # For systematics, we need to find a nice way to invert the CCF SNR equation when systematics are present
    warnings.warn('This function is incomplete at the moment. Double check all results for accuracy.')
    
    # Compute total obs. time from instrument object
    obs_time = (instrument.n_exposures * instrument.exposure_time).value
    
    # Remove time to get flux
    signal = signal / obs_time
    model  = model / obs_time
    
    #Get the noise variance
    total_noise_flux = (photon_noise**2 /obs_time) #+ (read_noise**2/instrument.n_exposures) #+ (systematics/ (obs_time**2))
    bad_noise = np.isnan(total_noise_flux)
    total_noise_flux[bad_noise]=np.inf

    #Calculate some normalization factor
    #Dimitri to explain this better. 
    norm = ((1-systematics_residuals)*sky_trans)
    
    #Get a median-filtered version of your model spectrum
    model_medfilt = medfilt(model,kernel_size=kernel_size)
    #Subtract the median version from the original model, effectively high-pass filtering the model
    model_filt = model.value-model_medfilt
    model_filt[np.isnan(model_filt)] = 0.
    model_filt[norm<norm_cutoff] = 0.
    model_filt[bad_noise] = 0.

    #Divide out the sky transmision
    normed_signal = signal/norm
    #High-pass filter like with the model
    signal_medfilt = medfilt(normed_signal,kernel_size=kernel_size)
    signal_filt = normed_signal.value-signal_medfilt
    signal_filt[np.isnan(signal_filt)] = 0.
    signal_filt[norm<norm_cutoff] = 0.
    signal_filt[bad_noise] = 0.
    
    #Now the actual ccf_snr
    min_exp_time = goal_ccf**2 / ((np.sum(signal_filt * model_filt/total_noise_flux))**2 / np.sum(model_filt * model_filt/total_noise_flux))

    return min_exp_time

# The following code is built around using stacked arrays
# If you are using multiple filters for example, pass through inputs like:
	#wave = np.array(all_wavelengths[blue], all_wavelenghts[red]) split according to the wavelength range of your filters

# Reduce the Noisy Spectrum and Cross-Correlate with Observed Spectrum
def reduce_spec(wave,delta_lb,samp,noisy_spec,model,photon_flux_atobj,photon_flux_thermal,ron,exp_time,sky_trans,cal, kernel_sz, norm_cutoff):
    '''
    Reduce Observed Spectra and Cross-Correlate

    Inputs:
	wave				- Your wavelength array
	delta_lb			- Array of Resolutions per filter (defined from Dmitiri's notebook - I don't know the actual values for Keck)
	samp				- Spacial sampling array per filter (np.array([3,3,3,3]))
	noisy_spec			- Simulated noisy spectrum
	model				- Model Spectrum (no sky)
	photon_flux_atobj	- Photon flux at the object, stellar flux multiplied bu contrast there
	photon_flux_thermal	- Thermal background and dark current photon flux
	ron					- Readout Noise
	exp_time			- Exposure time in units of seconds
	sky_trans			- The sky transmission
	cal					- Telluric calibration accuracy
	kernel_sz			- The default high-pass filter size
    norm_cutoff         - A cutoff below which we don't calculate the ccf-snr

    Returns:
	noisy_spec_filt		- Filtered Noisy Spectrum
	model_filt			- Filtered Model Spectrum 
	ccf					- Correlation of Noisy Spectrum with Model Spectrum
	ccf_model			- Correlation of Model with Model
	velocity			- Velocity in km/s
    '''

    noisy_spec_filt = np.copy(wave)
    model_filt = np.copy(wave)
    ccf = np.copy(wave)
    ccf_model = np.copy(wave)
    velocity = np.copy(wave)

    for i in range(wave.shape[0]):
        #This background subtraction was reducing the data to almost zero, so I got rid of it for now
        
        #total_background = photon_flux_atobj[i] * exp_time + photon_flux_thermal[i] * exp_time 
		#Background subtraction
        #noisy_spec_tmp = (noisy_spec[i] * exp_time - (1-cal) * total_background) / ((1-cal) * photon_flux_atobj[i] * exp_time)
        #noisy_spec_tmp *= u.electron
		# noisy_spec_tmp = (noisy_spec[i] - photon_flux_thermal[i].value * exp_time.value) / (photon_flux_atobj[i].value * exp_time.value)

		#Normalization by sky transmission
        norm = ((1-cal)*sky_trans[i])
        index=np.where(norm < norm_cutoff)
        noisy_spec[i] = noisy_spec[i]/norm
        noisy_spec[i][index] = 0.

		#High-pass filter
        noisy_spec_medfilt=median_filter(noisy_spec[i],kernel_sz,mode='nearest')
        noisy_spec_filt[i]=(noisy_spec[i]/noisy_spec[i].unit-noisy_spec_medfilt) * noisy_spec[i].unit
        noisy_spec_filt[i][np.isnan(noisy_spec_filt[i])]=0    
        noisy_spec_filt[i][index]=0    

        model_medfilt=median_filter(model[i],kernel_sz,mode='nearest')
        model_filt[i]=(model[i]/model[i].unit-model_medfilt) * model[i].unit
        model_filt[i][np.isnan(model_filt[i])]=0 
        model_filt[i][index]=0
		#Correlation
        ccf[i]=correlate(noisy_spec_filt[i],model_filt[i])
        ccf_model[i]=correlate(model_filt[i],model_filt[i])
        c=3e5 * u.km/u.s
        dvelocity = np.mean(delta_lb[i] * c / wave[i])
        velocity[i]=(np.arange(ccf_model[i].shape[0])-ccf_model[i].shape[0]/2) * dvelocity
		
    return noisy_spec_filt, model_filt, ccf, ccf_model, velocity

#Plot reduced Spectra with Feature lines
# plotting the features
def plot_features(wave,spec_model_filt,noisy_spec_filt,features):
    '''
    Plot the Reduced Data with selected atomic features

    Inputs:
    wave                 - Wavelength array
    spec_model_filt      - Filtered Model Spectrum (no sky)
    noisy_spec_filt      - Filtered Noisy spectrum
    features             - List of desired plotting features
    '''

    fontsize=18
    feature_labels = { \
        'h2o': {'label': r'H$_2$O', 'type': 'band', 'wavelengths': [[0.92,0.95],[1.08,1.20],[1.325,1.550],[1.72,2.14],[3.0,3.5]]}, \
        'ch4': {'label': r'CH$_4$', 'type': 'band', 'wavelengths': [[1.1,1.24],[1.28,1.44],[1.6,1.76],[2.2,2.35],[3.2,3.6]]}, \
        'c2h2': {'label': r'C$_2$H$_2$', 'type': 'band', 'wavelengths': [[3.6,3.9]]}, \
        'hcn': {'label': r'HCN', 'type': 'band', 'wavelengths': [[3.55,3.65],[3.75,3.95]]}, \
        'ch3d': {'label': r'CH$_3$D', 'type': 'band', 'wavelengths': [[3.21,4.0]]}, \
        'co': {'label': r'CO', 'type': 'band', 'wavelengths': [[2.29,2.39]]}, \
        'tio': {'label': r'TiO', 'type': 'band', 'wavelengths': [[0.76,0.80],[0.825,0.831]]}, \
        'vo': {'label': r'VO', 'type': 'band', 'wavelengths': [[1.04,1.08]]}, \
        'young vo': {'label': r'VO', 'type': 'band', 'wavelengths': [[1.17,1.20]]}, \
#        'feh': {'label': r'FeH', 'type': 'band', 'wavelengths': [[0.86,0.90],[0.98,1.03],[1.19,1.25],[1.57,1.64]]}, \
        'feh': {'label': r'FeH', 'type': 'band', 'wavelengths': [[0.98,1.03],[1.19,1.25],[1.57,1.64]]}, \
        'h2': {'label': r'H$_2$', 'type': 'band', 'wavelengths': [[1.5,2.4]]}, \
        'sb': {'label': r'*', 'type': 'band', 'wavelengths': [[1.6,1.64]]}, \
        'h': {'label': r'H I', 'type': 'line', 'wavelengths': [[1.004,1.005],[1.093,1.094],[1.281,1.282],[1.944,1.945],[2.166,2.166]]},\
        'hi': {'label': r'H I', 'type': 'line', 'wavelengths': [[1.004,1.005],[1.093,1.094],[1.281,1.282],[1.944,1.945],[2.166,2.166]]},\
        'h1': {'label': r'H I', 'type': 'line', 'wavelengths': [[1.004,1.005],[1.093,1.094],[1.281,1.282],[1.944,1.945],[2.166,2.166]]},\
        'na': {'label': r'Na I', 'type': 'line', 'wavelengths': [[0.8186,0.8195],[1.136,1.137],[2.206,2.209]]}, \
        'nai': {'label': r'Na I', 'type': 'line', 'wavelengths': [[0.8186,0.8195],[1.136,1.137],[2.206,2.209]]}, \
        'na1': {'label': r'Na I', 'type': 'line', 'wavelengths': [[0.8186,0.8195],[1.136,1.137],[2.206,2.209],[2.3378945,2.3378945]]}, \
        'mg': {'label': r'Mg I', 'type': 'line', 'wavelengths': [[1.7113336,1.7113336],[1.5745017,1.5770150],[1.4881595,1.4881847,1.5029098,1.5044356],[1.1831422,1.2086969],]}, \
        'mgi': {'label': r'Mg I', 'type': 'line', 'wavelengths': [[1.7113336,1.7113336],[1.5745017,1.5770150],[1.4881595,1.4881847,1.5029098,1.5044356],[1.1831422,1.2086969],]}, \
        'mg1': {'label': r'Mg I', 'type': 'line', 'wavelengths': [[1.7113336,1.7113336],[1.5745017,1.5770150],[1.4881595,1.4881847,1.5029098,1.5044356],[1.1831422,1.2086969],]}, \
        'ca': {'label': r'Ca I', 'type': 'line', 'wavelengths': [[2.263110,2.265741],[1.978219,1.985852,1.986764],[1.931447,1.945830,1.951105]]}, \
        'cai': {'label': r'Ca I', 'type': 'line', 'wavelengths': [[2.263110,2.265741],[1.978219,1.985852,1.986764],[1.931447,1.945830,1.951105]]}, \
        'ca1': {'label': r'Ca I', 'type': 'line', 'wavelengths': [[2.263110,2.265741],[1.978219,1.985852,1.986764],[1.931447,1.945830,1.951105]]}, \
        'caii': {'label': r'Ca II', 'type': 'line', 'wavelengths': [[1.184224,1.195301],[0.985746,0.993409]]}, \
        'ca2': {'label': r'Ca II', 'type': 'line', 'wavelengths': [[1.184224,1.195301],[0.985746,0.993409]]}, \
        'al': {'label': r'Al I', 'type': 'line', 'wavelengths': [[1.672351,1.675511],[1.3127006,1.3154345]]}, \
        'ali': {'label': r'Al I', 'type': 'line', 'wavelengths': [[1.672351,1.675511],[1.3127006,1.3154345]]}, \
        'al1': {'label': r'Al I', 'type': 'line', 'wavelengths': [[1.672351,1.675511],[1.3127006,1.3154345]]}, \
        'fe': {'label': r'Fe I', 'type': 'line', 'wavelengths': [[1.5081407,1.5494570],[1.25604314,1.28832892],[1.14254467,1.15967616,1.16107501,1.16414462,1.16931726,1.18860965,1.18873357,1.19763233]]}, \
        'fei': {'label': r'Fe I', 'type': 'line', 'wavelengths': [[1.5081407,1.5494570],[1.25604314,1.28832892],[1.14254467,1.15967616,1.16107501,1.16414462,1.16931726,1.18860965,1.18873357,1.19763233]]}, \
        'fe1': {'label': r'Fe I', 'type': 'line', 'wavelengths': [[1.5081407,1.5494570],[1.25604314,1.28832892],[1.14254467,1.15967616,1.16107501,1.16414462,1.16931726,1.18860965,1.18873357,1.19763233]]}, \
        'k': {'label': r'K I', 'type': 'line', 'wavelengths': [[0.7699,0.7665],[1.169,1.177],[1.244,1.252]]}, \
        'ki': {'label': r'K I', 'type': 'line', 'wavelengths': [[0.7699,0.7665],[1.169,1.177],[1.244,1.252]]}, \
        'k1': {'label': r'K I', 'type': 'line', 'wavelengths': [[0.7699,0.7665],[1.169,1.177],[1.244,1.252]]}, \
		'h19f': {'label': r'H$^{19}$F', 'type': 'line', 'wavelengths': [[2.3358329,2.3358329]]}, \
		's1': {'label': r'S I', 'type': 'line', 'wavelengths': [[1.0455449,1.0456757,1.0459406]]}			}
    plt.figure(figsize=(30,10))
    bound=np.array([np.min(wave/wave.unit),np.max(wave/wave.unit),2*np.min(noisy_spec_filt/noisy_spec_filt.unit),2*np.max(noisy_spec_filt/noisy_spec_filt.unit)])
    nsamples=wave.shape[0]
    plt.plot(wave,spec_model_filt, label='Model')
    plt.plot(wave,noisy_spec_filt, label='Data',alpha=0.5)
# 	plt.title('Host: '+str(host_mag)+' mag (Vega), '+str(host_temp)+'K star')
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Counts (ph)')

    plt.title('Noisy spectrum reduced, filtered')

    plt.legend()
    f = interp1d(wave,noisy_spec_filt,bounds_error=False,fill_value=0.)
    wvmax = np.arange(bound[0],bound[1],0.001)
    flxmax = f(wvmax)

    yoff = 0.05*(bound[3]-bound[2])
#         fontsize = 10-numpy.min([(multilayout[0]*multilayout[1]-1),6])
    for ftr in features:
        ftr = ftr.lower()
# 		print(ftr)
        if ftr in feature_labels:
            for ii,waveRng in enumerate(feature_labels[ftr]['wavelengths']):
# 				if ((np.min(waveRng) and np.max(waveRng)) > bound[0] or (np.min(waveRng) and np.max(waveRng)) < bound[1]):
                if np.min(waveRng) < bound[0] and bound[0] < np.max(waveRng) < bound[1]:
                    waveRng[0]=bound[0]
                if bound[0] < np.min(waveRng) < bound[1] and np.max(waveRng) > bound[1]:
                    waveRng[-1]=bound[1]
                if (np.min(waveRng) >= bound[0] and np.max(waveRng) <= bound[1]):

                    x = (np.arange(0,nsamples+1.0)/nsamples)*(np.nanmax(waveRng)-np.nanmin(waveRng)+0.04)+np.nanmin(waveRng)-0.02
                    f = interp1d(wvmax,flxmax,bounds_error=False,fill_value=0.)
                    y = np.nanmax(f(x))+0.5*yoff
                    if feature_labels[ftr]['type'] == 'band':
                        plt.plot(waveRng,[y+yoff]*2,color='k',linestyle='-')
                        plt.plot([waveRng[0]]*2,[y,y+yoff],color='k',linestyle='-')
                        plt.text(np.mean(waveRng),y+1.5*yoff,feature_labels[ftr]['label'],horizontalalignment='center',fontsize=fontsize)
                    else:
                        for w in waveRng:
                            plt.plot([w]*2,[y,y+yoff],color='k',linestyle='-')
                        plt.text(np.mean(waveRng),y+1.5*yoff,feature_labels[ftr]['label'],horizontalalignment='center',fontsize=fontsize)
                        waveRng = [waveRng[0]-0.02,waveRng[1]+0.02]   # for overlap
						# update offset
                    foff = [y+3*yoff if (w >= waveRng[0] and w <= waveRng[1]) else 0 for w in wvmax]
                    flxmax = [np.max([xx,yy]) for xx, yy in zip(flxmax, foff)]
    bound[3] = np.nanmax([np.nanmax(flxmax)+1.*yoff,bound[3]])

    plt.xlim([bound[0],bound[1]])
    plt.ylim([bound[2],bound[3]])

#Compute PRV accuracy
def compute_prv_sigma(wave,delta_lb,snr,model,sky_trans, cal, kernel_sz):
	c=3e8 * u.m/u.s
	sigma_rv_inst = 0.1 * u.m / u.s
	sigma_rv = np.copy(wave)
	for i in range(wave.shape[0]):
		dvelocity = delta_lb[i] * c / wave[i]
# 		model_medfilt=medfilt(model[i],kernel_size = kernel_sz) * model[i].unit
		dummy = model[i].value
		model_medfilt=median_filter(dummy,kernel_sz,mode='nearest') * model[i].unit
# 		model_medfilt=medfilt2d(model[i].value.reshape(1, -1),(1,kernel_sz))[0] * model[i].unit
		model_filt_norm=(model[i])/model_medfilt
		didv=np.gradient(model_filt_norm)/dvelocity
		didv[np.isnan(didv)]=0
		norm = ((1-cal)*sky_trans[i])
		index=np.where(norm < 0.8)
		didv[index]=0.0
		print(np.sum((didv)**2), np.sum(snr[i]**2))
		sigma_rv[i] = np.sqrt( (np.sqrt(1.0)/np.sqrt(np.sum((didv*snr[i])**2)))**2 + sigma_rv_inst**2)
	return sigma_rv
