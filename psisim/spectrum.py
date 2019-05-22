import scipy.ndimage as ndi
import astropy.units as u
import picaso
from picaso import justdoit as jdi

def generate_picaso_inputs(planet_table_entry, planet_type, clouds=True):
	'''
	A function that returns the required inputs for picaso, 
	given a row from a universe planet table

	Inputs:
	planet_table_entry - a single row, corresponding to a single planet
						 from a universe planet table [astropy table (or maybe astropy row)]
	planet_type - either "Terrestrial", "Ice" or "Giant" [string]
	clouds - cloud parameters. For now, only accept True/False to turn clouds on and off
	
	Outputs:
	params - picaso.justdoit.inputs class
	'''
	params = jdi.inputs()

	#phase angle
	params.phase_angle(planet_table_entry['Phase']) #radians

	#define gravity
	# TODO: calculate from mass/radius
	params.gravity(gravity=25, gravity_unit=u.Unit('m/(s**2)')) #any astropy units available

	#define star
	# TODO: make this not a random hard coded number
	# Need to convert SpT to temperature and gravity (assume Main sequence)
	params.star(opacity, 5000,0,4.0) #opacity db, pysynphot database, temp, metallicity, logg

	# define atmosphere PT profile and mixing ratios. 
	# Hard coded as Jupiters right now. 
	params.atmosphere(filename=jdi.jupiter_pt(), delim_whitespace=True)

	if clouds:
		# use Jupiter cloud deck for now. 
		params.clouds( filename= jdi.jupiter_cld(), delim_whitespace=True)

	return params

def simulate_spectrum(planet_table_entry, wvs, R, atmospheric_parameters, package="picaso"):
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
	opacity = jdi.opannection()
	model_wnos, model_alb = atmospheric_parameters.spectrum(opacity)
	model_wvs = 1./model_wnos * 1e4 # microns
	
	model_dwvs = np.abs(model_wvs - np.roll(model_wvs, 1))
	model_dwvs[0] = model_dwvs[1]
	model_R = model_wvs/model_dwvs

	highres_fp =  model_alb * (planet_table_entry['PlanetRadius']*u.earthRad.to(u.au)/planet_table_entry['SMA'])**2 # flux ratio relative to host star

	lowres_fp = downsample_spectrum(highres_fp, np.mean(model_R), R)

	fp = np.interp(wvs, model_wvs, lowres_fp)

	return fp

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
	fwhm = R_in/R_out
	sigma = fwhm/(2*np.sqrt(2*np.log(2)))

	new_spectrum = ndi.gaussian_filter(spectrum, sigma)

	return new_spectrum
