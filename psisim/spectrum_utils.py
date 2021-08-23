import numpy as np

def spec_to_constant_R(wave, flux, output_R=None):
    """
    Function to interpolate a spectrum, presumably on a constant wavlength or constant frequency spacing grid, to a super-sampled constant resolution spacing wavelength grid.
    Bringing a spectrum to a constant resolution spacing is necessary for convolving the spectrum (broaden_spec_to_variable_R function)


    Args:
        wave (numpy.ndarray): Input spectrum wavelength array of length N.
        flux (numpy.ndarray): Input spectrum flux array of length N.
        output_R (int/float, optional): Resolution of the outputed spectrum. The default is None in which case the max resolution of the input spectrum will be calculated.

    Returns:
        constant_R_wave (numpy.ndarray): Constant resolution wavelength array.
        constant_R_flux (numpy.ndarray): Constant resolution flux array.

    Written by Stefan Pelletier (2021)
    """
    
    # if no super-sample R is given, calculate
    if output_R is None:
        # calculate wavelength grid resolution at each point
        Resolution_per_wave = wave[:-1]/np.diff(wave)
        # find the maximum resolution
        max_R = np.max(Resolution_per_wave[np.isfinite(Resolution_per_wave)])
        # clip if max is above 1 million resolution
        output_R = max_R.clip(max=1e6)
    #re-generate a wavelength grid that is constant in resolution (super-sampled at output_R)
    wave_min = np.min(wave)
    wave_max = np.max(wave)
    length = output_R * np.log(wave_max / wave_min) + 1
    mid = (-output_R)*np.log(wave_min)  
    constant_R_wave = np.exp((np.arange(length) - mid) / output_R)
    # interpolate flux onto constant_R wavelength array
    constant_R_flux = np.interp(constant_R_wave, wave, flux)
    
    return constant_R_wave, constant_R_flux




def broaden_spec_to_variable_R(wave, flux, end_resolution=70000, start_resolution=None, max_sigma_width=4, Print=False):
    """
    Function that broadens a high-resolution spectrum to a lower spectral resolution.
    Spectrum must be on a constant resolution (lambda / delta lambda) grid 
    Desired end resolution can be constant (fixed), or defined at every point (variable)    

    Args:
    wave (numpy.ndarray): Input spectrum wavelength array of length N.
    flux (numpy.ndarray): Input spectrum flux array of length N.
    end_resolution (int/float or numpy.ndarray, optional): What resolution you want to downgrade your spectrum to.  int or float for a fixed resolution.  Array of length N for a variable resolution defined at each point.
    start_resolution (int/float, optional): The initial (constant) resolution of your model (default is 250,000 for scarlet). The default is None in which case it will be calculated.
    max_sigma_width (int/float, optional): Up to how many sigmas is the Gaussian kernel computed (not precise if less than 3, not much changes higher than 5). The default is 4.
    Print (bool, optional): True or False, whether to print info. The default is False.


    Returns: 
        broad_flux (numpy.ndarray): The spectrum, on the same wavelength grid, convolved down to the desired resolution.

    Written by Stefan Pelletier (2021)

    """
    
    # if the initial resolution of the model is not given, calculate it from the wavelength array of the model
    if start_resolution is None:
        # if start resolution is not defined then wave must be
        start_resolution = np.mean(wave[:-1])/np.mean(np.diff(wave[:-1]))
        if Print == True:
            print('Calculated Model Resolution = %s ' % np.round(start_resolution) )
            print('Downgrading to R = %s ' % np.round(end_resolution) )
    # determine FWHM of the Gaussian that we want to use to convolve our model with to downgrade its resolution
    FWHM = start_resolution/end_resolution
    if np.min(FWHM) < 2.5:
        print('Warning: Initial model resolution is less than 2.5 times higher than the desired end resolution at some wavelengths!')
    # convert that FWHM into sigma (standard conversion equation - wikipedia)
    sigma = FWHM / 2.35482 # / (2 * np.sqrt(2 * np.log(2)) )
    # determine width of Gaussian to use for convolution: go out sigma_width*sigma elements in each direction.  Pick max value if variable resolution
    max_n_elements = int(np.round(np.max(sigma)*max_sigma_width)*2+1) 
    extent = int((max_n_elements-1)/2)
    # generate convolution matrix: Gaussian kernel for each wavelength
    kernel = np.exp(-0.5*(np.outer(np.arange(max_n_elements) - extent, 1/sigma))**2) / np.sqrt(2 * np.pi * sigma**2)
    
    # if constant resolution, kernel is constant so do 1D convolution (faster)
    if np.size(sigma) == 1:
        broad_flux = np.convolve(flux, kernel[:,0], mode='same')
    # if variable resolution, kernel is defined per wavelength so do 2D convolution (slower)
    else:
        mod_size = int(wave.size)
        # matrix convolution
        shifted_spec_matrix = np.zeros([max_n_elements, mod_size])
        for ind in np.arange(extent-1)+1:
            shifted_spec_matrix[extent-ind,ind:]  = flux[:-ind]
            shifted_spec_matrix[extent+ind,:-ind] = flux[ind:] 
        shifted_spec_matrix[extent,:] = flux[:]
        
        broad_flux = np.sum(shifted_spec_matrix*kernel,axis=0)

    return broad_flux