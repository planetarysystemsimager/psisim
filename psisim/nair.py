# Copied from https://github.com/semaphoreP/nair, commit 3b701b
import numpy as np

def nMathar(wv, P, T, H=10):
    """
    Calculate the index of refraction as given by Mathar (2008): http://arxiv.org/pdf/physics/0610256v2.pdf
    ***Only valid for between 1.3 and 2.5 microns!
    Inputs:
        wv: wavelength in microns
        P:  Pressure in Pa
        T:  Temperature in Kelvin
        H:  relative humiditiy in % (i.e. between 0 and 100)
    Return:
        n:  index of refratoin
    """
    n = np.ones(np.size(wv)) # output. default to 1.
    wvnum = 1.e4/wv #cm^-1     # convert to wavenumbers

    # if it was passed in as a float, we need to convert it into a array for code reuse
    if not isinstance(wvnum, np.ndarray):
        wvnum = np.array([wvnum])

    # polynomial expansion in wavenumber
    # calcualate index of refraction by splitting it up by region
    # for region 1. < 2.65 microns, technically only valid for 1.3 to 2.5 microns
    region1 = np.where(wv < 2.8)
    if np.size(region1) > 0 :    
        wvnum0 = 1.e4/2.25 #cm^-1 #series expand around this wavenumber

        #do a 5th order expansion
        powers = np.arange(0,6)
        #calculate expansion coefficients
        coeffs = get_coeff_mathar(powers, P, T, H, 1)

        #sum of the power series expansion
        for coeff,power in zip(coeffs, powers):
            n[region1] += coeff * ((wvnum[region1] - wvnum0)**power)
    # next for everything greater than 2.65 microns. technically valid for 2.8 - 4.2 microns
    region2 = np.where(wv >= 2.8)
    if np.size(region2) > 0:
        wvnum0 = 1.e4/3.4 # cm^-1 #series expand around this wavenumber

        # do a 5th order expansion
        powers = np.arange(0,6)
        # calculate expansion coefficients
        coeffs = get_coeff_mathar(powers, P, T, H, 2)

        #sum of the power series expansion
        for coeff, power in zip(coeffs, powers):
            n[region2] += coeff * ((wvnum[region2] - wvnum0)**power)

    # return a int/float if that is what is passed in
    if isinstance(wv, (int, float)):
        n = n[0]

    return n

def get_coeff_mathar(i, P, T, H, wvrange=1):
    """
    Calculate the coefficients for the polynomial series expansion of index of refraction (Mathar (2008))
    ***Only valid for between 1.3 and 2.5 microns! and 2.8 through 4.2 microns!
    Inputs:
        i:  degree of expansion in wavenumber
        P:  Pressure in Pa
        T:  Temperature in Kelvin
        H:  relative humiditiy in % (i.e. between 0 and 100)
        wvrange (int): 1 = (1.3-2.5 um), 2 = (2.8 - 4.2 um)
    Return:
        coeff:  Coefficient [cm^-i]
    """

    #name all the constants in the model
    #series expansion in evironment parameters
    T0 = 273.15 + 17.5 #Kelvin
    P0 = 75000 #Pa
    H0 = 10 #%

    #delta terms for the expansion
    dT = 1./T - 1./T0
    dP = P - P0
    dH = H - H0

    #loads and loads of coefficients, see equation 7 in Mathar (2008)
    #use the power (i.e. i=[0..6]) to index the proper coefficient for that order
    if wvrange == 1:
        cref= np.array([ 0.200192e-3, 0.113474e-9, -0.424595e-14, 0.100957e-16,-0.293315e-20, 0.307228e-24]) # cm^i
        cT  = np.array([ 0.588625e-1,-0.385766e-7,  0.888019e-10,-0.567650e-13, 0.166615e-16,-0.174845e-20])  # K cm^i
        cTT = np.array([-3.01513,     0.406167e-3, -0.514544e-6,  0.343161e-9, -0.101189e-12, 0.106749e-16]) # K^2 cm^i
        cH  = np.array([-0.103945e-7, 0.136858e-11,-0.171039e-14, 0.112908e-17,-0.329925e-21, 0.344747e-25]) # cm^i / %
        cHH = np.array([ 0.573256e-12,0.186367e-16,-0.228150e-19, 0.150947e-22,-0.441214e-26, 0.461209e-30]) # cm^i / %^2
        cP  = np.array([ 0.267085e-8, 0.135941e-14, 0.135295e-18, 0.818218e-23,-0.222957e-26, 0.249964e-30]) # cm^i / Pa
        cPP = np.array([ 0.609186e-17,0.519024e-23,-0.419477e-27, 0.434120e-30,-0.122445e-33, 0.134816e-37]) # cm^i / Pa^2
        cTH = np.array([ 0.497859e-4,-0.661752e-8,  0.832034e-11,-0.551793e-14, 0.161899e-17,-0.169901e-21]) # cm^i K / %
        cTP = np.array([ 0.779176e-6, 0.396499e-12, 0.395114e-16, 0.233587e-20,-0.636441e-24, 0.716868e-28]) # cm^i K / Pa
        cHP = np.array([-0.206567e-15,0.106141e-20,-0.149982e-23, 0.984046e-27,-0.288266e-30, 0.299105e-34]) # cm^i / Pa %
    elif wvrange == 2:
        cref= np.array([ 0.200049e-3, 0.145221e-9,  0.250951e-12, -0.745834e-15,-0.161432e-17, 0.352780e-20]) # cm^i
        cT  = np.array([ 0.588431e-1,-0.825182e-7,  0.137982e-9,  0.352420e-13,-0.730651e-15,-0.167911e-18])  # K cm^i
        cTT = np.array([-3.13579,     0.694124e-3, -0.500604e-6, -0.116668e-8,  0.209644e-11, 0.591037e-14]) # K^2 cm^i
        cH  = np.array([-0.108142e-7, 0.230102e-11,-0.154652e-14,-0.323014e-17, 0.630616e-20, 0.173880e-22]) # cm^i / %
        cHH = np.array([ 0.586812e-12,0.312198e-16,-0.197792e-19,-0.461945e-22, 0.788398e-25, 0.245580e-27]) # cm^i / %^2
        cP  = np.array([ 0.266900e-8, 0.168162e-14, 0.353075e-17,-0.963455e-20,-0.223079e-22, 0.453166e-25]) # cm^i / Pa
        cPP = np.array([ 0.608860e-17,0.461560e-22, 0.184282e-24,-0.524471e-27,-0.121299e-29, 0.246512e-32]) # cm^i / Pa^2
        cTH = np.array([ 0.517962e-4,-0.112149e-7,  0.776507e-11, 0.172569e-13,-0.320582e-16,-0.899435e-19]) # cm^i K / %
        cTP = np.array([ 0.778638e-6, 0.446396e-12, 0.784600e-15,-0.195151e-17,-0.542083e-20, 0.103530e-22]) # cm^i K / Pa
        cHP = np.array([-0.217243e-15,0.104747e-20,-0.523689e-23, 0.817386e-26, 0.309913e-28,-0.363491e-31]) # cm^i / Pa %

    # use numpy arrays to calculate all the coefficients at the same time
    coeff = cref[i] + cT[i]*dT + cTT[i]*(dT**2) + cH[i]*dH + cHH[i]*(dH**2) + cP[i]*dP + cPP[i]*(dP**2) + cTH[i]*dT*dH + cTP[i]*dT*dP + cHP[i]*dH*dP

    return coeff


def nRoe(wv, P, T, H=10):
    """
    Compute n for air from the formula in Henry Roe's PASP paper: http://arxiv.org/pdf/astro-ph/0201273v1.pdf
    which in turn is referenced from Allen's Astrophysical Quantities.
    Inputs:
        wv: wavelength in microns
        P:  pressure in Pascal
        T:  temperature in Kelvin
        H:  relative humidity in % (0-100)
    Return:
        n:  index of refraction of air
    """

    #convert pressure from Pa to mbar
    P /= 100.

    #some constants in the function for n
    a1 = 64.328
    a2 = 29498.1
    a3 = 146.0
    a4 = 255.4
    a5 = 41.0

    Ts = 288.15   # K
    Ps = 1013.35 # mb

    #calculate n-1 for dry air
    K1 = 1e-6*(P/Ps * Ts/T)
    n1 = K1*(a1 + a2/(a3-wv**(-2)) + a4/(a5-wv**(-2))      )

    # water vapor correction
    # first compute partial pressure of water
    p_h20 = H/100. * saturation_pressure(T) # Pa
    fh20 = p_h20 / (1013.25 * 100)
    K2 = -43.49e-6 * fh20
    a6 = -7.956e-3
    nh2o = K2*(1 + a6*wv**(-2))

    return n1 + nh2o + 1

    
def saturation_pressure(temp):
    """
    Computes the saturation vapor pressure of water (from Voronin & Zheltikov 2017, eq 7)
    Args:
        temp (float): temperature in Kelvin
    
    Return:
        ps: saturation pressure in Pa
    """
    Tc = 647.096 # Kelvin, critical point temperature of water
    pc = 22.064e6 # Pa

    tau = Tc/temp
    theta = 1 - temp/Tc

    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502

    arg = a1*theta + a2*theta**1.5 + a3*theta**3 + a4*theta**3.5 + a5*theta**4 + a6*theta**7.5

    ps = pc * np.exp(tau * arg)

    return ps


def compute_dar(n, n_ref, zenith):
    """
    Compute differential atmospheric refraction for given index of air

    Args:
        n (np.array of float): index of air at wavelengths of interest
        n_ref (float): reference index of air to compute DAR relative to
        zenith (float): zenith angle in radians

    Returns:
        dz_mas (float): DAR in milliarcseconds.
    """
    dz = (n**2 - 1)/(2 * n**2) * np.tan(zenith)
    dz -= (n_ref**2 - 1)/(2 * n_ref**2) * np.tan(zenith)
    dz_mas = dz * 1000 * 206265

    return dz_mas