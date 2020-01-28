# PSI Simulator Tool

Tool for the Planet Systems Imager to simulate the universe, generate planet spectra, evaulate instrument performance, and determine how well PSI can do science.

Features of `psisim`:

  * Create various exoplanet populations using [EXOSIMS](https://github.com/dsavransky/EXOSIMS) (e.g., known RV planets, simulated planets following *Kepler* occurence rates)
  * Simulate reflected and thermal emission spectra of gas giants and terrestrial planets using [PICASO](https://github.com/natashabatalha/picaso)
  * Simulate planet fluxes based on evolutionary cooling models
  * Estimate polarized fluxes of exoplanets
  * Performance estimates of the AO system as a function of star brightness and ability to correct atmospheric turbulence
  * Handle detector noise, thermal emission, sky transmission
  * Object orietented so easy to implement multiple sites and instrumnets

Many of these features are works in progress so please contact Max or Jason if you're having troubles. 

## Installation

Pull this respository:
```
> git clone https://github.com/planetarysystemsimager/psisim.git
```
Move into that directory and run the `setup.py` script to install. Use the develop keyword to stay up to date with new changes:
```
> python setup.py develop
```

Optionally install [EXOSIMS](https://github.com/dsavransky/EXOSIMS) (if you want to simulate exoplanet populations) and [PICASO](https://github.com/natashabatalha/picaso) (if you want to generate exoplanet spectra on the fly) following the documentation for the respective packages.

We currently support pickles and Castelli-Kurucz stellar models through the **pysynphot** package. In order to use them you need to install pysynphot and download those stellar spectra. 

### High-resolution simulations
If you want to simulate high resolution spectra (e.g. for HISPEC, MODHIS, etc.) you also need to install the **speclite** and **PyAstronomy** packages.

We currently support high-resolution grids of Phoenix and Sonora models. For those you need to download the grids yourself and then pass their path as **user_params** to the **get_stellar_spectrum** function. Hopefully in the future we will add some more detailed instructions on how to set this up. 
