# PSI Simulator Tool

Tool for the Planet Systems Imager to simulate the universe, generate planet spectra, evaulate instrument performance, and determine how well PSI can do science. 

Features of `psisim`:

  * Create various exoplanet populations using [EXOSIMS](https://github.com/dsavransky/EXOSIMS) (e.g., known RV planets, simulated planets following *Kepler* occurence rates),
  * Simulate reflected and thermal emission spectra of gas giants and terrestrial planets using [PICASO](https://github.com/natashabatalha/picaso),
  * Simulate planet fluxes based on evolutionary cooling models,
  * Estimate polarized fluxes of exoplanets,
  * Estimate performance of the AO system as a function of star brightness and ability to correct atmospheric turbulence,
  * Handle detector noise, thermal emission, sky transmission, and
  * Object oriented so easy to implement multiple sites and instruments.


## Installation
First, install [EXOSIMS](https://github.com/dsavransky/EXOSIMS) and [PICASO](https://github.com/natashabatalha/picaso) following the documentation for the respective packages.

Then pull this respository:
```
> git clone https://github.com/planetarysystemsimager/psisim.git
```
Move into that directory and run the `setup.py` script to install. Use the develop keyword to stay up to date with new changes:
```
> python setup.py develop
```
