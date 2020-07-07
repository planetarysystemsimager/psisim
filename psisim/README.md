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

Pull this repository: 
```
> git clone https://github.com/planetarysystemsimager/psisim.git
```
Move into that directory and run the `setup.py` script to install. Use the develop keyword to stay up to date with new changes:
```
> python setup.py develop
```

In order to run the simulations, HISPEC error budgets and sampled atmospheric data must also be downloaded. At the moment, these are hosted online by [Jason](https://caltech.app.box.com/s/ce7hgt56usd1vfvzhn2kv8n7d9cojro6)

Optionally install [EXOSIMS](https://github.com/dsavransky/EXOSIMS) (if you want to simulate exoplanet populations) and [PICASO](https://github.com/natashabatalha/picaso) (if you want to generate exoplanet spectra on the fly) following the documentation for the respective packages. Installation can also be achieved through cloning the github repositories.

We currently support pickles and Castelli-Kurucz stellar models through the **pysynphot** package. In order to use them you need to install pysynphot and download those stellar spectra. 

### High-resolution simulations

If you want to simulate high resolution spectra (e.g. for HISPEC, MODHIS, etc.) you also need to install the **speclite** and **PyAstronomy** packages. We currently support high-resolution grids of Phoenix and Sonora models. Download these files into the same directory, e.g. **/scr3/dmawet/ETC/**, and make sure paths are as follows.

The full [Phoenix HiRes Library](ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/) can be downloaded through FTP. Specific Phoenix models can be installed [here](http://phoenix.astro.physik.uni-goettingen.de/?page_id=15) if desired.
Download Phoenix items into **/scr3/dmawet/ETC/** + *HIResFITS_lib/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/*. 

Your directory should read: ***/scr3/dmawet/ETC**/HIResFITS_lib/phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/host_model/host_filename*

[Sonora](https://zenodo.org/record/1309035#.XbtLtpNKhMA) files should be unzipped into **/scr3/dmawet/ETC**/Sonora. Your directory should read: ***/scr3/dmawet/ETC**/Sonora/file_name*. 
 
