import numpy as np
import astropy.units as u
import astropy.constants as constants
from astropy.table import QTable

class Universe():
    '''
    A universe class that includes

    Inherited from EXOSIMS? TBD

    Properties:
    planets    - A planet table that holds all the planet properties [Astropy table]. It has the following columns:

    '''

    def __init__(self):
        '''
        '''
        pass

    def save_planet_table(self,filename):
        '''
        Save the planet table to an astropy fits tables
        '''
        pass


    def load_planet_table(self,filename):
        '''
        Load a planet table that was saved by save_planet_table
        '''
        pass

class ExoSims_Universe(Universe):
    '''
    A child class of Universe that is adapted specfically to work with the outputs of EXOSIMS
    '''

    def __init__(self,exosims_config_filename):
        super(ExoSims_Universe, self).__init__()
        
        self.filename = exosims_config_filename
        self.planets = None

    def simulate_EXOSIMS_Universe(self):
        '''
        A function that runs EXOSIMS and takes the output to populate the planets table
        '''

        import EXOSIMS.MissionSim

        sim = EXOSIMS.MissionSim.MissionSim(self.filename, explainFiltering=True, fillPhotometry=True, nokoMap=True)

        flux_ratios = 10**(sim.SimulatedUniverse.dMag/-2.5)  # grab for now from EXOSIMS
        angseps = sim.SimulatedUniverse.WA.value * 1000 *u.mas # mas
        projaus = sim.SimulatedUniverse.d.value * u.AU # au
        phase = np.arccos(sim.SimulatedUniverse.r[:,2]/sim.SimulatedUniverse.d)# planet phase  [0, pi]
        smas = sim.SimulatedUniverse.a.value*u.AU # au
        eccs = sim.SimulatedUniverse.e # eccentricity
        incs = sim.SimulatedUniverse.I.value*u.deg # degrees
        masses = sim.SimulatedUniverse.Mp.value # earth masses
        radii = sim.SimulatedUniverse.Rp.value # earth radii
        grav = constants.G * (masses * u.earthMass)/(radii * u.earthRad)**2
        logg = np.log10(grav.to(u.cm/u.s**2).value) # logg cgs

        # stellar properties
        ras = [] # deg
        decs = [] # deg
        distances = [] # pc
        for index in sim.SimulatedUniverse.plan2star:
            coord = sim.TargetList.coords[index]
            ras.append(coord.ra.value)
            decs.append(coord.dec.value)
            distances.append(coord.distance.value)
        ras = np.array(ras)
        decs = np.array(decs)
        distances = np.array(distances)
        star_names =  np.array([sim.TargetList.Name[i] for i in sim.SimulatedUniverse.plan2star])
        spts = np.array([sim.TargetList.Spec[i] for i in sim.SimulatedUniverse.plan2star])
        sim.TargetList.stellar_mass() # generate masses if haven't
        host_mass = np.array([sim.TargetList.MsTrue[i].value for i in sim.SimulatedUniverse.plan2star])
        host_teff = sim.TargetList.stellarTeff(sim.SimulatedUniverse.plan2star).value
        # stellar photometry
        host_Bmags = np.array([sim.TargetList.Bmag[i] for i in sim.SimulatedUniverse.plan2star])
        host_Vmags = np.array([sim.TargetList.Vmag[i] for i in sim.SimulatedUniverse.plan2star])
        host_Rmags = np.array([sim.TargetList.Rmag[i] for i in sim.SimulatedUniverse.plan2star])
        host_Imags = np.array([sim.TargetList.Imag[i] for i in sim.SimulatedUniverse.plan2star])
        host_Jmags = np.array([sim.TargetList.Jmag[i] for i in sim.SimulatedUniverse.plan2star])
        host_Hmags = np.array([sim.TargetList.Hmag[i] for i in sim.SimulatedUniverse.plan2star])
        host_Kmags = np.array([sim.TargetList.Kmag[i] for i in sim.SimulatedUniverse.plan2star])
        
        # guess the radius and gravity from Vmag and Teff. This is of questionable reliability
        host_MVs = host_Vmags - 5 * np.log10(distances/10) # absolute V mag
        host_lums = 10**(-(host_MVs-4.83)/2.5) # L/Lsun
        host_radii = (5800/host_teff)**2 * np.sqrt(host_lums) # Rsun
        host_gravs = constants.G * (host_mass * u.solMass)/(host_radii * u.solRad)**2
        host_logg = np.log10(host_gravs.to(u.cm/u.s**2).value) # logg cgs

        all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phase, smas, eccs, incs, masses, radii, logg, spts, host_mass, host_teff, host_radii, host_logg, host_Bmags, host_Vmags, host_Rmags, host_Imags, host_Jmags, host_Hmags, host_Kmags]
        labels = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phase", "SMA", "Ecc", "Inc", "PlanetMass", "PlanetRadius", "PlanetLogg", "StarSpT", "StarMass", "StarTeff", "StarRad", "StarLogg", "StarBMag", "StarVmag", "StarRmag", "StarImag", "StarJmag", "StarHmag", "StarKmag"]

        planets_table = QTable(all_data, names=labels)

        self.planets = planets_table
        
class ExoArchive_Universe(Universe):
    '''
    A child class of Universe that is adapted to create a universe from known NASA Exoplanet Archive Data
    
    Uses the astroquery package to read in known exoplanets
    '''

    def __init__(self,table_filename):
        super(ExoArchive_Universe, self).__init__()
       
        self.filename = table_filename
        self.planets = None

    def Load_ExoArchive_Universe(self):
        '''
        A function that reads the Exoplanet Archive data and takes the output to populate the planets table
        
        If the filename provided in the constructor is new, new data will be pulled from the archive
        If the filename already exists, we'll try to load that file as an astroquery QTable
        '''

        #-- Load/Pull data depending on provided filename
        import os
        
        if os.path.isfile(self.filename):
            print("%s already exists:\n    we'll attempt to read this file as an astroquery table"%self.filename)
            
            NArx_table = QTable.read(self.filename, format='ascii.ecsv')
            
            # When reading table from a file, ra and dec are hidden in sky_coord; extract
            NArx_table['ra']  = NArx_table['sky_coord'].ra
            NArx_table['dec'] = NArx_table['sky_coord'].dec
            
        else:
            print("%s does not exist:\n    we'll pull new data from the archive and save it to this filename"%self.filename)
            
            # Import astroquery package used to read table using Exoplanet Archive API
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive as NArx
            # TODO: switch to TAP service; Archive saves the API will be deprecated soon:
                # https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
                # We could consider using PyVO: https://pyvo.readthedocs.io/en/latest/
            
            # Define the columns to read. NOTE: add columns here if needed
            col2pull = "pl_angsep,pl_orbsmax,pl_orbeccen,pl_orbincl,pl_bmasse,pl_rade,ra,dec,st_dist,pl_hostname,st_spstr,st_mass,st_teff,st_bj,st_vj,st_rc,st_ic,st_j,st_h,st_k,st_rad,st_logg"
            
            # Pull data. Note: pulling "confirmed planets" table. 
                #  Could use table="compositepars" to pull "Composite Planet Data" table 
                # that'll have more entries but this table could be self INconsistent.
            NArx_table = NArx.query_criteria(table="exoplanets",select=col2pull)
            
            # Save raw table for future use (use ascii.ecsv so Quantity objects are saved)
            NArx_table.write(self.filename,format='ascii.ecsv')

            
        #-- Compute/Populate pertinent entries into planet table
        
        # TODO: the following are not easily available in table; need approx for these
            # For now, set them to nan
        flux_ratios= np.zeros(len(NArx_table)); flux_ratios[:] = np.nan
        projaus    = np.zeros(len(NArx_table)); projaus[:] = np.nan
        phase      = np.zeros(len(NArx_table)); phase[:] = np.nan
        
        # Note: we use np.array() call in variable def's below to avoid pass-by-ref

        angseps    = NArx_table['pl_angsep'].copy()     # mas
        smas       = NArx_table['pl_orbsmax'].copy()    # au
        eccs       = np.array(NArx_table['pl_orbeccen'])# eccentricity
        incs       = NArx_table['pl_orbincl'].copy()    # deg
        masses     = np.array(NArx_table['pl_bmasse'])  # earth masses (best estimate)
        radii      = np.array(NArx_table['pl_rade'])    # earth radii
        grav = constants.G * (masses * u.earthMass)/(radii * u.earthRad)**2
        logg = np.log10(grav.to(u.cm/u.s**2).value)     # logg cgs

        # stellar properties
        ras        = np.array(NArx_table['ra'])         # deg
        decs       = np.array(NArx_table['dec'])        # deg 
        distances  = np.array(NArx_table['st_dist'])    # pc
        star_names = np.array(NArx_table['pl_hostname'])
        spts       = NArx_table['st_spstr'].copy() # NOTE: string format might not be 
        
        host_mass  = np.array(NArx_table['st_mass'])    # solar masses
        host_teff  = np.array(NArx_table['st_teff'])    # K
        # stellar photometry
        host_Bmags = np.array(NArx_table['st_bj'])      # Johnson
        host_Vmags = np.array(NArx_table['st_vj'])      # Johnson
        host_Rmags = np.array(NArx_table['st_rc'])      # Cousins
        host_Imags = np.array(NArx_table['st_ic'])      # Cousins
        host_Jmags = np.array(NArx_table['st_j'])       # 2MASS
        host_Hmags = np.array(NArx_table['st_h'])       # 2MASS
        host_Kmags = np.array(NArx_table['st_k'])       # 2MASS
        
        host_radii = np.array(NArx_table['st_rad'])     # Rsun
        host_logg  = np.array(NArx_table['st_logg'])    # logg cgs
        
        # Guess the radius and gravity from Vmag and Teff only if the
             # table-provided value is 0. This guess will be questionably reliable
        host_MVs = host_Vmags - 5 * np.log10(distances/10) # absolute V mag
        host_lums = 10**(-(host_MVs-4.83)/2.5) # L/Lsun
        tmp_radii = (5800/host_teff)**2 * np.sqrt(host_lums) # Rsun
        guessinds  = np.where(host_radii == 0) # Indices of radii to populate with guess
        host_radii[guessinds] = tmp_radii[guessinds]
        host_gravs = constants.G * (host_mass * u.solMass)/(host_radii * u.solRad)**2
        tmp_logg = np.log10(host_gravs.to(u.cm/u.s**2).value) # logg cgs
        guessinds  = np.where(host_logg == 0) # Indices of logg to populate with guess
        host_logg[guessinds] = tmp_logg[guessinds]

        all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phase, smas, eccs, incs, masses, radii, logg, spts, host_mass, host_teff, host_radii, host_logg, host_Bmags, host_Vmags, host_Rmags, host_Imags, host_Jmags, host_Hmags, host_Kmags]
        labels = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phase", "SMA", "Ecc", "Inc", "PlanetMass", "PlanetRadius", "PlanetLogg", "StarSpT", "StarMass", "StarTeff", "StarRad", "StarLogg", "StarBMag", "StarVmag", "StarRmag", "StarImag", "StarJmag", "StarHmag", "StarKmag"]

        planets_table = QTable(all_data, names=labels)

        self.planets = planets_table
