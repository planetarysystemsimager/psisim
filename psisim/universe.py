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
        
        self.MJUP2EARTH = 317.82838    # conversion from Jupiter to Earth masses
        self.MSOL2EARTH = 332946.05    # conversion from Solar to Earth masses
        self.RJUP2EARTH = 11.209       # conversion from Jupiter to Earth radii
        
        #-- Chen & Kipping 2016 constants
            # ref.: https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
        # Exponent terms from paper (Table 1)
        self._CKe0  = 0.279    # Terran 
        self._CKe1  = 0.589    # Neptunian 
        self._CKe2  =-0.044    # Jovian 
        self._CKe3  = 0.881    # Stellar 
        # Object-type transition points from paper (Table 1) - Earth-mass units
        self._CKMc0 = 2.04                   # terran-netpunian transition
        self._CKMc1 = 0.414*self.MJUP2EARTH  # neptunian-jovian transition 
        self._CKMc2 = 0.080*self.MSOL2EARTH  # jovian-stellar transition
        # Coefficient terms
        self._CKC0  = 1.008    # Terran - from paper (Table 1)
        self._CKC1  = 0.808    # Neptunian - computed from intercept with terran domain
        self._CKC2  = 17.74    # Jovian - computed from intercept neptunian domain
        self._CKC3  = 0.00143  # Stellar - computed from intercept with jovian domain
        
        #-- Thorngren 2019 Constants
            # ref.: https://doi.org/10.3847/2515-5172/ab4353
        # Coefficient terms from paper
        self._ThC0  = 0.96
        self._ThC1  = 0.21
        self._ThC2  =-0.20
        # Define Constraints
        self._ThMlow = 15            # [M_earth] Lower bound of applicability
        self._ThMhi  = 12*self.MJUP2EARTH # [M_earth] Upper bound of applicability
        self._ThThi  = 1000          # [K] Temperature bound of applicability
        
    def Load_ExoArchive_Universe(self, est_pl_radius=False, est_pl_mass=False):
        '''
        A function that reads the Exoplanet Archive data and takes the output to populate the planets table
        
        If the filename provided in the constructor is new, new data will be pulled from the archive
        If the filename already exists, we'll try to load that file as an astroquery QTable
        
        Kwargs:
        est_pl_radius    - Boolean denoting if planet radius should be approximated using mass-radius relations [default False]
        est_pl_mass      - Boolean denoting if planet mass should be approximated using mass-radius relations [default False]
        
        *** Note: the resulting planet table will have 0's where data is missing/unknown. 
            Eg. when a planet is missing a radius measurement, the 'PlanetRadius' column for that planet will == 0.        
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
            col2pull = "pl_angsep,pl_orbsmax,pl_orbeccen,pl_orbincl,pl_bmasse,pl_rade,pl_eqt,ra,dec,st_dist,pl_hostname,st_spstr,st_mass,st_teff,st_bj,st_vj,st_rc,st_ic,st_j,st_h,st_k,st_rad,st_logg"
            
            # Pull data. Note: pulling "confirmed planets" table. 
                #  Could use table="compositepars" to pull "Composite Planet Data" table 
                # that'll have more entries but this table could be self INconsistent.
            NArx_table = NArx.query_criteria(table="exoplanets",select=col2pull)
            
            # Save raw table for future use (use ascii.ecsv so Quantity objects are saved)
            NArx_table.write(self.filename,format='ascii.ecsv')

            
        #-- Compute/Populate pertinent entries into planet table
        
        # TODO: the following are not easily available in table; need approx for these
            # For now, set them to nan
        flux_ratios= np.zeros(len(NArx_table))#; flux_ratios[:] = np.nan
        projaus    = np.zeros(len(NArx_table))#; projaus[:] = np.nan
        phase      = np.zeros(len(NArx_table))#; phase[:] = np.nan
        
        # Note: we use np.array() call in variable def's below to avoid pass-by-ref

        angseps    = NArx_table['pl_angsep'].copy()     # mas
        smas       = NArx_table['pl_orbsmax'].copy()    # au
        eccs       = np.array(NArx_table['pl_orbeccen'])# eccentricity
        incs       = NArx_table['pl_orbincl'].copy()    # deg
        eqtemps    = np.array(NArx_table['pl_eqt'])     # K
        masses     = np.array(NArx_table['pl_bmasse'])  # earth masses (best estimate)
        radii      = np.array(NArx_table['pl_rade'])    # earth radii        
        # Compute missing radii using mass-radius relations if requested
        if est_pl_radius:
            radii = self.approximate_radii(masses,radii,eqtemps)
        # Compute missing masses using mass-radius relations if requested
        if est_pl_mass:
            masses = self.approximate_masses(masses,radii,eqtemps)

        # Compute logg (being wary of 0 values in mass or radii)
        mask_0 = (masses != 0) & (radii != 0)
        grav = constants.G * (masses[mask_0] * u.earthMass)/(radii[mask_0] * u.earthRad)**2
        logg = np.zeros(len(NArx_table))
        logg[mask_0] = np.log10(grav.to(u.cm/u.s**2).value)     # logg cgs

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
        
        # Guess the radius and gravity from Vmag and Teff only if the table-provided value 
            # for rad and grav  are 0. This guess will be questionably reliable
        # Create mask to be careful about 0's elsewhere too
        mask_0 = (host_radii == 0) & (host_Vmags != 0) & (distances != 0) & (host_teff != 0)
        host_MVs = host_Vmags[mask_0] - 5 * np.log10(distances[mask_0]/10) # absolute V mag
        host_lums = 10**(-(host_MVs-4.83)/2.5) # L/Lsun
        host_radii[mask_0] = (5800/host_teff[mask_0])**2 * np.sqrt(host_lums) # Rsun
        mask_0 = (host_logg == 0) & (host_mass != 0) & (host_radii != 0) 
        host_gravs = constants.G * (host_mass[mask_0] * u.solMass)/(host_radii[mask_0] * u.solRad)**2
        host_logg[mask_0] = np.log10(host_gravs.to(u.cm/u.s**2).value) # logg cgs

        all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phase, smas, eccs, incs, masses, radii, logg, eqtemps, spts, host_mass, host_teff, host_radii, host_logg, host_Bmags, host_Vmags, host_Rmags, host_Imags, host_Jmags, host_Hmags, host_Kmags]
        labels = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phase", "SMA", "Ecc", "Inc", "PlanetMass", "PlanetRadius", "PlanetLogg", "PlanetTeq", "StarSpT", "StarMass", "StarTeff", "StarRad", "StarLogg", "StarBMag", "StarVmag", "StarRmag", "StarImag", "StarJmag", "StarHmag", "StarKmag"]

        planets_table = QTable(all_data, names=labels)

        self.planets = planets_table
    
    def approximate_radii(self,masses,radii,eqtemps):
        '''
        Approximate planet radii given the planet masses
        
        Arguments:
        masses    - ndarray of planet masses
        radii     - ndarray of planet radii. 0-values will be replaced with approximation.
        eqtemps   - ndarray of planet equilibrium temperatures (needed for Thorngren constraints)
        
        Returns:
        radii     - ndarray of planet radii after approximation.
        
        Methodology:
        - Uses Thorngren 2019 for targets with 15M_E < M < 12M_J and T_eq < 1000 K.
            ref.: https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
        - Uses Chen and Kipping 2016 for all other targets.
            ref.: https://doi.org/10.3847/2515-5172/ab4353
        * Only operates on 0-valued elementes in radii vector (ie. prioritizes Archive-provided radii).
        '''
        
        ##-- Find indices for missing radii so we don't replace Archive-provided values
        rad_mask = (radii == 0.0)        
        
        
        ##-- Compute radii assuming Chen&Kipping 2016 (for hot giants)
        # Compute radii for "Terran"-like planets
        ter_mask = (masses < self._CKMc0) # filter for terran-mass objects
        com_mask = rad_mask & ter_mask # planets in terran range and missing radius value
        radii[com_mask] = self._CKC0*(masses[com_mask]**self._CKe0)
        
        # Compute radii for "Neptune"-like planets
        nep_mask = (masses < self._CKMc1) # filter for neptune-mass objects
        com_mask = rad_mask & np.logical_not(ter_mask) & nep_mask # planets in neptune range and missing radius value
        radii[com_mask] = self._CKC1*(masses[com_mask]**self._CKe1)
        
        # Compute radii for "Jovian"-like planets
        jov_mask = (masses < self._CKMc2) # filter for jovian-mass objects
        com_mask = rad_mask & np.logical_not(nep_mask) & jov_mask # planets in jovian range and missing radius value
        radii[com_mask] = self._CKC2*(masses[com_mask]**self._CKe2)
        
        # Compute radii for "stellar" objects
        ste_mask = (masses > self._CKMc2) # filter for stellar-mass objects
        com_mask = rad_mask & ste_mask # planets in stellar range and missing radius value
        radii[com_mask] = self._CKC3*(masses[com_mask]**self._CKe3)
        
        
        ##-- Compute radii assuming Thorngren 2019 (for cool giants)
        # Create mask to find planets that meet the constraints
        Mlow_mask = (masses  > self._ThMlow)
        Mhi_mask  = (masses  < self._ThMhi)
        tmp_mask  = (eqtemps < self._ThThi) & (eqtemps != 0.0)  # omit temp=0 since those are actually empties
        com_mask  = rad_mask & Mlow_mask & Mhi_mask & tmp_mask
        # Convert planet mass vector to M_jup for equation
        logmass_com = np.log10(masses[com_mask]/self.MJUP2EARTH)
        # Apply equation to said planets (including conversion back to Rad_earth)
        radii[com_mask] = (self._ThC0 + self._ThC1*logmass_com + self._ThC2*(logmass_com**2))*self.RJUP2EARTH

        return radii
    
    def approximate_masses(self,masses,radii,eqtemps):
        '''
        Approximate planet masses given the planet radii
        
        Arguments:
        masses    - ndarray of planet masses. 0-values will be replaced with approximation.
        radii     - ndarray of planet radii
        eqtemps   - ndarray of planet equilibrium temperatures (needed for Thorngren constraints)
        
        Returns:
        masses    - ndarray of planet masses after approximation.
        
        Methodology:
        - Uses Thorngren 2019 for targets with ~ 3.7R_E < R < 10.7R_E and T_eq < 1000 K.
            ref.: https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
        - Uses Chen and Kipping 2016 for all other targets.
            ref.: https://doi.org/10.3847/2515-5172/ab4353
        * Only operates on 0-valued elementes in masses vector (ie. prioritizes Archive-provided masses).
        '''
        
        ##-- Find indices for missing masses so we don't replace Archive-provided values
        mss_mask = (masses == 0.0)

        
        ##-- Compute masses assuming Chen&Kipping 2016 (for hot giants)
        # Transition points (in radii) - computed by solving at critical mass points
        R_TN = self._CKC1*(self._CKMc0**self._CKe1)
        R_NJ = self._CKC2*(self._CKMc1**self._CKe2)
        R_JS = self._CKC3*(self._CKMc2**self._CKe3)
        
        # Compute masses for Terran objects
            # These are far below Jovian range so no concern about invertibility
        ter_mask = (radii < R_TN) # filter for terran-size objects
        com_mask = mss_mask & ter_mask # planets in terran range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC0)**(1/self._CKe0)

        # Compute masses for Neptunian objects
            # Cut off computation at lower non-invertible radius limit (Jovian-stellar crit point)
        nep_mask = (radii < R_JS) # filter for neptune-size objects in invertible range
        com_mask = mss_mask & np.logical_not(ter_mask) & nep_mask # planets in invertible neptune range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC1)**(1/self._CKe1)

        # Ignore Jovian objects since in non-invertible range

        # Compute masses for Stellar objects
            # Cut off computation at upper non-invertible radius limit (Neptune-Jovian crit point)
        ste_mask = (radii > R_NJ) # filter for stellar-size objects in invertible range
        com_mask = mss_mask & ste_mask # planets in invertible stellar range and missing mass values
        masses[com_mask] = (radii[com_mask]/self._CKC3)**(1/self._CKe3)

        
        ##-- Compute masses assuming Thorngren 2019 (for cool giants)
        #- Use mass constraints to determine applicabile domain in radii
        # Convert constraint masses to M_jup for equation and compute log10 for simplicity in eq.
        log_M = np.log10(np.array([self._ThMlow,self._ThMhi])/self.MJUP2EARTH)
        # Apply equation (including conversion back to Rad_earth)
        cool_Rbd = (self._ThC0 + self._ThC1*log_M + self._ThC2*(log_M**2))*self.RJUP2EARTH
        # Extract bounds (in Earth radii) where Thorngren is applicable
        cool_Rlow = cool_Rbd[0]; cool_Rhi = cool_Rbd[1]; 

        # Create mask to find planets that meet the bounds
        Rlow_mask = (radii   > cool_Rlow)
        Rhi_mask  = (radii   < cool_Rhi)
        tmp_mask  = (eqtemps < self._ThThi) & (eqtemps != 0.0)  # omit temp=0 since those are actually empties
        com_mask  = mss_mask & Rlow_mask & Rhi_mask & tmp_mask

        # Convert planet radius vector to R_jup for equation
        rad_com = radii[com_mask]/self.RJUP2EARTH
        # Apply equation to said planets
            # Use neg. side of quad. eq. so we get the mass values on the left side of the curve
        logM    = (-1*self._ThC1 - np.sqrt(self._ThC1**2 - 4*self._ThC2*(self._ThC0-rad_com)))/(2*self._ThC2)
        masses[com_mask] = (10**logM)/self.MJUP2EARTH    # convert back to Earth mass

        return masses