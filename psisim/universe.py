import numpy as np
import astropy.units as u
import astropy.constants as constants
from astropy.table import QTable, MaskedColumn
import scipy.interpolate as si
import pyvo
import json
from random import random
from math import pi

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
        
        # TODO: decide  units to use for photometric mags

        import EXOSIMS.SimulatedUniverse.SAG13Universe

        with open(self.filename) as ff:
            specs = json.loads(ff.read())

        # sim = EXOSIMS.MissionSim.MissionSim(self.filename, explainFiltering=True, fillPhotometry=True, nokoMap=False)
        su = EXOSIMS.SimulatedUniverse.SAG13Universe.SAG13Universe(**specs)

        flux_ratios = 10**(su.dMag/-2.5)  # grab for now from EXOSIMS
        angseps = su.WA.value * 1000 *u.mas # mas
        projaus = su.d.value * u.AU # au
        phase = np.arccos(su.r[:,2]/su.d)# planet phase  [0, pi]
        smas = su.a.value*u.AU # au
        eccs = su.e # eccentricity
        incs = su.I.value*u.deg # degrees
        masses = su.Mp  # earth masses
        radii = su.Rp # earth radii
        grav = constants.G * (masses)/(radii)**2
        logg = np.log10(grav.to(u.cm/u.s**2).value)*u.dex(u.cm/u.s**2) # logg cgs

        # stellar properties
        ras = [] # deg
        decs = [] # deg
        distances = [] # pc
        for index in su.plan2star:
            coord = su.TargetList.coords[index]
            ras.append(coord.ra.value)
            decs.append(coord.dec.value)
            distances.append(coord.distance.value)
        ras = np.array(ras)
        decs = np.array(decs)
        distances = np.array(distances)
        star_names =  np.array([su.TargetList.Name[i] for i in su.plan2star])
        spts = np.array([su.TargetList.Spec[i] for i in su.plan2star])
        su.TargetList.stellar_mass() # generate masses if haven't
        host_mass = np.array([su.TargetList.MsTrue[i].value for i in su.plan2star])*u.solMass
        host_teff = su.TargetList.stellarTeff(su.plan2star)
        # stellar photometry
        host_Bmags = np.array([su.TargetList.Bmag[i] for i in su.plan2star])
        host_Vmags = np.array([su.TargetList.Vmag[i] for i in su.plan2star])
        host_Rmags = np.array([su.TargetList.Rmag[i] for i in su.plan2star])
        host_Imags = np.array([su.TargetList.Imag[i] for i in su.plan2star])
        host_Jmags = np.array([su.TargetList.Jmag[i] for i in su.plan2star])
        host_Hmags = np.array([su.TargetList.Hmag[i] for i in su.plan2star])
        host_Kmags = np.array([su.TargetList.Kmag[i] for i in su.plan2star])
        
        # guess the radius and gravity from Vmag and Teff. This is of questionable reliability
        host_MVs = host_Vmags - 5 * np.log10(distances/10) # absolute V mag
        host_lums = 10**(-(host_MVs-4.83)/2.5) # L/Lsun
        host_radii = (5800/host_teff.value)**2 * np.sqrt(host_lums)  *u.solRad# Rsun
        host_gravs = constants.G * host_mass/(host_radii**2)
        host_logg = np.log10(host_gravs.to(u.cm/u.s**2).value) *u.dex(u.cm/(u.s**2))# logg cgs

        all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phase, smas, eccs, incs, masses, radii, logg, spts, host_mass, host_teff, host_radii, host_logg, host_Bmags, host_Vmags, host_Rmags, host_Imags, host_Jmags, host_Hmags, host_Kmags]
        labels = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phase", "SMA", "Ecc", "Inc", "PlanetMass", "PlanetRadius", "PlanetLogg", "StarSpT", "StarMass", "StarTeff", "StarRad", "StarLogg", "StarBMag", "StarVmag", "StarRmag", "StarImag", "StarJmag", "StarHmag", "StarKmag"]

        planets_table = QTable(all_data, names=labels)

        self.planets = planets_table
        
class GPIES_Universe(Universe):
    '''
    A child class of Universe that is specifically for simulating planets around the stars studied in the GPIES survey.
    '''

    def __init__(self):
        super(GPIES_Universe, self).__init__()
        
        self.planets = None

    def simulate_GPIES_Universe(self):
        '''
        A function that uses GPIES data from https://arxiv.org/pdf/1904.05358.pdf Table 4 
        '''
        import numpy as np
        import astropy.units as u
        import astropy.constants as constants
        from astropy.table import QTable, MaskedColumn
        import scipy.interpolate as si
        import pyvo
        import json
        from random import random
        from math import pi

        name = []
        StarSpT = []
        D = []
        Imag = []
        Hmag = []
        MGroup = []
        Age = []
        starMass = []
        f_Mass = []
        with open("GPIUniverse.txt") as file:
            lines = file.readlines()
            lines = lines[94:425]
            for line in lines:
                name += [line[0:16].rstrip()]
                StarSpT += [line[41:43].rstrip()]
                D += [line[60:68]]
                Imag += [line[79:82]]
                Hmag += [line[83:86]]
                MGroup += [line[87:100].rstrip()]
                Age += [line[101:104]]
                starMass += [line[108:112]]
                f_Mass += [line[113:114].rstrip()]
        name = np.array(name)
        StarSpT = np.array(StarSpT)
        D = np.array(list(np.float_(D)))*u.parsec
        Imag = np.array(list(np.float_(Imag)))
        Hmag = np.array(list(np.float_(Hmag)))
        MGroup = np.array(MGroup)
        Age = np.array(list(np.float_(Age)))*u.megayear
        starMass = np.array(list(np.float_(starMass)))*u.Msun
        f_Mass = np.array(f_Mass)

        starData = [name, StarSpT, D, Imag, Hmag, MGroup, Age, starMass, f_Mass]
        starLabels = ["Name", "StarSpT", "Distance", "Imag", "Hmag", "Moving Group", "StarAge", "StarMass", "f_Mass"]
        stars = QTable(starData, names=starLabels)



        names = []
        for i in reversed(range(len(stars))):
            if stars[i][0] in names:
                stars.remove_row(i)
            names += [stars[i][0]]
        # Code not necessary for GPIES simulation use, but can add spectral type if necessary
        #         for k in reversed(range(len(stars))):
        #             if stars['StarSpT'][k] == '-':
        #                 stars.remove_row(k)

        #         SpT_I_V = np.loadtxt("./SpT_I-V.csv", dtype = str, skiprows = 0)
        #         SpT = []
        #         I_V = []
        #         for i in SpT_I_V:
        #             SpT += [i[0]]
        #             I_V += [float(i[1])] 

        #         for j in reversed(range(len(stars['StarSpT']))):    
        #             if stars['StarSpT'][j] not in SpT:
        #                 stars.remove_row(j)

        #         Vmag = []
        #         for p in range(len(stars['StarSpT'])):
        #             for q in range(len(SpT)):
        #                 if stars['StarSpT'][p] == SpT[q]:
        #                     Vmag += [round(Imag[p] + I_V[q],2)]

        #         stars['Vmag'] = Vmag           

        hasPlanet = []
        masses = []
        SMAs = []
        starIndex = []
        starNames = []
        distances = []
        SpTypes = []
        Imags = []
        Hmags = []
        Vmags = []
        starMasses = []
        Ages = []
        for i in np.arange(len(stars)):
            probability = 1.26*(stars[i]['StarMass'].value/1.75)**1.15
            numPlanets = 1
            probs = []
            if probability > 1:
                while probability > 1:
                    numPlanets += 1
                    probability -= 1
                    probs += [1]
            probs += [probability]
            for j in probs:
                    chances = [j,1-j]
                    options = [True,False]
                    sample = np.random.choice(a = options,p = chances)
                    if sample == True:
                        probMass = random()
                        mass = (.47/(1-probMass))**(.917)*u.Mjup
                        while mass.value > 80:
                            probMass = random()
                            mass = (.47/(1-probMass))**(.917)*u.Mjup
                        masses += [mass.value]
                        probSMA = random()*1.27
                        SMA = (2.06/(1.27-probSMA))**(2.27)*u.au
                        SMAs += [SMA.value]
                        starIndex += [i]
                        starNames += [stars[i]['Name']]
                        distances += [stars[i]['Distance']]
                        SpTypes += [stars[i]['StarSpT']]
                        Imags += [stars[i]['Imag']]
                        Hmags += [stars[i]['Hmag']]
                #                 Vmags += [stars[i]['Vmag']]
                        Ages += [stars[i]['StarAge']]
                        starMasses += [stars[i]['StarMass']]
        separations = []
        for j in masses:
            rand = random()
            theta = np.arccos(rand)*u.rad # inclination angle
            nu = random()*pi*2*u.rad # true anomaly
            d1 = SMAs[masses.index(j)]*SMA.unit*np.sqrt((np.sin(theta)*np.cos(nu))**2 + np.sin(nu)**2) # this is the distance between the star and the planet as we see it
            d1 = d1.to(u.pc)
            angSep = 206265*u.arcsec*d1.value/stars[starIndex[masses.index(j)]]['Distance'].value
            separations += [angSep.value]
        masses = np.array(masses) * mass.unit
        SMAs = np.array(SMAs) * SMA.unit
        separations = np.array(separations) * angSep.unit

        planetData = [starNames, distances, SpTypes, separations, SMAs, masses, Imags, Hmags, Ages, starMasses]
        planetLabels = ["Name", "Distance", "StarSpT", "AngSep", "SMA", "PlanetMass", "StarImag", "StarHmag", "Age", "StarMass"]
        planets = QTable(planetData, names = planetLabels)   
        return planets

class ExoArchive_Universe(Universe):
    '''
    A child class of Universe that is adapted to create a universe from known NASA Exoplanet Archive Data
    
    Uses the pyVO package to read in known exoplanets
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
        
    def Load_ExoArchive_Universe(self, composite_table=True, force_new_pull=False, fill_empties=True):
        '''
        A function that reads the Exoplanet Archive data to populate the planet table
        
        Unless force_new_pull=True:
        If the filename provided in constructor is new, new data is pulled from the archive
        If the filename already exists, we try to load that file as an astroquery QTable
        
        Kwargs:
        composite_table  - Bool. True [default]: pull "Planetary Systems Composite
                           Parameters Table". False: pull simple "Planetary Systems" Table
                           NOTE: see Archive website for difference between these tables
                           
        force_new_pull   - Bool. False [default]: loads table from filename if filename
                           file exists. True: pull new archive data and overwrite filename
                           
        fill_empties     - Bool. True [default]: approximate empty table values using
                           other values present in data. Ex: radius, mass, logg, angsep, etc.
                           NOTE: When composite_table=True we do not approximate the planet 
                             radius or mass; we keep the archive-computed approx.
                           
        
        Approximation methods:
        - AngSep     - theta[mas] = SMA[au]/distance[pc] * 1e3
        - logg       - logg [log(cgs)] = log10(G*mass/radius**2)
        - StarLum    - absVmag = Vmag - 5*log10(distance[pc]/10)
                       starlum[L/Lsun] = 10**-(absVmag-4.83)/2.5
        - StarRad    - rad[Rsun] = (5800/Teff[K])**2 *sqrt(starlum)
        - PlanetRad  - ** when composite_table=True, keep archive-computed approx
                       Based on Thorngren 2019 and Chen&Kipping 2016
        - PlanetMass - ^^ Inverse of PlanetRad
        
        
        *** Note: the resulting planet table will have nan's where data is missing/unknown. 
            Ex. if a planet lacks a radius val, the 'PlanetRadius' for will be np.nan        
        '''

        #-- Define columns to read. NOTE: add columns here if needed. 
          # col2pull entries should be matched with colNewNames entries
        col2pull =  "pl_name,hostname,pl_orbsmax,pl_orbeccen,pl_orbincl,pl_bmasse,pl_rade," + \
                    "pl_eqt,ra,dec,sy_dist,st_spectype,st_mass,st_teff," + \
                    "st_rad,st_logg,st_lum,st_age,st_vsin,st_radv," + \
                    "st_met,sy_plx,sy_bmag,sy_vmag,sy_rmag,sy_icmag," + \
                    "sy_jmag,sy_hmag,sy_kmag,discoverymethod"
        colNewNames = ["PlanetName","StarName","SMA","Ecc","Inc","PlanetMass","PlanetRadius",
                       "PlanetTeq","RA","Dec","Distance","StarSpT","StarMass","StarTeff",
                       "StarRad","StarLogg","StarLum","StarAge","StarVsini","StarRadialVelocity",
                       "StarZ","StarParallax","StarBMag","StarVmag","StarRmag","StarImag",
                       "StarJmag","StarHmag","StarKmag","DiscoveryMethod"]

        #-- Load/Pull data depending on provided filename
        import os
        if os.path.isfile(self.filename) and not force_new_pull:
            
            # Existing filename was provided so let's try use that
            
            print("%s already exists:\n    we'll attempt to read this file as an astropy QTable"%self.filename)

            NArx_table = QTable.read(self.filename, format='ascii.ecsv')
            
            # Check that the provided table file matches the requested table type
            if NArx_table.meta['isPSCOMPPARS'] != composite_table:
                err0 = '%s contained the wrong table-type:'%self.filename
                err1 = 'pscomppars' if composite_table else 'ps'
                err2 = 'pscomppars' if NArx_table.meta['isPSCOMPPARS'] else 'ps'
                err3 = " Expected '{}' table but found '{}' table.".format(err1,err2)
                err4 = ' Consider setting force_new_pull=True.'
                raise ValueError(err0+err3+err4)

        else:
            # New filename was provided or a new pull was explicitly requested. Pull new data
            
            if not force_new_pull:
                print("%s does not exist:\n    we'll pull new data from the archive and save it to this filename"%self.filename)
            else:
                print("%s may or may not exist:\n    force_new_pull=True so we'll pull new data regardless and overwrite as needed"%self.filename) 

            # Import pyVO package used to query the Exoplanet Archive
            import pyvo as vo

            # Create a "service" which can be used to access the archive TAP server
            NArx_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")

            # Create a "query" string formatted per the TAP specifications
              # 'select': specify which columns to pull
              # 'from': specify which table to pull 
              # 'where': (optional) specify parameters to be met when choosing what to pull
                # Add where flag for ps to only pull the best row for each planet
            tab2pull = "pscomppars" if composite_table else "ps where default_flag=1"
            query = "select "+col2pull+" from "+tab2pull

            # Pull the data and convert to astropy masked QTable
            NArx_res = NArx_service.search(query) 
            
            NArx_table = QTable(NArx_res.to_table())

            # Add a flag to the table metadata to denote what kind of table it was
              # This'll prevent trying to read the table as the wrong type later
            NArx_table.meta['isPSCOMPPARS'] = composite_table
            # Save raw table for future use 
            NArx_table.write(self.filename,format='ascii.ecsv',overwrite=force_new_pull)
            # Read table back in to ensure that formatting from a fresh pull matches
              # the formatting from an old pull (as done when filename exists)
            NArx_table = QTable.read(self.filename, format='ascii.ecsv')
            
        #-- Rename columns to psisim-expected names
        NArx_table.rename_columns(col2pull.split(','),colNewNames)
        
        #-- Change fill value from default 1e20 to np.nan
        for col in NArx_table.colnames:
            if isinstance(NArx_table[col],MaskedColumn) and isinstance(NArx_table[col].fill_value,(int,float)):
                # Only change numeric fill values to nan
                NArx_table[col].fill_value = np.nan
        
        #-- Add new columns for values not easily available or computable from table
          # TODO: for now, these are masked but we should find a good way to populate them
        NArx_table.add_columns([MaskedColumn(length=len(NArx_table),mask=True,fill_value=np.nan)]*3,
                               names=['Flux Ratio','ProjAU','Phase'])
        
        if fill_empties:
            #-- Compute missing planet columns
            # Compute missing masses and radii using mass-radius relations
            if not composite_table:
                # NOTE: composite table already has radius-mass approximation so we'll
                  # only repeat them if we don't pull that table
                    
                # Convert masked columns to ndarrays with 0's instead of mask
                  # as needed by the approximate_... functions
                masses   = np.array(NArx_table['PlanetMass'].filled(fill_value=0.0))
                radii    = np.array(NArx_table['PlanetRadius'].filled(fill_value=0.0))
                eqtemps  = np.array(NArx_table['PlanetTeq'].filled(fill_value=0.0))
                # Perform approximations
                radii = self.approximate_radii(masses,radii,eqtemps)
                masses = self.approximate_masses(masses,radii,eqtemps)
                # Create masks for non-zero values (0's are values where data was missing)
                rad_mask = (radii != 0.)
                mss_mask = (masses != 0.)
                # Create mask to only missing values in NArx_table with valid values
                rad_mask = NArx_table['PlanetRadius'].mask & rad_mask
                mss_mask = NArx_table['PlanetMass'].mask & mss_mask
                # Place results back in the table
                NArx_table['PlanetRadius'][rad_mask] = radii[rad_mask]
                NArx_table['PlanetMass'][mss_mask] = masses[mss_mask]
        
            # Angular separation
            NArx_table['AngSep'] = NArx_table['SMA']/NArx_table['Distance'] * 1e3
            # Planet logg
            grav = constants.G * (NArx_table['PlanetMass'].filled()*u.earthMass) / (NArx_table['PlanetRadius'].filled()*u.earthRad)**2
            NArx_table['PlanetLogg'] = np.ma.log10(MaskedColumn(np.ma.masked_invalid(grav.cgs.value),fill_value=np.nan))  # logg cgs

            #-- Guess star luminosity, radius, and gravity for missing (masked) values only
              # The guesses will be questionably reliabile
            # Star Luminosity
            host_MVs = NArx_table['StarVmag'] - 5*np.ma.log10(NArx_table['Distance']/10)  # absolute v mag
            host_lum = -(host_MVs-4.83)/2.5    #log10(L/Lsun)
            NArx_table['StarLum'][NArx_table['StarLum'].mask] = host_lum[NArx_table['StarLum'].mask]

            # Star radius
            host_rad = (5800/NArx_table['StarTeff'])**2 *np.ma.sqrt(10**NArx_table['StarLum'])   # Rsun
            NArx_table['StarRad'][NArx_table['StarRad'].mask] = host_rad[NArx_table['StarRad'].mask]

            # Star logg
            host_grav = constants.G * (NArx_table['StarMass'].filled()*u.solMass) / (NArx_table['StarRad'].filled()*u.solRad)**2
            host_logg = np.ma.log10(np.ma.masked_invalid(host_grav.cgs.value))  # logg cgs
            NArx_table['StarLogg'][NArx_table['StarLogg'].mask] = host_logg[NArx_table['StarLogg'].mask]
        else:
            # Create fully masked columns for AngSep and PlanetLogg
            NArx_table.add_columns([MaskedColumn(length=len(NArx_table),mask=True,fill_value=np.nan)]*2,
                       names=['AngSep','PlanetLogg'])

            
        #-- Deal with units (conversions and Quantity multiplications)
        # Set host luminosity to L/Lsun from log10(L/Lsun)
        NArx_table['StarLum'] = 10**NArx_table['StarLum']    # L/Lsun
        
        # Make sure all number fill_values are np.nan after the column manipulations
        for col in NArx_table.colnames:
            if isinstance(NArx_table[col],MaskedColumn) and isinstance(NArx_table[col].fill_value,(int,float)):
                # Only change numeric fill values to nan
                NArx_table[col].fill_value = np.nan
                
        # Fill in masked values 
        NArx_table = NArx_table.filled()
        # Apply units
        NArx_table['SMA'] *= u.AU
        NArx_table['Inc'] *= u.deg
        NArx_table['PlanetMass'] *= u.earthMass
        NArx_table['PlanetRadius'] *= u.earthRad
        NArx_table['PlanetTeq'] *= u.K
        NArx_table['RA'] *= u.deg
        NArx_table['Dec'] *= u.deg
        NArx_table['Distance'] *= u.pc
        NArx_table['StarMass'] *= u.solMass
        NArx_table['StarTeff'] *= u.K
        NArx_table['StarRad'] *= u.solRad
        NArx_table['StarLogg'] *= u.dex(u.cm/(u.s**2))
        NArx_table['StarLum'] *= u.solLum
        NArx_table['StarAge'] *= u.Gyr
        NArx_table['StarVsini'] *= u.km/u.s
        NArx_table['StarRadialVelocity'] *= u.km/u.s
        #NArx_table['StarZ']  *= u.dex
        NArx_table['StarParallax'] *= u.mas
        NArx_table['ProjAU'] *= u.AU
        NArx_table['Phase'] *= u.rad
        NArx_table['AngSep'] *= u.mas
        NArx_table['PlanetLogg'] *= u.dex(u.cm/(u.s**2))
        
        self.planets = NArx_table
    
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
