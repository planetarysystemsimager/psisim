import numpy as np
import astropy.units as u
import astropy.constants as constants
from astropy.table import Table
import EXOSIMS.MissionSim

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

        sim = EXOSIMS.MissionSim.MissionSim(self.filename, explainFiltering=True, fillPhotometry=True, nokoMap=True)

        flux_ratios = 10**(sim.SimulatedUniverse.dMag/-2.5)  # grab for now from EXOSIMS
        angseps = sim.SimulatedUniverse.WA.value * 1000 # mas
        projaus = sim.SimulatedUniverse.d.value # au
        phase = np.arccos(sim.SimulatedUniverse.r[:,2]/sim.SimulatedUniverse.d) # planet phase  [0, pi]
        smas = sim.SimulatedUniverse.a.value # au
        eccs = sim.SimulatedUniverse.e # eccentricity
        incs = sim.SimulatedUniverse.I.value # degrees
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

        planets_table = Table(all_data, names=labels)

        self.planets = planets_table
        

