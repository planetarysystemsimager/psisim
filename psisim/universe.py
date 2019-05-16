import numpy as np
from astropy.table import Table
import EXOSIMS.MissionSim

class Universe():
	'''
	A universe class that includes

	Inherited from EXOSIMS? TBD

	Properties:
	planets	- A planet table that holds all the planet properties [Astropy table]

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
		self.planet_table = None

	def simulate_EXOSIMS_Universe(self):
		'''
		A function that runs EXOSIMS and takes the output to populate the planets table
		'''

		sim = EXOSIMS.MissionSim.MissionSim(self.filename, explainFiltering=True, fillPhotometry=True, nokoMap=True)

		flux_ratios = 10**(sim.SimulatedUniverse.dMag/-2.5)  # grab for now from EXOSIMS
		angseps = sim.SimulatedUniverse.WA.value * 1000 # mas
		projaus = sim.SimulatedUniverse.d.value # au
		phis = sim.SimulatedUniverse.phi # planet phase  [0, 1]
		smas = sim.SimulatedUniverse.a.value # au
		eccs = sim.SimulatedUniverse.e # eccentricity
		incs = sim.SimulatedUniverse.I.value # degrees
		masses = sim.SimulatedUniverse.Mp.value # earth masses
		radii = sim.SimulatedUniverse.Rp.value # earth radii


		# stellar properties
		ras = [] # deg
		decs = [] # deg
		distances = [] # pc
		for index in sim.SimulatedUniverse.plan2star:
			coord = sim.TargetList.coords[index]
			ras.append(coord.ra.value)
			decs.append(coord.dec.value)
			distances.append(coord.distance.value)
		# stellar photometry
		host_Bmags = np.array([sim.TargetList.Bmag[i] for i in sim.SimulatedUniverse.plan2star])
		host_Vmags = np.array([sim.TargetList.Vmag[i] for i in sim.SimulatedUniverse.plan2star])
		host_Rmags = np.array([sim.TargetList.Rmag[i] for i in sim.SimulatedUniverse.plan2star])
		host_Imags = np.array([sim.TargetList.Imag[i] for i in sim.SimulatedUniverse.plan2star])
		host_Jmags = np.array([sim.TargetList.Jmag[i] for i in sim.SimulatedUniverse.plan2star])
		host_Hmags = np.array([sim.TargetList.Hmag[i] for i in sim.SimulatedUniverse.plan2star])
		host_Kmags = np.array([sim.TargetList.Kmag[i] for i in sim.SimulatedUniverse.plan2star])
		star_names =  np.array([sim.TargetList.Name[i] for i in sim.SimulatedUniverse.plan2star])

		all_data = [star_names, ras, decs, distances, flux_ratios, angseps, projaus, phis, smas, eccs, incs, masses, radii, host_Bmags, host_Vmags, host_Rmags, host_Imags, host_Jmags, host_Hmags, host_Kmags]
		labels = ["StarName", "RA", "Dec", "Distance", "Flux Ratio", "AngSep", "ProjAU", "Phi", "SMA", "Ecc", "Inc", "PlanetMass", "PlanetRadius", "StarBMag", "StarVmag", "StarRmag", "StarImag", "StarJmag", "StarHmag", "StarKmag"]

		planet_table = Table(all_data, names=labels)

		self.planet_table = planet_table
		

