from psisim import telescope,instrument,observation,spectrum,universe,plots


tmt = telescope.TMT()
psi_blue = instrument.psi_blue()
psi_blue.set_observing_mode(60,40,'z',10) #60s, 40 exposures,z-band, R of 10

uni = universe.ExoSims_Universe()

planet_table = uni.planet_table

