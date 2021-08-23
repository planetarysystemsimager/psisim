import numpy as np
import psisim.instrument as instrument

def test_hispec_dar_coupling():
    """
    Test calculation of coupling into the single mode fiber. 
    """
    inst = instrument.hispec()
    wvs = np.linspace(1.98, 2.38, 100)
    wvs0 = wvs[-1]
    inst.set_observing_mode(60, 1, "TwoMASS-K", wvs, zenith=60)
    dar_coupling = inst.get_dar_coupling_throughput(wvs)
    assert np.all((dar_coupling <= 1) & (dar_coupling >= 0))

    dar_coupling_with_refwv = inst.get_dar_coupling_throughput(wvs, wvs0=wvs0)
    assert dar_coupling_with_refwv[-1] == 1

    # zenith of 0 gives no loss in coupling due to DAR
    inst.set_observing_mode(60, 1, "TwoMASS-K", wvs, zenith=0)
    dar_coupling_zero = inst.get_dar_coupling_throughput(wvs)
    assert np.all(dar_coupling_zero == 1)

if __name__ == "__main__":
    test_hispec_dar_coupling()