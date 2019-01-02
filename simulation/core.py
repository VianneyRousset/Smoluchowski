#!/usr/bin/env python3.6

from .utils import divergence
from scipy.ndimage.filters import laplace
from numpy import sum, power
import time

def solvr(t, p, sim):

    t1 = time.clock()

    # variables acquisition
    D = sim.data['D']
    mu = sim.data['mu']
    XYZ = sim.XYZ
    dim = len(sim.shape)
    p = p.reshape(sim.shape)
    t2 = time.clock()

    # math
    diff = D * laplace(p) / sum(power(sim.d, 2))
    t3 = time.clock()
    drift = mu * 1e-20 * divergence([-p*E for E in sim.E], sim.d)
    t4 = time.clock()

    # saving computation time 
    sim.tcomp_var   += t2 - t1
    sim.tcomp_diff  += t3 - t2
    sim.tcomp_drift += t4 - t3

    return (diff - drift).reshape(-1)
