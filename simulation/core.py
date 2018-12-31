#!/usr/bin/env python3.6

from .utils import divergence
from scipy.ndimage.filters import laplace
from numpy import sum, power

def solvr(t, p, sim):
    D = sim.data['D']
    mu = sim.data['mu']
    XYZ = sim.XYZ
    dim = len(sim.shape)
    p = p.reshape(sim.shape)
    diff = D * laplace(p) / sum(power(sim.d, 2))
    drift = mu * divergence(sim.d, *[-p*E for E in sim.E])
    return (diff - drift).reshape(-1)
