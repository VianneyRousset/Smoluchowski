#!/usr/bin/env python

from simulation import Simulation
from simulation.utils import meshgrid_from_lim, single_point
import numpy as np

def test_dirac(shape, dt, t, D, V):
    dim = len(shape)
    XYZ = meshgrid_from_lim(shape, *[[-10, 10]]*dim)
    p0 = single_point(shape)
    sim = Simulation()
    XYZ = {k:v for k,v in zip(['X', 'Y', 'Z'], XYZ)}
    sim.init(shape=shape, dt=dt, p0=p0, D=D, V=V, **XYZ)
    sim.run(t)
    return sim


def diffusion(mu, T=300):
    from scipy.constants import k
    return k*T*mu


def gamma(mu):
    return 1/mu


if __name__ == '__main__':

    X = load_TDR('X.csv')
    mu = load_TDR('input/mu.csv')
    V = load_TDR('input/V.csv')
    D = diffusion(mu, T)
    g = gamma()
    p0 = single_point(X.shape)

    sim = Simulation()
    sim.init(shape=X.shape, dt=0.1, t=10, p0=p0, D=D, V=V, g=1, X=X, Y=Y)

