#!/usr/bin/env python

import numpy as np
import simulation.utils as ut
import simulation
import scipy.constants as const

res = 0.01
x,y,v = np.loadtxt('input/out_substrate_ElectrostaticPotential.txt', unpack=True)
lim, XYZ, V = ut.rasterized_region((x,y), v, res)

X,Y = XYZ
shape = X.shape
p0 = ut.single_dot((2,1.5), *XYZ)
sim = simulation.Simulation()
mu = 600*1e10 # um^2 * V^-1 * s^-1
k,T = const.k, 300
D = mu*k*T/const.e
sim.init(dt=0.1e-15, p0=p0, V=-V, D=D, g=1/mu, X=X, Y=Y, padding=[(0, 5), (5, 5)])
sim.run(0.5e-12, sampling=50)
F = np.linalg.norm(np.gradient(sim.data['V'], *sim.d), axis=0)
sim.export_images('output', prefix='img_', log=True, title=r'$t = {tf}\quad I = {sum:6.3f}$', background=-F)
