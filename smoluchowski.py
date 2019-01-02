#!/usr/bin/env python3.6

import simulation
import simulation.utils as ut

import numpy as np
import scipy.constants as const
from sys import argv, exit
from os import remove


def load_data(prefix, res, crop=None):
    fields = {
            'V':        'ElectrostaticPotential',
            'mu_e':     'eMobility',
            'mu_h':     'hMobility',
            'R':        'TotalRecombination',
            'tau_e':    'eLifetime',
            'tau_h':    'hLifetime',
            'a_e':      'eAlphaAvalanche',
            'a_h':      'hAlphaAvalanche',
            }
    data = {f:load_field(f'{prefix}{fname}.txt', res, crop)
            for f,fname in fields.items()}
    fields = {k:v[2] for k,v in data.items()}
    lim = np.array([i[0] for i in data.values()][0])
    XYZ = np.array([i[1] for i in data.values()][0])
    return fields,lim,XYZ


def load_field(path, res, crop=None):
    data = np.loadtxt(path, unpack=True)
    xyz,f = data[:-1],data[-1]
    return ut.rasterized_region(tuple(xyz), f, res, crop)


if __name__ == '__main__':

    ROI = [(0,6), (0,2)]
    
    # read arguments
    if len(argv) < 2:
        ut.printerr(f'Usage: {argv[0]} prefix [res] \n'
                'For example, \'./smoluchowski.py n20_ 0.05\' will '
                'use files starting with \'n20_\' a 0.05um resolution')
        exit(1)
    prefix = argv[1]
    res = 0.05
    if len(argv) > 2:
        res = float(argv[2])

    # load data
    print('* Loading fields')
    fields,lim,XYZ = load_data(prefix, res, crop=ROI)

    # simulation
    print('* Inititializing simulator')
    dt  = 100e-9
    t   = np.mean(fields['tau_e'])
    p0  = -(fields['R'] * (fields['R'] < 0))
    V   = fields['V']
    mu = fields['mu_e'] * 1e4 # cm^2 * V^-1 * s^-1 -> um^2 * V^-1 * s^-1
    k,T = const.k, 300
    D = mu*k*T/const.e
    X,Y = XYZ

    sim = simulation.Simulation()
    sim.init(shape=X.shape, dt=dt, p0=p0, V=V, D=D, mu=mu, X=X, Y=Y)

    print('** V (min)\t= {: .2e}'.format(np.min(V)))
    print('** V (max)\t= {: .2e}'.format(np.max(V)))
    print('** mu (mean)\t= {: .2e}'.format(np.mean(mu)))
    print('** D (mean)\t= {: .2e}'.format(np.mean(D)))
    print('** tau (mean)\t= {: .2e}'.format(t))
    print('** Size [px]\t=  {}x{}'.format(*sim.XYZ[0].shape))

    # starting simulation
    print('* Starting simulation')
    sim.run(t, sampling=1)

    # export images
    print('* Exporting images')
    E = np.linalg.norm(sim.E, axis=0)
    sim.export_field_image('D', 'output/D.png', log=False, title=r'Diffusion coef. $D$')
    sim.export_field_image('mu', 'output/mu.png', log=False, title=r'Mobility. $\mu$')
    sim.export_field_image('V', 'output/V.png', log=False, title=r'Electrostatic potential $V$')

    sim.export_images('output', prefix='img_', log=True, title='$t = {tf}$', background=-E)

    print('* DONE *')

