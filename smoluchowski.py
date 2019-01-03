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

    # unit conversion to [um]
    fields['mu_e']  = fields['mu_e'] * 1e4
    fields['mu_h']  = fields['mu_h'] * 1e4
    fields['R']     = fields['R'] * 1e-12
    fields['a_e']   = fields['a_e'] * 1e-4
    fields['a_h']   = fields['a_h'] * 1e-4
    
    # other fields and scalar
    fields['tau_e'] = np.mean(fields['tau_e'])
    fields['tau_h'] = np.mean(fields['tau_h'])
    fields['T'] = 300
    fields['D_e'] = const.k*fields['T']/const.e * fields['mu_e']
    fields['D_h'] = const.k*fields['T']/const.e * fields['mu_h']

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
    data,lim,(X,Y)= load_data(prefix, res, crop=ROI)

    # simulation
    print('* Inititializing simulator')
    dt  = 1e-10
    t   = 1e-7
    p0  = -(data['R'] * (data['R'] < 0))
#    p0  = ut.gaussian_dot((1.5, 1.8), 0.1, X, Y)
    sim = simulation.Simulation()
    sim.init(
            shape   = X.shape,
            dt      = dt,
            p0      = p0,
            V       = data['V'],
            D       = data['D_e'],
            mu      = data['mu_e'],
            a_e     = data['a_e'],
            a_h     = data['a_h'],
            X       = X,
            Y       = Y,
            padding = [(0, 1), (1, 1)],
            a_region = data['a_e'] >= 1,
            )

    print('** V (min)\t= {: .2e} V'.format(np.min(data['V'])))
    print('** V (max)\t= {: .2e} V'.format(np.max(data['V'])))
    print('** E (max)\t= {: .2e} V um-1'.format(np.max(sim.E)))
    print('** E (min)\t= {: .2e} V um-1'.format(np.min(sim.E)))
    print('** mu (mean)\t= {: .2e} um2 V-1 s-1'.format(np.mean(data['mu_e'])))
    print('** D (mean)\t= {: .2e} um2 s-1'.format(np.mean(data['D_e'])))
    print('** tau (mean)\t= {: .2e} s'.format(data['tau_e']))
    print('** Size [px]\t=  {}x{} px'.format(*sim.XYZ[0].shape))

    # starting simulation
    print('* Starting simulation')
    sim.run(t, sampling=1)

    # avalanche count
    print('* Plotting avalanche count')
    from matplotlib import pyplot as plt
    t = [t for t in sim.data]
    a = [sim.data[t, 'a'] for t in sim.data]
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(t, a)
    fig.savefig('output/a.pdf')


    # export images
    print('* Exporting images')
    E = np.linalg.norm(sim.E, axis=0)
    sim.export_field_image('D', 'output/D.png', log=False, title=r'Diffusion coef. $D$')
    sim.export_field_image('mu', 'output/mu.png', log=False, title=r'Mobility. $\mu$')
    sim.export_field_image('V', 'output/V.png', log=False, title=r'Electrostatic potential $V$')
    sim.export_field_image('a_e', 'output/a_e.png', log=False, title=r'Ionization coefficient $\alpha_e$')
    sim.export_field_image('a_h', 'output/a_h.png', log=False, title=r'Ionization coefficient $\alpha_h$')
    sim.export_field_image('E', 'output/E.png', log=False, title=r'Electric field $E$')
    sim.export_field_image('a_region', 'output/a_region.png', log=False, title=r'Avalanche region')

    sim.export_images('output', prefix='img_', log=False, colorbar=True, title='$t = {tf}$')

    print('* DONE *')

