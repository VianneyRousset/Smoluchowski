#!/usr/bin/env python

import simulation
import simulation.utils as ut

import numpy as np
import scipy.constants as const
from sys import exit
from os import remove


def load_field(path, res):
    data = np.loadtxt(path, unpack=True)
    xyz,f = data[:-1],data[-1]
    return ut.rasterized_region(tuple(xyz), f, res)


def max_deviation(v):
    v = np.array(v)
    vmean = np.mean(v, axis=0)
    return np.max(np.abs(v - vmean))


if __name__ == '__main__':

    res = 0.05
    node = '1130'
    maxdev = 1e-9
    dt = 1e-9
    t = 0.1e-6
    data_file_path = 'output/data.h5'

    # load fields
    print('* Loading fields')
    fields = {
            'V':        'ElectrostaticPotential',
            'mu_e':     'eMobility',
            'mu_h':     'hMobility',
            'G':        'TotalRecombination',
            'tau_e':    'eLifetime',
            'tau_h':    'hLifetime',
            }
    data = {f:load_field(f'input/n{node}_{fname}.txt', res) for f,fname in fields.items()}
    fields = {k:v[2] for k,v in data.items()}
    lim = [i[0] for i in data.values()]
    XYZ = [i[1] for i in data.values()]

    ## check deviation on lim
    print('** Checking max lim deviation... ', end='')
    m = max_deviation(lim)
    if  m < maxdev:
        print(f'OK ({m} < {maxdev})')
    else:
        print(f'FAIL ({m} > {maxdev})')
        exit(1)

    ## check deviation on XYZ
    print('** Checking max XYZ deviation... ', end='')
    m = max_deviation(XYZ)
    if  m < maxdev:
        print(f'OK ({m} < {maxdev})')
    else:
        print(f'FAIL ({m} > {maxdev})')
        exit(1)

    lim = np.mean(lim, axis=0)
    XYZ = np.mean(XYZ, axis=0) 
    
    # some constants
#    p0 = fields['G']
    p0 = ut.single_dot([2, 1.5], *XYZ)
    V = fields['V']
    mu = fields['mu_e'] * 1e4 # cm^2 * V^-1 * s^-1 -> um^2 * V^-1 * s^-1
    k,T = const.k, 300
    D = mu*k*T/const.e
    X,Y = XYZ
    tau = np.mean(fields['tau_e'])

    print('** V (mean)\t= {: .2e}'.format(np.mean(V)))
    print('** mu (mean)\t= {: .2e}'.format(np.mean(mu)))
    print('** D (mean)\t= {: .2e}'.format(np.mean(D)))
    print('** tau (mean)\t= {: .2e}'.format(tau))

    # init simulator
    print('* Inititializing simulator')
    try:
        remove(data_file_path)
    except:
        pass
    sim = simulation.Simulation(data_file_path)
    sim.init(shape=XYZ[0].shape, dt=dt, p0=p0, V=V, D=D, mu=mu, X=X, Y=Y)

    # starting simulation
    print('* Starting simulation')
    sim.run(t, sampling=100)

    # export images
    print('* Exporting images')
    E = np.linalg.norm(sim.E, axis=0)
    sim.export_field_image('D', 'output/D.png', log=False, title=r'Diffusion coef. $D$')
    sim.export_field_image('mu', 'output/mu.png', log=False, title=r'Mobility. $\mu$')
    sim.export_field_image('V', 'output/V.png', log=False, title=r'Electrostatic potential $V$')

    sim.export_images('output', prefix='img_', log=True, title='$t = {tf}$', background=-E)

    print('* DONE *')

