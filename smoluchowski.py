#!/usr/bin/env python3.6

import simulation
import simulation.utils as ut

import numpy as np
import scipy.constants as const

res_default = 0.1
out_default = './'

def read_argv():
    from sys import argv, exit
    from optparse import OptionParser

    parser = OptionParser(usage='%prog [-h] [-t T] [-r RES] [-R ROI] [-p PAD] '
            '[-s] [-d TS] [-a] [-o OUT] [-w WL] [-l LEN] -a MODE INPUT')

    parser.description = 'Compute field drift-diffusion time evolution using ' \
        'input prefix INPUT'

    parser.add_option('-t', '--time', dest='t_goal', metavar='TIME',
            default=None, help='set simulation goal time, default is computed '
            'with the mean of particle lifetime')

    parser.add_option('-r', '--resolution', dest='res', metavar='RES',
            default=res_default, help=f'set simulation resolution in distance '
            'unit, default is {res_default}')

    parser.add_option('-R', '--roi', dest='ROI', metavar='ROI',
            default=None, help=f'set region of interest to crop in the '
            'following format: x-,x+,y-,y+,z-,z+')

    parser.add_option('-p', '--padding', dest='padding', metavar='PAD',
            default=None, help=f'set the padding to be added in distance unit '
            'in the following format: x-,x+,y-,y+,z-,z+')

    parser.add_option('-s', '--plot-static', action='store_true',
            dest='plot-static', default=False, help='plot static fields '
            '(2D simulation only)')

    parser.add_option('-d', '--plot-dynamic', dest='t_sampling', metavar='TS',
            default=False, help='plot dynamic evolution with sampling perdiod '
            'TS (2D simulation only)')

    parser.add_option('-o', '--output', dest='output', metavar='OUTPUT',
            default=out_default, help='set plot output prefix')

    parser.add_option('-m', '--mode', dest='mode', metavar='MODE',
            default='', help='set simulation mode: DCR or PDP')

    parser.add_option('-w', '--wavelength', dest='wavelength', metavar='WL',
            default=None, help='operating wavelength [nm], needed for PDP mode')

    parser.add_option('-l', '--length', dest='active_length', metavar='LEN',
            default=None, help='set active area length (radius in 3D) LEN, needed for PDP mode in 2D and 3D')

    parser.add_option('-a', '--avalanche-only', dest='avalanche',
            action='store_true', default=False, help='only save dynamic '
            'avalanche result data in txt, requieres -d')


    options,input_prefix= parser.parse_args()

    # missing input prefix
    if len(input_prefix) != 1:
        ut.printerr(parser.format_help())
        exit(1)

    # is plot dynamic fields true
    options = options.__dict__
    options['plot-dynamic'] = options['t_sampling'] is not False

    # invalid simulation mode
    options['mode'] = options['mode'].upper().strip()
    if options['mode'] not in ['DCR', 'PDP']:
        ut.printerr('Invalid simulation mode: ', options['mode'])
        ut.printerr(parser.format_help())
        exit(1)

    # missing wavelength in PDP mode
    if options['mode'] == 'PDP' and options['wavelength'] is None:
        ut.printerr('Missing wavelength (requiered for PDP mode)')
        ut.printerr(parser.format_help())
        exit(1)

    # missing active area length in PDP mode
    if options['mode'] == 'PDP' and options['active_length'] is None:
        ut.printerr('Missing active area length (requiered for PDP mode)')
        ut.printerr(parser.format_help())
        exit(1)

    # missing dynamic with avalanche
    if options['avalanche'] and not options['plot-dynamic']:
        ut.printerr('-a option requieres -d')
        ut.printerr(parser.format_help())
        exit(1)

    # convertions
    if options['t_goal'] is not None:
        options['t_goal']   = float(options['t_goal'])
    options['res']      = float(options['res'])
    if options['ROI'] is not None:
        options['ROI']      = [float(x) for x in options['ROI'].split(',')]
        options['ROI']      = np.array(options['ROI']).reshape(-1,2)
    if options['padding'] is not None:
        options['padding']  = [float(x) for x in options['padding'].split(',')]
        options['padding']  = np.array(options['padding']).reshape(-1,2)
    if options['plot-dynamic']:
        options['t_sampling']   = float(options['t_sampling'])
    options['wavelength']   = float(options['wavelength'])
    options['active_length']   = float(options['active_length'])

    return options,input_prefix[0]


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

    def load_field(path, res, crop=None):
        data = np.loadtxt(path, unpack=True)
        xyz,f = data[:-1],data[-1]
        return ut.rasterize_region(tuple(xyz), f, res, crop)

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


def prepare_simulation(options, data, XYZ, particle):

    if particle not in ['electron', 'hole']:
        raise ValueError(f'Invalid particle \'{particle}\'')

    if options['mode'] == 'PDP':
        p0 = PDP_profile(options, data, XYZ)
    elif options['mode'] == 'DCR':
        p0 = DCR_profile(options, data, XYZ, tau)

    sim = simulation.Simulation(avalanche_only=options['avalanche'],
            particle=particle)

    from scipy.ndimage import gaussian_filter
    sim.init(
            p0      = p0,
            shape   = XYZ[0].shape,
            V       = data['V'],
            D       = data['D_e'] if particle == 'electron' else data['D_h'],
            mu      = data['mu_e'] if particle == 'electron' else data['mu_h'],
            a_e     = data['a_e'],
            a_h     = data['a_h'],
            padding = options['padding'],
            a_region =  data['a_e'] / np.max(data['a_e']),
            **{k:v for k,v in zip('XYZ', XYZ)},
            )

    return sim


def get_timing(options, data, particle):

    if particle not in ['electron', 'hole']:
        raise ValueError(f'Invalid particle \'{particle}\'')

    # t_goal
    t_goal = options['t_goal']
    if t_goal is None:
        tau = 'tau_e' if particle == 'electron' else 'tau_h'
        tau = np.mean(data[tau])
        t_goal = tau 

    # t_sampling
    t_sampling = options['t_sampling'] if options['plot-dynamic'] else np.inf

    return t_goal, t_sampling


def print_simulation_info(sim, data, t_goal, t_sampling, particle):

    tau = 'tau_e' if particle == 'electron' else 'tau_h'

    info = {
            'Vmin':         np.min(sim.data['V']),
            'Vmax':         np.max(sim.data['V']),
            'Emin':         np.min(sim.E),
            'Emax':         np.max(sim.E),
            'mu':           np.mean(sim.data['mu']),
            'D':            np.mean(sim.data['D']),
            'tau':          np.mean(data[tau]),
            't_goal':       t_goal,
            't_sampling':   t_sampling,
            }

    txt =   '** V (min)\t= {Vmin: .2e} V\n' \
            '** V (max)\t= {Vmax: .2e} V\n' \
            '** E (min)\t= {Emin: .2e} V um-1\n' \
            '** E (max)\t= {Emax: .2e} V um-1\n' \
            '** mu (mean)\t= {mu: .2e} um2 V-1 s-1\n' \
            '** D (mean)\t= {D: .2e} um2 s-1\n' \
            '** tau (mean)\t= {tau: .2e} s\n' \
            '** t_goal\t= {t_goal: .2e} s\n' \
            '** t_sampling\t= {t_sampling: .2e} s'

    print(txt.format(**info))
    size = 'x'.join([str(w) for w in sim.data['X'].shape])
    print('** Size [px]\t=  {} px\n'.format(size))


def DCR_profile(options, data, XYZ):
    shape = XYZ[0].shape
    return np.zeros(shape)


def PDP_profile(options, data, XYZ):
    d = ut.get_dx(*XYZ)
    a = ut.absorption_coefficient_silicon(options['wavelength'])
    x = XYZ[-1]
    dx = d[-1]
    p = a*np.exp(-a*x)*dx
    r = np.linalg.norm(XYZ[:-1], axis=0)
    p[r > options['active_length']] = 0
    return p / (np.sum(p) * np.product(d))


def plot_static(sim, out):

    print('* Exporting static fields')

    fields = [
            ('D', 'Diffusion coef.', 'D', r'\square\micro\meter'),
            ('mu', 'Mobility', r'\mu', r'\cubic\micro\meter\per\volt'),
            ('V', 'Potential', r'V', r'\volt'),
            ('a_e', r'Electron avalanche coef.', r'\alpha_e', r'\per\micro\meter'),
            ('a_h', r'Hole avalanche coef.', r'\alpha_h', r'\per\micro\meter'),
            ('E', r'Electric field', 'E', r'\volt\per\micro\meter'),
            ('a_region', 'Avalanche region', None, None),
            ('X', 'X', None, None),
            ('Y', 'Y', None, None),
            ]

    N = len(fields)
    for i,(f,t,n,u) in enumerate(fields):
        if n and u:
            t += ' ' + ut.siLabel(f'${n}$', u)
        elif n:
            t += '$' + n + '$'

        print('\b'*8, end='', flush=True)
        print(f'{i+1:d}/{N:d}', end='', flush=True)
        sim.export_static_field_img(f, f'{out}{f}.png', log=False, title=t)
    print('\b'*8, end='', flush=True)


def plot_dynamic(sim, out, avalanche=False):

    # avalanche only
    filepath = f'{out}avalanche.txt'
    print(f'* Saving avalanche count to \'{filepath}\'')
    N = len(sim.data)
    X = np.array([[t*(not print('\b'*16, '\t', n+1, ' / ', N, end='', flush=True)),
        sim.data[t, 'a']] for n,t in enumerate(sim.data)])
    print()
    with open(filepath, 'w') as f:
        f.write('# time[s]\tavalancheCount[s-1]\n')
        np.savetxt(f, X)

    if avalanche:
        return

    # avalanche count
    print('* Plotting avalanche count')
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(*X.T)
    fig.savefig(f'{out}avalanche.pdf')

    # dynamic fields
    print('* Exporting dynamic fields')
    l = np.max(sim.data['p0'])
    title  = r'\textbf[{}]'.format(particle.capitalize()) + r'\\'
    title += r' $t = {tf} \quad \mathrm[sum] = {sum} \\'
    title += r'\mathrm[min] = {min} \\ \mathrm[max] = {max}$'
    sim.export_dynamic_field_img(
            prefix=out, 
            log=False,
            colorbar=True,
            clim=[-l, l],
            title=title, 
            background=None)


if __name__ == '__main__':

    # read arguments
    options,input_prefix = read_argv()
    
    # load data
    print('* Loading fields')
    data,lim,XYZ = load_data(input_prefix, options['res'], crop=options['ROI'])


    for particle in ('electron', 'hole'):

        print('{:^80s}'.format('## ' + particle.upper() + ' ##'))

        # simulation
        print('* Inititializing simulator')
        sim = prepare_simulation(options=options, data=data, XYZ=XYZ,
                particle=particle)
        t_goal,t_sampling = get_timing(options=options, data=data,
                particle=particle)
        print_simulation_info(sim, data, t_goal, t_sampling, particle)

        # starting simulation
        print('* Starting simulation')
        sim.run(t_goal, t_s=t_sampling)

        # plots
        if options['plot-static']:
            plot_static(sim, out=options['output'] + f'{particle}_')
        if options['plot-dynamic']:
            plot_dynamic(sim, out=options['output'] + f'{particle}_',
                    avalanche=options['avalanche'])


    print('* DONE *')

