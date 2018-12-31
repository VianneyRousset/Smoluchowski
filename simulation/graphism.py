#!/usr/bin/env python3.6

import matplotlib as mpl
from matplotlib import pyplot as plt
from cycler import cycler

rc = {
        'text.usetex'           : True,
        'text.latex.unicode'	: True,
        'font.family'	        : 'STIXGeneral',
        'mathtext.fontset'	: 'stix',
        'text.color'		: 'black',
        'axes.facecolor'	: 'white',
        'axes.grid'		: True,
        'axes.labelcolor'	: 'black',
        'axes.labelweight'	: 'bold',

        'xtick.color'		: 'black',
        'xtick.labelsize'	: 'small',
        'xtick.direction'	: 'inout',
        'xtick.minor.visible'	: True,
        'ytick.color'		: 'black',
        'ytick.labelsize'	: 'small',
        'ytick.direction'	: 'inout',
        'ytick.minor.visible'	: True,

        'grid.color'		: 'black',
        'grid.alpha'		: 0.2,

        'figure.titleweight'	: 'bold',
        'figure.facecolor'	: 'white',
        }


def save_image(path, data, colorbar=True, background=None, clim=None,
        title=None, size=(3,3), dpi=500, xlim=None, ylim=None):

    with mpl.rc_context(rc=rc):
        plt.ioff()
        fig = plt.figure()

        extent = None
        if xlim and ylim:
            extent = [*xlim, *ylim]

        if background is None:
            plt.imshow(data, clim=clim, extent=extent)
        else:
            plt.imshow(background, cmap='Greys', extent=extent)
            plt.imshow(colormap(data, clim=clim), extent=extent)

        if title:
            plt.title(title)

        if colorbar:
            plt.colorbar()

        fig.set_size_inches(size)
        fig.savefig(path, dpi=dpi, facecolor='white')
        plt.close()


def value_unit(xmin):
    from numpy import log10, floor, ceil
    prefixes = {
            -24: r'\yocto', -21: r'\zepto', -18: r'\atto',  -15: r'\femto',
            -12: r'\pico',   -9: r'\nano',   -6: r'\micro',  -3: r'\milli',
              0: r'',         3: r'\kilo',    6: r'\mega',    9: r'\giga',
             12: r'\tera',   15: r'\peta',   18: r'\exa',    21: r'\zetta',
             24: r'\yotta'
             }

    o = log10(xmin)/3
    o = floor(o) if o > 0 else -ceil(-o)
    o = int(o)*3
    return 10**o, prefixes[o]


def si_value(value, unit):
    return r'\SI{' + f'{value}' + r'}{' + f'{unit}' + r'}'


def colormap(v, clim=None):
    from numpy import min, max, ones_like, zeros_like, array, transpose
    if clim is None:
        clim = [min(v), max(v)]

    if clim[0] == clim[1]:
        v = zeros_like(v)
    else:
        v = (v - clim[0]) / (clim[1] - clim[0])
    I = ones_like(v)
    O = zeros_like(v)
    v = array([I,O,O,v])
    return transpose(v, axes=(1,2,0))
