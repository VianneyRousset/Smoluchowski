#!/usr/bin/env python3

from scipy.integrate import ode
from scipy.ndimage.filters import laplace


def perlin_noise(shape, blur):
    from scipy.ndimage import gaussian_filter
    from numpy.random import random
    return gaussian_filter(random(size=shape), sigma=blur)


def single_dot(shape):
    from numpy import zeros
    v = zeros(shape)
    v[int(shape[0]/2), int(shape[1]/2)] = 1
    return v
    
def XY(xlim, ylim, shape):
    from numpy import linspace, meshgrid
    return meshgrid(linspace(*xlim, shape[0]), linspace(*ylim, shape[1]))
