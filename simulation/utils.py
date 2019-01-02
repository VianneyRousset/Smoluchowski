#!/usr/bin/env python3.6

def spread(v, shape, mode='const'):
    from numpy import ones, isscalar
    if mode != 'const':
        raise ValueError(f'Mode \'{mode}\' not implemented yet')
    if isscalar(v):
        return v*ones(shape)
    elif v.shape != shape:
        raise ValueError(f'v {v.shape} need to be a scalar or to have the same shape as \
                shape {shape} has to ') 
    return v


def meshgrid_from_lim(shape, *args):
    from numpy import linspace, meshgrid
    return meshgrid(*[linspace(*lim,s) for lim,s in zip(args, shape)])


def divergence(v, d):
    from numpy import ufunc, add, gradient, atleast_2d
    dim = len(v)
    if dim == 1:
        return gradient(v[0], *d)
    return ufunc.reduce(add, [atleast_2d(gradient(v[i], d[i], axis=i))
        for i in range(dim)])

def perlin_noise(shape, blur):
    from scipy.ndimage import gaussian_filter
    from numpy.random import random
    return gaussian_filter(random(size=shape), sigma=blur)


def load_dat(path, dataset, validity, header_limit=1000):
    import re
    pattern = r'^\s*Dataset\s*\("' + str(dataset) + r'"\)\s*\{\s*$'
    pattern = pattern + '[^{]{0,' + str(header_limit) + '}'
    pattern = pattern + r'^\s*validity\s*=\s*\[\s*"' + validity + r'"\s*\]\s*$'
    pattern = pattern + '[^{]{0,' + str(header_limit) + '}'
    pattern = pattern + r'^\s*Values\s*\((\d*)\)\s*\{'
    pattern = pattern + r'([^}]+)'
    res = re.findall(pattern, open(path).read(), flags=re.MULTILINE|re.DOTALL)
    if len(res) == 0:
        raise ValueError(f'No occurence of dataset \'{dataset}\ with validity '
                '\'{validity}\' in \'{path}\'')
    if len(res) > 1:
        raise ValueError(f'Multiple occurence of dataset \'{dataset}\ with '
                'validity \'{validity}\' in \'{path}\'')
    res = res[0]
    n = int(res[0])
    v = [float(i) for i in res[1].split()]
    if n != len(v):
        raise ValueError(f'Inconsistant data in \'{path}\'')
    return v


def load_grd(path, region, header_limit=100):
    import re
    from numpy import array
    
    # vertices
    pattern = r'^\s*Vertices\s*\((\d*)\)\s*\{\s*$'
    pattern = pattern + r'([^}]+)'
    res = re.findall(pattern, open(path).read(), flags=re.MULTILINE|re.DOTALL)
    if len(res) == 0:
        raise ValueError(f'No set of vertices found in \'{path}\'')
    if len(res) > 1:
        raise ValueError(f'Multiple sets of vertices found in \'{path}\'')
    res = res[0]
    n = int(res[0]) 
    dim = len(res[1].strip().split('\n')[0].split())
    vertices = array([float(i) for i in res[1].split()]).reshape(-1, dim)
    if n != len(vertices):
        raise ValueError(f'Inconsistant vertices data in \'{path}\'')

    # indices
    pattern = r'^\s*Region\s*\(\s*"' + str(region) + '"\s*\)\s*\{\s*$'
    pattern = pattern + '[^{]{0,' + str(header_limit) + '}'
    pattern = pattern + r'^\s*Elements\s*\((\d*)\)\s*\{'
    pattern = pattern + r'([^}]+)'
    res = re.findall(pattern, open(path).read(), flags=re.MULTILINE|re.DOTALL)
    if len(res) == 0:
        raise ValueError(f'No region called \'{region}\' found in \'{path}\'')
    if len(res) > 1:
        raise ValueError(f'Multiple regions called \'{region}\' found in '
                '\'{path}\'')
    res = res[0]
    n = int(res[0])
    indices = array([int(i) for i in res[1].split()])
    if n != len(indices):
        raise ValueError(f'Inconsistant indices data in \'{path}\'')

    return vertices


def rasterized_region(points, values, resolution, crop=None, method='linear'):
    from scipy.interpolate import griddata
    from numpy import meshgrid, linspace, diff
    dim = len(points)
    if crop is None:
        crop = [(min(i), max(i)) for i in points]
    XYZ = meshgrid(*[linspace(*l, abs(diff(l))/resolution) for l in crop])
    return crop, XYZ, griddata(points, values, tuple(XYZ), method='linear')


def single_dot(pos, *args):
    from numpy import argmin, abs, ones_like, min, max
    out = ones_like(args[0])
    d = get_dx(*args)
    for x,p,dx in zip(args, pos, d):
        if p < min(x) or p > max(x):
            raise ValueError(f'Position {pos} out of boundaries')
        v = x.reshape(-1)[argmin(abs(x-p))]
        out[x!=v] = 0
        out = out / dx
    return out


def gaussian_dot(pos, radius, *args):
    from numpy import array, pi, exp, power, product, sum
    from numpy.linalg import norm
    r = norm(array([x-p for x,p in zip(pos, args)]), axis=0)
    p = exp(-0.5 * power(r / radius, 2))
    return p / sum(p) /product(get_dx(*args))


def get_dim(fields):
    return sum([x in fields for x in 'XYZ'])


def get_XYZ(fields):
    return [fields[x] for x in 'XYZ'[:get_dim(fields)]]


def get_lim(*args):
    from numpy import min, max
    return [(min(x), max(x)) for x in args]


def get_dx(*args):
    lim = get_lim(*args)
    shape = args[0].shape
    o = [1,0,2] if len(args) > 1 else [0,1,2]
    return tuple([(lmax - lmin) / (shape[s] - 1) for (lmin,lmax),s in zip(lim, o)])


def printerr(*args, **kwargs):
    from sys import stderr
    print(*args, file=stderr, **kwargs)


def pad(v, width, mode='constant'):
    from numpy import pad, linspace

    def padding_xyz(v, width, i, kwargs):
        a,b = width[0], width[1]
        dx, = get_dx(v[a:-b])
        v[:a] = linspace(v[a] - dx*a, v[a]-dx, a)
        v[-b:] = linspace(v[-b-1]+dx, v[-b-1]+dx*b, b)
        return v

    if mode == 'xyz':
        mode = padding_xyz

    return pad(v, width, mode)

