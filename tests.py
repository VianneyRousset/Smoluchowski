#!/usr/bin/env python3.6

from simulation import Simulation
from simulation.utils import meshgrid_from_lim, gaussian_dot
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt


def test_gaussian(dim, mu, D, Ex):
	# geom
	shape   = (101,)*dim
	size    = [-30, 30]
	XYZ     = meshgrid_from_lim(shape, *[size]*dim)
	pos     = (0,)*dim
	radius  = 5

	# time
	t   = 10
	dt  = 1

	V   = Ex*XYZ[0]

	# error
	err_max = 1e-3

	# running sim and check
	sim = sim_gaussian(
			shape    = shape,
			pos      = pos,
			radius   = radius,
			dt       = dt,
			t        = t,
			D        = D,
			mu       = mu,
			V        = V,
			size     = size,
			sampling = 'last')

	t = nearest_t(sim, t)
	result = sim.data[t, 'p']
	reference = general_solution(XYZ=XYZ, mu=mu, D=D, Ex=Ex, t=t, r0=radius)

	err = max_deviation(result, ref=reference)
	if not print_check_result(err, err_max):
		if dim == 1:
			plt.close()
			plt.plot(XYZ[0], result)
			plt.plot(XYZ[0], reference)
			plt.savefig('plot.png', facecolor='white')
		if dim == 2:
			plt.close()
			plt.imshow(reference)
			plt.colorbar()
			plt.savefig('reference.png', facecolor='white')
			plt.close()
			plt.imshow(result)
			plt.colorbar()
			plt.savefig('result.png', facecolor='white')
	return err


def sim_gaussian(shape, pos, radius, dt, t, D, mu, V, size=[-10, 10], sampling=1):
	dim = len(shape)
	XYZ = meshgrid_from_lim(shape, *[size]*dim)
	p0 = gaussian_dot(pos, radius, *XYZ)
	sim = Simulation()
	XYZ = {k:v for k,v in zip(['X', 'Y', 'Z'], XYZ)}
	sim.init(shape=shape, dt=dt, p0=p0, D=D, V=V, mu=mu, **XYZ)
	sim.run(t, sampling=sampling)
	return sim


def nearest_t(sim, t):
    times = np.array([i for i in sim.data])
    return times[np.argmin(np.abs(times-t))]


def max_deviation(v, ref=None, axis=0):
    v = np.array(v)
    if ref is None:
        ref = np.mean(v, axis=axis)
    return np.max(np.abs(v - ref))


def print_check_result(err, err_max):
	print('\033[1m', end='') # bold
	if err < err_max:
		print('\033[32m', end='') # green background color
		print('\t-> OK ', end='')
		print('\033[2m', end='') # un-bold
		print('({:.2e} < {:.2e})'.format(err, err_max), end='')
		print('\033[0m')
		return True 
	else:
		print('\033[31m', end='') # red background color
		print('\t-> FAIL ', end='')
		print('\033[2m', end='') # un-bold
		print('({:.2e} > {:.2e})'.format(err, err_max), end='')
		print('\033[0m')
		return False


def general_solution(XYZ, mu, D, Ex, t, r0=0):
	pos = [Ex*mu*t, 0, 0]
	r = np.sqrt(np.power(r0, 2) + 2*D*t)
	return gaussian_dot(pos, r, *XYZ)


def get_dx(*args):
    lim = [(np.min(x), np.max(x)) for x in args]
    shape = args[0].shape
    return tuple([(lmax - lmin) / (s - 1) for (lmin,lmax),s in zip(lim, shape)])

if __name__ == '__main__':

	tests = ('static', 'diffusion', 'drift', 'drift-diffusion')
	dimensions = [1,2,3]
	err = []

	if 'static' in tests:
		for dim in dimensions:
			print(f'* Testing static ({dim}D)')
			err += [test_gaussian(dim=dim, mu=1, D=0, Ex=0)]

	if 'diffusion' in tests:
		for D in [0.1, 0.5]:
			for dim in dimensions:
				print(f'* Testing diffusion ({dim}D) with D = {D:.2e}')
				err += [test_gaussian(dim=dim, mu=1, D=D, Ex=0)]

	if 'drift' in tests:
		for Ex in [0.1, 0.5, -0.5]:
			for dim in dimensions:
				print(f'* Testing drift ({dim}D) with Ex = {Ex:.2e}')
				err += [test_gaussian(dim=dim, mu=1, D=0, Ex=Ex)]

	if 'drift-diffusion' in tests:
		for mu,D,Ex in [(1.0, 0.5, 0.1), (1.0, 0.5, 0.5),
				(0.1, 0.5, -0.5), (1.0, 0.5, -0.5)]:
			for dim in dimensions:
				print(f'* Testing drift-diffusion ({dim}D) with mu = {mu:.2e}'
						f', D = {D:.2e} and Ex = {Ex:.2e}')
				err += [test_gaussian(dim=dim, mu=mu, D=D, Ex=Ex)]


	# print summary
	import sys
	print('* Error summary in percent:')
	err= np.array(err).reshape(-1, len(dimensions))
	np.savetxt(sys.stdout, 100*err.T, fmt='%.3e')
