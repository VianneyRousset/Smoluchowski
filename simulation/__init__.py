#!/usr/bin/env python3.6

from .progressBar import ProgressBar

__all__ = ['utils']

class Simulation:

    def __init__(self, filename=None, quiet=False, avalanche_only=False,
            particle='electron'):
        from .data import SimData
        self.quiet = quiet
        self.data = SimData(filename, mode='a')
        self.avalanche_only = False
        self.particle = particle


    def init(self, p0, X, **kwargs):

        from .utils import get_dim, get_XYZ, get_lim, get_dx
        from numpy import isscalar, atleast_1d, gradient

        padding = self._convert_padding_width(kwargs, kwargs.get('padding'))

        # static
        static = {
                'X': X,
                'Y': kwargs.get('Y'),
                'Z': kwargs.get('Z'),
                'D': kwargs.get('D', 0),
                'mu': kwargs.get('mu', 0),
                'V': kwargs.get('V', 0),
                'a_e': kwargs.get('a_e', 0),
                'a_h': kwargs.get('a_h', 0),
                'a_region': kwargs.get('a_region', 0),
                'p0': p0,
                }

        dim = get_dim(static)
        XYZ = [static[x] for x in 'XYZ'[:dim]]

        ## static scalars 
        static_scalars = {k:v for k,v in static.items() if v is not None and isscalar(v)}
        self.data.init_static_scalars(static_scalars)

        ## static fields
        static_fields = {k:v for k,v in static.items() if v is not None and not isscalar(v)}
        static_fields = self._add_padding(static_fields, padding)
        shape = atleast_1d(static_fields['X']).shape
        self.data.init_static_fields(shape, static_fields)

        # dynamic
        dynamic = {'a': 0}
        if not self.avalanche_only:
            dynamic['p'] = p0

        ## dynamic scalars and fields
        dynamic_scalars = {k:v for k,v in dynamic.items() if v is not None and isscalar(v)}
        dynamic_fields = {k:v for k,v in dynamic.items() if v is not None and not isscalar(v)}
        dynamic_fields = self._add_padding(dynamic_fields, padding)
        self.data.init_dynamic_scalars_and_fields(shape, dynamic_scalars, dynamic_fields)

        # usefull variables
        self.a = 0
        self.shape = atleast_1d(static_fields['X']).shape
        self.lim = get_lim(*self.XYZ)
        self.d = get_dx(*self.XYZ)
        if isscalar(self.data['V']):
            self.E = [0]*self.dim
        else:
            self.E = gradient(-self.data['V'], *self.d)
            if self.dim == 1:
                self.E = [self.E]

        # avalanche probability 
#        self.P_e, self.P_h = self._compute_avalanche_probability()


    def reset(self):
        for i in f.root:
            f.remove_node(i)


    def run(self, t_goal, t_s=None):

        from .core import smoluchowski 
        from scipy.integrate import solve_ivp
        from numpy import inf 
        import time

        start_time = time.time()
        self.t_s = t_s

        if not self.quiet:
            print('* Setting up simulation')
        
        # setting up function
        D = self.data['D']
        mu = self.data['mu']
        E = self.E
        a_region = self.data['a_region']
        XYZ = self.XYZ
        d = self.d
        shape = self.shape
        progress = ProgressBar()
        self._t = 0
        self._t_last_sample = 0
        self._p = None
        self._a = 0 
        if t_s is None or type(t_s) is str and t_s == 'last':
            max_step = inf
        else:
            max_step = t_s

        def fun(t, p):

            # core
            p = p.reshape(shape)
            a,dp = smoluchowski(t, p, D=D, mu=mu, E=E, a_region=a_region, d=d,
                    charge_sign={'electron':-1, 'hole':1}[self.particle])
            self._a += a

            # progress bar
            if not self.quiet:
                progress.set_ratio(t / t_goal)

            # recording
            if t_s is not None and type(t_s) is not str and t > self._t_last_sample + t_s:
                dt = t - self._t_last_sample
                if self.avalanche_only:
                    self.data.snapshot(t, a=self._a/dt)
                else:
                    self.data.snapshot(t, p=p, a=a/dt)
                self._t = t
            elif t_s == 'last': 
                self._t = t
                self._p = p

            return dp.reshape(-1)

        # starting simulation
        if not self.quiet:
            print('* Starting simulation (goal: {:.2e}s)'.format(t_goal))
        solve_ivp(fun, t_span=[0, t_goal], y0=self.data['p0'].reshape(-1),
                max_step=max_step, vectorized=True)
        progress.end()

        # recording last
        if t_s == 'last':
            if self.avalanche_only:
                self.data.snapshot(t, a=self._a)
            else:
                self.data.snapshot(t, p=self._p, a=self._a)

        if not self.quiet:
            print('* Done after {:.2f}s with {} snapshots.'.format(
                time.time() - start_time, len(self.data)))

    
    def export_static_field_img(self, field, path, log=False, title=None,
            colorbar=True, log_min_value=1e-20):

        from numpy import isscalar
        from .graphism import save_image, value_unit, si_value

        if field == 'E':
            from scipy.linalg import norm
            data = norm(self.E, axis=0)
        else:
            data = self.data[field]

        if isscalar(data):
            return print(f'\'{field}\' is not a field but a scalar = {data}')
        xlim,ylim = (self.lim[0], (self.lim[1][1], self.lim[1][0]))
        save_image(path=path, data=data, title=title,
                xlim=xlim, ylim=ylim, colorbar=colorbar)


    def export_dynamic_field_img(self, prefix='', log=False, title='$t = {tf}$',
            colorbar=False, background=None, clim='auto', log_min_value=1e-20):

        from .graphism import save_image, value_unit, si_value
        from numpy import min, max, abs, log10, sum, product
        from string import Formatter

        N = len(self.data)

        # field extrema
        if type(clim) is str and clim == 'auto':
            if not self.quiet:
                print('* Looking for field extrema')
            clim = self._get_clim(log=log)
            if not self.quiet:
                print('* Done')
        if log and clim:
            clim = [max([l,log_min_value]) for l in clim]
            clim = log10(clim)
        
        # time unit 
        tmin = self.t_s
        if N > 1:
            t_order,t_prefix = value_unit(tmin)

        # gen images
        if not self.quiet:
            print('* Generating images')
        progress = ProgressBar()
        for n,t in enumerate(self.data):

            # progress bar
            progress.set_ratio(n / (N - 1))

            # usefull variables
            data = self.data[t, 'p']
            i = int(round(t/t_order))
            path = f'{prefix}{i:09d}.png'
            title_values = {'t': '{:e}'.format(t), 'N': N, 'i': i, 'path': path,
                'tf': si_value(round(t/t_order), t_prefix + r'\second')}

            # compute integration, min and max if needed
            if 'sum' in {v[1] for v in Formatter().parse(title)}:
                title_values['sum'] = sum(data) * product(self.d)
                title_values['sum'] = '{: .2e}'.format(title_values['sum'])
                title_values['sum'] = si_value(title_values['sum'], 'part.')
            if 'min' in {v[1] for v in Formatter().parse(title)}:
                title_values['min'] = min(data)
                title_values['min'] = '{: .2e}'.format(title_values['min'])
                title_values['min'] = si_value(title_values['min'], r'\per\micro\meter')
            if 'max' in {v[1] for v in Formatter().parse(title)}:
                title_values['max'] = max(data)
                title_values['max'] = '{: .2e}'.format(title_values['max'])
                title_values['max'] = si_value(title_values['max'], r'\per\micro\meter')


            # log
            if log:
                data[data < log_min_value] = log_min_value 
                data = log10(data)

            # format title
            ftitle = ''
            if title:
                ftitle = title.format(**title_values)
                ftitle = ftitle.replace('[', '{').replace(']', '}')

            # export image
            xlim,ylim = (self.lim[0], (self.lim[1][1], self.lim[1][0]))
            save_image(path=path, data=data, background=background,
                    colorbar=colorbar, clim=clim, title=ftitle,
                    xlim=xlim, ylim=ylim)

        progress.end()

        if not self.quiet:
            print('* Done')


    @property
    def dim(self):
        from .utils import get_dim
        return get_dim(self.data.static_fields)


    @property
    def XYZ(self):
        return [self.data[x] for x in 'XYZ'[:self.dim]]


    def _get_clim(self, log=False):
        from numpy import min, max, abs
        N = len(self.data)
        progress = ProgressBar()
        minimum,maximum = float('inf'),-float('inf')
        for n,t in enumerate(self.data):
            if not self.quiet:
                progress.set_ratio(n / (N - 1))
            p = self.data[t, 'p']
            if log:
                p = abs(p)
            minimum = min([minimum, min(p)])
            maximum = max([maximum, max(p)])

        progress.end()
        return [minimum, maximum]


    def _convert_padding_width(self, fields, padding):
        from .utils import get_dim, get_dx
        from numpy import atleast_2d, isscalar, ones, round 
        if padding is None:
            return None
        dim = get_dim(fields)
        XYZ = [fields[x] for x in 'XYZ'[:dim]]
        d = atleast_2d(get_dx(*XYZ))

        if isscalar(padding):
            padding = ones((len(d), 2))
        padding = atleast_2d(padding)
        return round(padding/d.T).astype(int)


    def _add_padding(self, fields, padding):
        from .utils import pad

        modes = {
                'X': 'xyz',
                'Y': 'xyz',
                'Z': 'xyz',
                'D': 'edge',
                'g': 'edge',
                'V': 'edge',
                'p': 'constant',
                'mu': 'edge',
                'a_e': 'constant',
                'a_h': 'constant',
                'a_region': 'constant'
                }

        if padding is None:
            return fields

        return {k:pad(fields[k], width=padding, mode=modes[k]) for k in fields}

    
    def _compute_avalanche_probability(self):
        from .core import avalanche_probability
        from .utils import get_dim
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d
        from numpy import zeros, ones, array

        dim = self.dim 
        XYZ = self.XYZ
        a_e = self.data['a_e']
        a_h = self.data['a_h']
        if dim == 1:
            x = XYZ[0]
            axis = 0
            shape = 1
            shape = a_e.shape[1:]
        elif dim == 2:
            x = XYZ[0][0]
            axis = 1
            shape = a_e.shape[1:]
        elif dim == 3:
            x = XYZ[2][0][0]
            axis = 2
            shape = a_e.shape[0:2]


        a_e = interp1d(x, a_e, axis=axis)
        a_h = interp1d(x, a_h, axis=axis)

        fun = lambda x,P: avalanche_probability(x, P, a_e, a_h) 
        x_span = (min(x), max(x))
        P0 = array([ones(shape, dtype=complex), zeros(shape)])


        P = solve_ivp(fun=fun, t_span=x_span, y0=P0, t_eval=x)
        print(p.shape)
        return P
