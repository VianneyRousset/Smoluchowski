#!/usr/bin/env python3.6

from .progressBar import ProgressBar

from scipy.integrate import ode

__all__ = ['utils']

class Simulation:

    def __init__(self, filename=None, quiet=False):
        from .data import SimData
        self.quiet = quiet
        self.data = SimData(filename, mode='a')


    def init(self, dt, padding=None, **kwargs):

        from .utils import get_dim, get_XYZ, get_lim, get_dx
        from numpy import isscalar, atleast_1d, gradient

        padding = self._convert_padding_width(kwargs, padding)

        # static
        static = {
                'dt': dt,
                'X': kwargs['X'],
                'Y': kwargs.get('Y'),
                'Z': kwargs.get('Z'),
                'D': kwargs.get('D', 0),
                'mu': kwargs.get('mu', 1),
                'V': kwargs.get('V', 0),
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
        dynamic = {
                'p': kwargs['p0'],
                'a': 0
                }

        ## dynamic scalars and fields
        dynamic_scalars = {k:v for k,v in dynamic.items() if v is not None and isscalar(v)}
        dynamic_fields = {k:v for k,v in dynamic.items() if v is not None and not isscalar(v)}
        dynamic_fields = self._add_padding(dynamic_fields, padding)
        self.data.init_dynamic_scalars_and_fields(shape, dynamic_scalars, dynamic_fields)

        # usefull variables
        self.shape = atleast_1d(static_fields['X']).shape
        self.lim = get_lim(*self.XYZ)
        self.d = get_dx(*self.XYZ)
        if isscalar(self.data['V']):
            self.E = [0]*self.dim
        else:
            self.E = gradient(-self.data['V'], *self.d)
            if self.dim == 1:
                self.E = [self.E]


    def reset(self):
        for i in f.root:
            f.remove_node(i)


    def run(self, t, sampling=1):
        import time
        start_time = time.time()
        if not self.quiet:
            print('* Setting up simulation')
        self._setup_simulation(self.data[0,'p'])
        progress = ProgressBar()

        # computation times reset
        self.tcomp_var   = 0
        self.tcomp_diff  = 0
        self.tcomp_drift = 0

        if not self.quiet:
            print('* Starting simulation')

        n = 1
        while self.r.successful() and self.r.t < t:
            if not self.quiet:
                progress.set_ratio(self.r.t/t)
            p = self.r.integrate(self.r.t + self.data['dt'])
            p = p.reshape(self.shape)
            if type(sampling) is not str and (n % sampling) == 0:
                self.data.snapshot(self.r.t, p=p, a=0)
            n = n + 1
        progress.end()

        if sampling == 'last':
            self.data.snapshot(self.r.t, p=p, a=0)

        if not self.quiet:
            print('* Done after {:.2f}s'.format(time.time() - start_time))
            print('** Variable computation time : {:.2f}s'.format(self.tcomp_var))
            print('** Diffusion computation time : {:.2f}s'.format(self.tcomp_diff))
            print('** Drift computation time : {:.2f}s'.format(self.tcomp_drift))

    
    def export_field_image(self, field, path, log=False, title=None,
            colorbar=True, log_min_value=1e-20):

        from .graphism import save_image, value_unit, si_value
        xlim,ylim = (self.lim[0], (self.lim[1][1], self.lim[1][0]))
        save_image(path=path, data=self.data[field], title=title,
                xlim=xlim, ylim=ylim, colorbar=colorbar)


    def export_images(self, output_dir, prefix='', log=False, title='$t = {tf}$',
            colorbar=True, background=None, clim=None, log_min_value=1e-20):

        from .graphism import save_image, value_unit, si_value
        from numpy import min, max, abs, log10, sum, product
        from string import Formatter

        N = len(self.data)

        # field extrema
        if not clim:
            if not self.quiet:
                print('* Looking for field extrema')
            clim = self._get_clim(log=log)
            if not self.quiet:
                print('* Done')
        if log:
            clim = [max([l,log_min_value]) for l in clim]
            clim = log10(clim)
        
        # time unit 
        tmin = self.data['dt']
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
            name = '{}{:09d}'.format(prefix, i)
            path = f'{output_dir}/{name}.png'
            title_values = {'t': '{:e}'.format(t), 'N': N, 'i': i, 'name': name, 'path': path,
                'tf': si_value(round(t/t_order), t_prefix + r'\second')}

            # compute integration if needed
            if 'sum' in {v[1] for v in Formatter().parse(title)}:
                title_values['sum'] = sum(data) * product(self.d)

            # log
            if log:
                data[data < log_min_value] = log_min_value 
                data = log10(data)

            # format title
            ftitle = ''
            if title:
                ftitle = title.format(**title_values)

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


    def _setup_simulation(self, p0):
        from .core import solvr
        self.r = ode(solvr).set_integrator('dop853')
        self.r.set_initial_value(y=p0.reshape(-1), t=0)
        self.r.set_f_params(self)


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
                'D': 'constant',
                'g': 'edge',
                'V': 'edge',
                'p': 'constant',
                'mu': 'constant'
                    }

        if padding is None:
            return fields

        return {k:pad(fields[k], width=padding, mode=modes[k]) for k in fields}

