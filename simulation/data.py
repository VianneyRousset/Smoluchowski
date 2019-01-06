#!/usr/bin/env python3.6

from tables import IsDescription, Float32Col, Float64Col, open_file
from tempfile import mktemp
from os.path import isfile


class SimData:

    def __init__(self, path=None, mode='r'):
        if not path:
            path = mktemp()
            mode = 'a'

        # open HDF5 file
        self.f = open_file(path, mode=mode,
                title="Smoluchowski drift-diffusion simulation data")
        try:
            self.static_scalars = self.f.root.static_scalars
            self.static_fields = self.f.root.static_fields
            self.dynamic_scalar = self.f.root.dynamic_scalar
            self.dynamic_fields = self.f.root.dynamic_fields
        except:
            pass


    def init_static_scalars(self, scalars):

        # create tab
        self.static_scalars = set(scalars)
        if not self.static_scalars:
            return
        cols_desc = {c: Float64Col() for c in scalars}
        self.tab_static_scalars = self.f.create_table(self.f.root,
                'static_scalars', cols_desc, 'Static scalars')
        self.tab_static_scalars.row.append()
        self.tab_static_scalars.flush()

        # save values
        for k,v in scalars.items():
            self[k] = v


    def init_static_fields(self, shape, fields):

        # create tab
        self.static_fields = set(fields)
        cols_desc = {f: Float64Col(shape) for f in fields}
        self.tab_static_fields = self.f.create_table(self.f.root,
                'static_fields', cols_desc, 'Static fields')
        self.tab_static_fields.row.append()
        self.tab_static_fields.flush()

        # save values
        for k,v in fields.items():
            self[k] = v


    def init_dynamic_scalars_and_fields(self, shape, scalars, fields):

        # create scalars tab
        self.dynamic_scalars = set(scalars)
        cols_desc = {c: Float64Col() for c in scalars}
        cols_desc['t'] = Float32Col()
        self.tab_dynamic_scalars = self.f.create_table(self.f.root,
                'dynamic_scalars', cols_desc, 'Dynamic scalars')
        self.tab_dynamic_scalars.row.append()
        self.tab_dynamic_scalars.flush()

        # create fields tab
        self.dynamic_fields = set(fields)
        cols_desc = {f: Float64Col(shape) for f in fields}
        cols_desc['t'] = Float32Col()
        self.tab_dynamic_fields = self.f.create_table(self.f.root,
                'dynamic_fields', cols_desc, 'Dynamic fields')

        # snapshot
        self.snapshot(t=0.0, **scalars, **fields)


    # SimData['D'] = D, to set static scalar or field 'D' 
    # SimData[42.0] = {'p': p} to create a snapshot of all dynamic scalars and
    # fields at t = 42.0
    def __setitem__(self, k, v):
        if type(k) is str:
            if k in self.static_scalars:
                self.set_static_scalar(k, v)
            else:
                self.set_static_field(k, v)
        else:
            self.snapshot(k, **v)


    # SimData['D'] to get static scalar of field 'D' 
    # SimData[42.0] to get the snapshot at t = 42.0
    # SimData[42.0,'D'] to get scalar or field 'D' at t = 42.0
    def __getitem__(self, k):
        if type(k) is str:
            if k in self.static_scalars:
                return self.tab_static_scalars[0][k]
            else:
                return self.tab_static_fields[0][k]
        elif type(k) is tuple:
            i = [t for t in self].index(k[0])
            if k[1] in self.dynamic_scalars:
                return self.tab_dynamic_scalars.cols.__getattribute__(k[1])[i]
            else:
                return self.tab_dynamic_fields.cols.__getattribute__(k[1])[i]
        else:
            i = [t for t in self].index(k[0])
            row = self.dynamic_scalars[[t for t in self].index(k)]
            snapshot = {f: row[f] for f in self.dynamic_fields}
            row = self.dynamic_fields[[t for t in self].index(k)]
            snapshot.update({f: row[f] for f in self.dynamic_fields})
            return snapshot


    def set_static_scalar(self, k, v):
        if k not in self.static_scalars:
            raise KeyError(str(f))
        self.tab_static_scalars.cols.__getattribute__(k)[0] = v
        self.tab_static_scalars.flush()


    def set_static_field(self, k, v):
        if k not in self.static_fields:
            raise KeyError(str(f))
        self.tab_static_fields.cols.__getattribute__(k)[0] = v
        self.tab_static_fields.flush()


    def snapshot(self, t, **kwargs):
        if [r for r in self.tab_dynamic_fields.where(f't == {t}')]:
            raise ValueError(f'Snapshot at t={t} already exists')
        if set(kwargs) != self.dynamic_scalars.union(self.dynamic_fields):
            raise ValueError('Invalid or missing keyword arguments '
            f'(got {repr(set(kwargs))})')

        # setting scalars
        row = self.tab_dynamic_scalars.row
        row['t'] = t
        for k,v in {k:v for k,v in kwargs.items() if k in self.dynamic_scalars}.items():
            row[k] = v
        row.append()
        self.tab_dynamic_scalars.flush()

        # setting fields 
        row = self.tab_dynamic_fields.row
        row['t'] = t
        for k,v in {k:v for k,v in kwargs.items() if k in self.dynamic_fields}.items():
            row[k] = v
        row.append()
        self.tab_dynamic_fields.flush()


    def __iter__(self):
        return sorted(self.tab_dynamic_fields.cols.t.__iter__()).__iter__()


    def __len__(self):
        return len(self.tab_dynamic_fields)

