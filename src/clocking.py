#!/usr/bin/env python
# encoding: utf-8

'''
Channel for time evolved clocking fields
'''

__author__      = 'Jake Retallick'
__copyright__   = 'Apache License 2.0'
__version__     = '1.2'
__date__        = '2019-03-08'  # last update

import numpy as np
from scipy.interpolate import interp1d

from channel import Channel


class Clock(Channel):
    ''' '''

    name = 'clock'
    scale = 1.0

    active = False  # clocking fields are surface independent
    sdflag = False  # no hopping to/from clocking electrodes

    length = 2e3        # clocking signal spacing period, angstroms
    freq = 1e-1         # clock frequency, Hz

    flat = False

    # default waveform parameters
    wf_A    = 1     # waveform amplification factor
    wf_0    = 0.    # waveform offset, eV

    # time stepping
    dp = 1e-2           # fraction of clocking period between samples

    def __init__(self, fname=None, *args, **kwargs):
        '''Initialise clock'''
        super(Clock, self).__init__()

        self.t = 0.         # internal clock time
        self.fname = fname  # placeholder for externally defined clocking fields

    # inherited methods

    def setup(self, X, Y, kt):
        super(Clock, self).setup(X, Y, kt)

        self.fgen0 = self._prepareFields()
        self.bias = self.fgen(self.t)

    def tick(self):
        ''' '''
        return self.dp/self.freq

    def run(self, dt):
        '''Advance the clocking fields by the given amount'''
        if self.enabled:
            super(Clock, self).run(dt)
            self.t += dt
            self.bias = self.fgen(self.t)

    def fgen(self, t):
        '''Get the clocking potential at each DB for the given time'''
        phase = 2*np.pi*(t*self.freq % 1)
        return self.wf_0 + self.wf_A*self.fgen0(phase)

    def rates(self):
        return np.zeros(len(self.occ), dtype=float)

    def biases(self, occ):
        return self.bias if self.enabled else np.zeros(len(self.occ))

    def update(self, occ, nocc, beff):
        self.occ, self.nocc = occ, nocc


    # internal methods

    def _prepareFields(self):
        '''Precompute generator for time dependent fields'''
        import json

        if self.fname is None:
            return lambda phase: self.waveform(self.X, phase)

        with open(self.fname, 'r') as fp:
            data = json.load(fp)
        
        assert 'pots' in data, 'JSON file missing "pots" keyword'
        
        if 'phases' in data:
            phases = np.array(data['phases']).reshape(-1,)
        else:
            phases = np.linspace(0, 2*np.pi, len(data['pots']))

        pots = np.array(data['pots'])
        if len(pots)>1:
            kind = 'quadratic' if len(pots)>2 else 'linear'
            return interp1d(phases, pots.T, kind=kind)
        else:
            return lambda x: np.asarray(pots)

    def waveform(self, x, ph):
        '''Travelling wave approximation of clocking fields'''
        xx = x/self.length if not self.flat else 0
        return np.sin(2*np.pi*xx - ph)

    def _sinus(self, x):
        '''periodic function bounded by -1 and 1 with a period of 2*pi'''
        b = 0
        return np.sqrt((1+b*b)/(1+(b*np.sin(x))**2))*np.sin(x)
