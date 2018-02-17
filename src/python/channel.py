#!/usr/bin/env python
# encoding: utf-8

'''
Channels for charges hopping on and off the surface
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-02-16'  # last update

import numpy as np


class Channel(object):
    '''Virtual base class for all channels of charge on and off the surface'''

    # machine precision
    MTR     = 1e-16     # minimum tickrate, avoid zero division

    # useful lambdas
    rebirth = np.random.exponential # reset for hop-on lifetime

    def __init__(self, kt):
        '''Initialise a Channel instance with the given thermal energy'''
        self.kt = kt

    def setup(self, X, Y):
        '''Set up channel for the given DB positions'''
        self.X, self.Y = X, Y
        self.lifetime = self.rebirth()
        self.tickrate = 1.

    def peek(self):
        '''Time until next charge hops onto the surface'''

        return self.lifetime/(self.tickrate+self.MTR)

    def run(self, dt):
        '''Advance the channel lifetime by the given amount'''
        self.lifetime -= dt*self.tickrate

    def pop(self):
        '''Pop a charge off the channel. The HoppingModel should update its
        charge state and immediately pass new occ/nocc back'''

        P = self.offrates
        ind = np.random.choice(self.nocc, p=P/P.sum())
        self.lifetime = self.rebirth()
        return ind

    def rates(self):
        '''Get the hopping rates onto the channel for the given occupied DBs'''
        return self.onrates

    def update(self, occ, nocc, beff):
        '''Tell the channel which DBs are occupied and what the local levels
        of each DB are.

        inputs:
            occ     : list of occupied DBs
            nocc    : list of unoccupied DBs
            beff    : array of local levels of each DB
        '''
        raise NotImplementedError()

class Bulk(Channel):
    '''Passive charge transfer between the surface and the bulk'''

    # prefactors
    nu_on   = 1.    # maximum rate of hops onto the bulk
    nu_off  = 1.    # maximum rate of hops from the bulk

    # energy offsets
    mu_on   = .2    # local energy at which electrons start hopping onto Bulk
    mu_off  = .2    # local energy at which electrons start hopping from Bulk

    alpha   = 1.e3  # damping factor for kt, higher means sharper transition

    # inherited methods

    def init(self, kt):
        super(Bulk, self).__init__(kt)
        self.kt /= self.alpha

    def update(self, occ, nocc, beff):
        self.occ, self.nocc = occ, nocc
        self.offrates = self._compute_offrates(beff)
        self.onrates = self._compute_onrates(beff)
        self.tickrate = self.offrates.sum()

    # channel specific methods

    def _compute_onrates(self, beff):
        '''Compute the hopping rates from occupied DBs to the Bulk'''
        return self.nu_on/(1.+np.exp((beff[self.occ]+self.mu_on)/self.kt))

    def _compute_offrates(self, beff):
        '''Compute the hopping rates from the Bulk onto unoccupied DBs'''
        return self.nu_off/(1.+np.exp(-(beff[self.nocc]+self.mu_off)/self.kt))
