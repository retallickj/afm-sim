#!/usr/bin/env python
# encoding: utf-8

'''
Simulator for the non-equilibrium surface dynamics of charges in QSi's DB
arrangements
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-02-22'  # last update

import numpy as np
from scipy.special import erf


from model import models as Models
from channel import Channel, channels as Channels
from tip_model import TipModel

from itertools import combinations, chain
from collections import defaultdict

from time import clock as tick
import sys

# add non-standard channels
Channels['tip'] = TipModel

class HoppingModel:
    '''Time dependent surface hopping model for charge transfer in DBs'''

    # machine precision
    MTR     = 1e-16         # for tickrates division

    # energy parameters
    debye   = 25.           # debye screening length, angstroms
    erfdb   = 2.            # erf based screening length
    eps0    = 8.854e-12     # F/m
    q0      = 1.602e-19     # C
    kb      = 8.617e-05     # eV/K
    T       = 4.0           # system temperature, K
    epsr    = 6.35          # relative permittivity

    Kc = 1e10*q0/(4*np.pi*epsr*eps0)    # Coulomb strength, eV.angstrom

    # lattice parameters
    a = 3.84    # lattice vector in x, angstroms    (intra dimer row)
    b = 7.68    # lattice vector in y, angstroms    (inter dimer row)
    c = 2.25    # dimer pair separation, angstroms

    # general settings
    fixed_pop = False    # fixed number of electrons
    free_rho = 0.5       # filling density if not fixed_pop (Nel = round(N*free_rho))
    burn_count = 0      # number of burns hops per db

    enable_cohop = False      # enable cohopping

    # useful lambdas
    rebirth = np.random.exponential     # reset for hopping lifetimes

    debye_factor = lambda self, R: np.exp(-R/self.debye)
    debye_factor = lambda self, R: erf(R/self.erfdb)*np.exp(-R/self.debye)

    coulomb = lambda self, R: (self.Kc/R)*self.debye_factor(R)

    def __init__(self, pos, model='marcus', **kwargs):
        '''Construct a HoppingModel for a DB arrangement with the given x and
        optional y coordinates in unit of the lattice vectors. For now, assume
        only the top site of each dimer pair can be a DB.

        inputs:
            pos     : Iterable of DB locations. Each elements of pos should be a
                      3-tuple (x,y,b) with x and y the dimer column and row and
                      b true if the DB is at the bottom of the dimer pair. If
                      pos[i] is an integer x, it gets mapped to (x,0,0).
            model   : Type of hopping rate model

        optional key-val arguments:
            None
        '''

        # format and store db locations and number
        self._parseX(pos)

        self.charge = np.zeros([self.N,], dtype=int)    # charges at each db

        self.bias = np.zeros([self.N,])         # bias energy at each site
        self.dbias = np.zeros([self.N,])        # temporary additional bias

        # distance matrix
        dX = self.a*(self.X-self.X.reshape(-1,1))
        dY = self.b*(self.Y-self.Y.reshape(-1,1))
        self.R = np.sqrt(dX**2+dY**2)

        # electrostatic couplings
        self.V = self.coulomb(np.eye(self.N)+self.R)
        np.fill_diagonal(self.V,0)

        # by default, use
        self.Nel = int(round(self.N*self.free_rho))

        self.vprint = print

        # create model, setup on initialisation
        if model not in Models:
            raise KeyError('Invalid model type. Choose from [{0}]'.format(
                                ', '.join(Models.keys())))
        self.model = Models[model]()
        self.channels = []
        self.initialised = False

    def fixElectronCount(self, n):
        '''Fix the number of electrons in the system. Use n<0 to re-enable
        automatic population mechanism.'''

        print('setting electron count to {0}'.format(n))
        if n<0:
            self.Nel = int(round(self.N*self.free_rho))
            self.fixed_pop = False
        else:
            assert n <= self.N, 'Invalid number of electrons'
            self.Nel = n
            self.fixed_pop = True

        if self.initialised:
            self.charge[self.state[:self.Nel]]=1
            self.charge[self.state[self.Nel:]]=0
            self.update()

    def addBiasGradient(self, F, v=[1,0]):
        '''Apply a bias gradient of strength F (eV/angstrom) along the
        direction v'''

        self.bias += F*(v[0]*self.a*self.X+v[1]*self.b*self.Y)


    def getChannel(self, name):
        '''Get a handle for the Channel with the given name. If it doesn't
        exists, return None'''

        for channel in self.channels:
            if channel.name == name:
                return channel
        return None


    def addChannel(self, channel):
        '''Add a Channel instance to the HoppingModel.

        inputs:
            channel : Channel to include. Must either be a Channel instance or
                     a string indicating an accepted channel type in Channels.
        return:
            handle for the added Channel
        '''

        if isinstance(channel, str):
            if channel not in Channels:
                raise KeyError('Invalid channel type. Choose from [{0}]'.format(
                    ', '.join(k for k in Channels if k != 'base')))
            self.channels.append(Channels[channel]())
        elif isinstance(channel, Channel):
            self.channels.append(channel)
        else:
            raise KeyError('Unrecognized Channel format: must be either a str \
                                or Channel derived class')
        return self.channels[-1]


    # FUNCTIONAL METHODS

    def initialise(self, charges=None):
        '''Initialise all necessary system parameters'''

        # surface state described by a single array with the first Nel values
        # occupied sites and the remaining N-Nel unoccupied

        if charges is None: # random
            self.state = np.random.permutation(range(self.N))
        else:
            occ, nocc = [], []
            for i,c in enumerate(charges):
                (occ if c==1 else nocc).append(i)
            Nel = len(occ)
            assert Nel == self.Nel, 'Electron count mismatch'
            assert self.N == len(occ+nocc), 'DB count mismatch'
            self.state = np.array(occ+nocc)

        self.charge[self.state[:self.Nel]]=1
        self.charge[self.state[self.Nel:]]=0

        # setup model and channels
        X, Y, kt  = self.a*self.X, self.b*self.Y, self.kb*self.T
        self.model.setup(X, Y, kt)
        for channel in self.channels:
            channel.setup(X, Y, kt)

        self.update()

        # lifetimes[i] is the lifetime of an electron at the i^th DB
        self.lifetimes = np.array([self.rebirth() for _ in range(self.N)])

        # cohopping setup, cohop_lt[ij] is the lifetime of the electrons at DBs i,j
        if self.enable_cohop:
            self.cohop_lt = {ij: self.rebirth() for ij in self.cohop_tickrates}

        self.energy = self.computeEnergy()

        self.initialised = True

        if charges is None:
            # burn off random initial state
            self.burn(self.burn_count*self.N)


    def computeBeff(self, occ, wch=True):
        '''Compute the effective bias at each site for the given occupation.
        Optional include channel contributions'''

        beff = self.bias - np.sum(self.V[:,occ], axis=1)
        if wch:
            beff += sum(ch.scale*ch.biases(occ) for ch in self.channels)
        return beff

    def update(self):
        '''Update the energy deltas, tunneling rates, and tick rates'''

        occ, nocc = self.state[:self.Nel], self.state[self.Nel:]

        #t = tick()

        # TODO: include channel contributions to beff
        # effective bias at each location
        beff = self.bias-np.dot(self.V, self.charge)

        # add channel biases to beff
        beff += sum(ch.scale*ch.biases(occ) for ch in self.channels)
        for channel in self.channels:
            channel.update(occ, nocc, beff)

        # energy deltas
        NEW = True
        self.dG = np.zeros([self.Nel, self.N-self.Nel], dtype=float)
        for n, src in enumerate(occ):
            tbeff = self.computeBeff(np.delete(occ, n), wch=not NEW)
            self.dG[n,:] = tbeff[src] - tbeff[nocc]
        if NEW:
            self.dG += sum(ch.scale*ch.computeDeltas() for ch in self.channels)

        self.beff = beff    # store for dE on channel hop
        self.trates = self.model.rates(self.dG, occ, nocc)
        self.tickrates = np.sum(self.trates, axis=1)
        if self.channels and not self.fixed_pop:
            self.crates = np.array([channel.rates() for channel in self.channels]).T
            self.tickrates += np.sum(self.crates, axis=1)

        #t1, t = tick()-t, tick()

        # cohopping
        if self.enable_cohop:
            self.cohop_rates = defaultdict(dict)        # hopping rate for each cohop
            self.cohop_tickrates = defaultdict(dict)    # tickrate per electron pair
            self.cohop_dG = defaultdict(dict)           # energy delta for each cohop
            for ij in combinations(range(self.Nel), 2):
                for kl in combinations(range(self.N-self.Nel), 2):
                    (i,j),(k,l) = ij, kl
                    sites = self.state[[i,j, self.Nel+k, self.Nel+l]]
                    dG = self._compute_cohop_dG(i,j,k,l, *sites)
                    self.cohop_rates[ij][kl] = self.model.cohopping_rate(dG,*sites)
                    self.cohop_dG[ij][kl] = dG
                self.cohop_tickrates[ij] = sum(self.cohop_rates[ij].values())

        #self.vprint('update :: hopping = {0:.3e} <::> cohop = {1:.3e}'.format(t1, tick()-t))


    def peek(self):
        '''Return the time before the next tunneling event and the index of the
        event. Does not account for time evolution of channel bias contributions

        output:
            dt  : time to next hopping event, s
            ind : index of event: if ind < self.Nel the electron at state[ind]
                 hops, otherwise an electron hops onto the surface from
                 channel (ind-self.Nel).
        '''

        occ = self.state[:self.Nel]

        times = []
        times.append(enumerate(self.lifetimes[occ]/(self.tickrates+self.MTR)))
        # optional cohopping
        if self.enable_cohop:
            cohop_times = {ij: self.cohop_lt[ij]/(self.cohop_tickrates[ij]+self.MTR)
                                                    for ij in self.cohop_tickrates}
            times.append(cohop_times.items())
        # optional channel hopping
        if self.channels and not self.fixed_pop:
            ch_times = [channel.peek() for channel in self.channels]
            times.append(enumerate(ch_times, self.Nel))

        ind, dt = min(chain(*times), key=lambda x:x[1])

        return dt, ind

    def burn(self, nhops):
        '''Burns through the given number of hopping events'''

        # supress printing for burn
        self.vprint, tprint = lambda *a, **k: None, self.vprint

        for n in range(nhops):
            sys.stdout.write("\rBurning: {0:3.1f}%".format((n+1)*100./nhops))
            sys.stdout.flush()
            self.run(self.peek()[0])

        self.vprint = tprint


    def step(self, dt=np.inf):
        '''Advance the HoppingModel by the smaller of dt or its internal
        time step.

        returns:
            tick    : time advancement
        '''

        # figure out the time step
        dt_hop, ind = self.peek()   # hopping events
        dt_ch = min(ch.tick() for ch in self.channels)

        # advance lifetimes and channel states
        tick = min(dt, dt_hop, dt_ch)
        self._advance(tick)

        # handle hops
        if dt_hop==tick:
            self._hop_handler(ind)
        self.update()

        return tick


    def run(self, dt):
        '''Run the inherent dynamics for the given number of seconds'''

        while dt>0:
            dt -= self.step(dt)

    def measure(self, ind, dt=0.):
        '''Make a measurement of the indicated db after the given number of
        seconds.'''

        # keep hopping until the measurement event
        self.run(dt)

        # return the charge state of the requested db
        return self.charge[ind]

    def getLifetime(self, n):
        '''Get the expected lifetime, in seconds, of the given DB'''
        try:
            ind = int(np.where(self.state==n)[0])
        except:
            print('Invalid DB index')
            return None

        if ind < self.Nel:
            return self.lifetimes[n]/(self.MTR+self.tickrates[ind])
        return 0.

    def computeEnergy(self, occ=None):
        '''Direct energy computation for the current charge configurations'''

        inds = self.state[:self.Nel] if occ is None else occ
        beff = self.bias - .5*np.sum(self.V[:,inds], axis=1)
        beff += sum(ch.biases(inds) for ch in self.channels)
        return -np.sum(beff[inds])

    def addCharge(self, x, y, pos=True):
        '''Add the potential contribution from a charge at location (x,y). If pos
        is False, removes the influence of that charge'''

        dX, dY = self.a*self.X - x, self.b*self.Y - y
        R = np.sqrt(dX**2+dY**2)
        V = self.coulomb(R)
        self.bias -= V if pos else -V
        self.update()


    # internal methods

    def _parseX(self, X):
        '''Parse the DB location information'''

        f = self.c/self.b   # dimer pair relative separation factor

        X, Y, B = zip(*map(lambda x: (x,0,0) if isinstance(x,int) else x, X))
        self.X = np.array(X).reshape([-1,])
        self.Y = np.array([y+f*bool(b) for y,b in zip(Y,B)]).reshape([-1,])
        self.N = len(self.X)

    def _advance(self, dt):
        '''Advance the lifetimes and channels by the given time'''

        # lifetime updates
        self.lifetimes[self.state[:self.Nel]] -= dt*self.tickrates
        if self.channels:
            for channel in self.channels:
                channel.run(dt)
        if self.enable_cohop:
            for ij, tickrate in self.cohop_tickrates.items():
                self.cohop_lt[ij] -= dt*tickrate

    def _hop_handler(self, ind):
        '''Handle all the possible hopping cases.'''

        if isinstance(ind, tuple):
            self._cohop(ind)
        elif ind < self.Nel:
            self._hop(ind)
        else:
            self._channel_pop(ind-self.Nel)
        self.energy = self.computeEnergy()

    def _cohop(self, ij):
        '''Perform a cohop with the given pair i,j'''

        # determine targets
        targets, P = zip(*self.cohop_rates[ij].items())
        ind = np.random.choice(range(len(targets)), p=np.array(P)/sum(P))
        k, l = targets[ind]

        # modify charge state
        self._surface_hop(ij[0],k)
        self._surface_hop(ij[1],l)


    def _hop(self, ind):
        '''Perform a hop with the given electron'''

        src = self.state[ind]   # index of the electron to hop

        # determine target
        P, targets = self.trates[ind], list(range(self.N-self.Nel))
        if self.channels and not self.fixed_pop:
            P = np.hstack([P, [np.sum(self.crates[ind])]])
            targets.append(-1)
        t_ind = np.random.choice(targets,p=P/P.sum())

        # modify the charge state
        if t_ind < 0:
            self._channel_hop(ind)
        else:
            self._surface_hop(ind, t_ind)

    def _channel_hop(self, ind):
        '''Hop the electron given off the surface to some channel'''

        src = self.state[ind]
        self.charge[src] = 0
        self.state[ind], self.state[self.Nel-1] = self.state[self.Nel-1], self.state[ind]

        self.Nel -= 1

        # forget all cohopping lifetimes relatted to self.Nel-1
        if self.enable_cohop:
            for i in range(self.Nel):
                ij = (i,self.Nel)
                if ij in self.cohop_lt:
                    del self.cohop_lt[ij]

    def _channel_pop(self, cind):
        '''Hop an electron from the give channel onto the surface'''
        ind = self.channels[cind].pop()+self.Nel
        target = self.state[ind]
        self.charge[target] = 1
        self.lifetimes[target] = self.rebirth()
        self.state[self.Nel], self.state[ind] = self.state[ind], self.state[self.Nel]

        # add on new set of cohopping lifetimes
        if self.enable_cohop:
            for i in range(self.Nel):
                self.cohop_lt[(i,self.Nel)] = self.rebirth()

        self.Nel += 1

    def _surface_hop(self, ind, t_ind):
        '''Hop the electron given by ind to the empty db given by t_ind'''

        print('Surface Hop: {0} -> {1} :: dE = {2:.3f}'.format(
                self.state[ind], self.state[t_ind], self.dG[ind, t_ind]))

        src, target = self.state[ind], self.state[self.Nel+t_ind]
        self.charge[src], self.charge[target] = 0, 1
        self.state[ind], self.state[self.Nel+t_ind] = target, src
        self.lifetimes[target] = self.rebirth()

        # reset all cohopping times involving ind
        if self.enable_cohop:
            for ij in self.cohop_lt:
                if ind in ij:
                    self.cohop_lt[ij]= self.rebirth()

    def _compute_cohop_dG(self, i, j, k, l, si, sj, sk, sl):
        return self.dG[i,k]+self.dG[j,l] \
            + (self.V[sk,sl]+self.V[si,sj]) \
            - (self.V[si,sl]+self.V[sj,sk])
