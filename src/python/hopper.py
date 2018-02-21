#!/usr/bin/env python
# encoding: utf-8

'''
Simulator for the non-equilibrium surface dynamics of charges in QSi's DB
arrangements
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-02-14'  # last update

import numpy as np
from model import models as Models
from channel import Channel, channels as Channels

class HoppingModel:
    '''Time dependent surface hopping model for charge transfer in DBs'''

    # machine precision
    MTR     = 1e-16         # for tickrates division

    # energy parameters
    debye   = 50            # debye screening length, angstroms
    eps0    = 8.854e-12     # F/m
    q0      = 1.602e-19     # C
    kb      = 8.617e-05     # eV/K
    T       = 4.0           # system temperature, K
    epsr    = 11.7          # relative permittivity

    Kc = 1e10*q0/(4*np.pi*epsr*eps0)    # Coulomb strength, eV.angstrom

    # lattice parameters
    a = 3.84    # lattice vector in x, angstroms    (intra dimer row)
    b = 7.68    # lattice vector in y, angstroms    (inter dimer row)
    c = 2.25    # dimer pair separation, angstroms

    # general settings
    fixed_pop = False    # fixed number of electrons
    free_rho = 0.5       # filling density if not fixed_pop (Nel = round(N*free_rho))
    burn_count = 10      # number of burns hops per db

    # useful lambdas
    rebirth = np.random.exponential     # reset for hopping lifetimes

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
        self.V = self.Kc/(np.eye(self.N)+self.R)*np.exp(-self.R/self.debye)
        np.fill_diagonal(self.V,0)

        # by default, use
        self.Nel = int(round(self.N*self.free_rho))

        # create model, setup on initialisation
        if model not in Models:
            raise KeyError('Invalid model type. Choose from [{0}]'.format(
                                ', '.join(Models.keys())))
        self.model = Models[model]()
        self.channels = []

        # TODO: remove... leave to user to include bulk
        self.addChannel('bulk')

    def fixElectronCount(self, n):
        '''Fix the number of electrons in the system. Use n<0 to re-enable
        automatic population mechanism.'''

        if n<0:
            self.Nel = int(round(self.N*self.free_rho))
            self.fixed_pop = False
        else:
            assert n <= self.N, 'Invalid number of electrons'
            self.Nel = n
            self.fixed_pop = True

    # TODO: this part sucks... insert Lucian's model
    def writeBias(self, bias, ind, dt, sigma):
        '''Influence of the tip on system when over the given db. Applies a square
        wave potential bias to the indicated db of length sigma seconds and
        ending at dt'''

        self.run(max(0,dt-sigma))
        self.dbias[ind] = bias
        self.update()
        self.run(sigma)
        self.dbias[ind]=0
        self.update()

    def setBiasGradient(self, F, v=[1,0]):
        '''Apply a bias gradient of strength F (eV/angstrom) along the
        direction v'''

        self.bias = F*(v[0]*self.a*self.X+v[1]*self.b*self.Y)

    def addChannel(self, channel):
        '''Add a Channel instance to the HoppingModel.

        inputs:
            channel : Channel to include. Must either be a Channel instance or
                     a string indicating an accepted channel type in Channels.
        '''

        if isinstance(channel, str):
            if channel not in Channels:
                raise KeyError('Invalid channel type. Choose from [{0}]'.format(
                    ', '.join(k for k in Channels if k != 'base')))
            self.channels.append(Channels[channel]())
        elif isinstance(channel, Channel):
            self.channels.append(channel)


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
            assert Nel==self.Nel, 'Electron count mismatch'
            self.state = np.array(occ+nocc)

        self.charge[self.state[:self.Nel]]=1
        self.charge[self.state[self.Nel:]]=0

        # setup model and channels
        X, Y, kt  = self.a*self.X, self.b*self.Y, self.kb*self.T
        self.model.setup(X, Y, kt)
        for channel in self.channels:
            channel.setup(X, Y, kt)

        self.update()

        self.lifetimes = np.array([self.rebirth() for _ in range(self.N)])
        self.energy = self.computeEnergy()

        if charges is None:
            # burn off random initial state
            self.burn(self.burn_count*self.N)



    def update(self):
        '''Update the energy deltas, tunneling rates, and tick rates'''

        occ, nocc = self.state[:self.Nel], self.state[self.Nel:]

        # TODO: include channel contributions to beff
        # effective bias at each location
        beff = self.bias+self.dbias-np.dot(self.V, self.charge)

        for channel in self.channels:
            channel.update(occ, nocc, beff)

        # update parameters.... magic math
        self.dG = beff[occ].reshape(-1,1) - beff[nocc] - self.V[occ,:][:,nocc]

        self.beff = beff    # store for dE on channel hop
        self.trates = self.model.rates(self.dG, occ, nocc)
        self.tickrates = np.sum(self.trates, axis=1)
        if self.channels and not self.fixed_pop:
            self.crates = np.array([channel.rates() for channel in self.channels]).T
            self.tickrates += np.sum(self.crates, axis=1)

    def peek(self):
        '''Return the time before the next tunneling event and the index of the
        event:

        output:
            dt  : time to next hopping event, s
            ind : index of event: if ind < self.Nel the electron at state[ind]
                 hops, otherwise an electron hops onto the surface from
                 channel (ind-self.Nel).
        '''

        occ = self.state[:self.Nel]
        tdeltas = self.lifetimes[occ]/(self.tickrates+self.MTR)
        if self.channels and not self.fixed_pop:
            cdeltas = [channel.peek() for channel in self.channels]
            tdeltas = np.hstack([tdeltas, cdeltas])
        ind = np.argmin(tdeltas)

        return tdeltas[ind], ind

    def hop(self, ind):
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
            self._surface_hop(t_ind, ind)

        self.update()


    def burn(self, nhops):
        '''Burns through the given number of hopping events'''

        for n in range(nhops):
            self.run(self.peek()[0])


    def run(self, dt):
        '''Run the inherent dynamics for the given number of seconds'''

        while dt>0:
            tick, ind = self.peek()
            mtick = min(dt, tick)
            self.lifetimes[self.state[:self.Nel]] -= mtick*self.tickrates
            if tick<=dt:
                if ind < self.Nel:
                    self.hop(ind)
                else:
                    self._channel_pop(ind-self.Nel)
            dt -= mtick


    def measure(self, ind, dt):
        '''Make a measurement of the indicated db after the given number of
        seconds.'''

        # keep hopping until the measurement event
        self.run(dt)

        # return the charge state of the requested db
        return self.charge[ind]

    def computeEnergy(self):
        '''Direct energy computation for the current charge configurations'''

        inds = self.state[:self.Nel]
        return -np.sum((self.bias+self.dbias)[inds]) + .5*np.sum(self.V[inds,:][:,inds])

    def _parseX(self, X):
        '''Parse the DB location information'''

        f = self.c/self.b   # dimer pair relative separation factor

        X, Y, B = zip(*map(lambda x: (x,0,0) if isinstance(x,int) else x, X))
        self.X = np.array(X).reshape([-1,])
        self.Y = np.array([y+f*bool(b) for y,b in zip(Y,B)]).reshape([-1,])
        self.N = len(self.X)

    def _channel_hop(self, ind):
        '''Hop the electron given off the surface to some channel'''

        src = self.state[ind]
        self.energy += self.beff[src]
        self.charge[src] = 0
        self.state[ind], self.state[self.Nel-1] = self.state[self.Nel-1], self.state[ind]
        self.Nel -= 1

    def _channel_pop(self, cind):
        '''Hop an electron from the give channel onto the surface'''
        ind = self.channels[cind].pop()+self.Nel
        target = self.state[ind]
        self.energy -= self.beff[target]
        self.charge[target] = 1
        self.lifetimes[target] = self.rebirth()
        self.state[self.Nel], self.state[ind] = self.state[ind], self.state[self.Nel]
        self.Nel += 1
        self.update()

    def _surface_hop(self, t_ind, ind):
        '''Hop the elctron given by ind to the empty db given by t_ind'''

        src, target = self.state[ind], self.state[self.Nel+t_ind]
        self.energy += self.dG[ind, t_ind]
        self.charge[src], self.charge[target] = 0, 1
        self.state[ind], self.state[self.Nel+t_ind] = target, src
        self.lifetimes[target] = self.rebirth()
