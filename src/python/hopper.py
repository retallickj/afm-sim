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
from models import models

class HoppingModel:
    '''Time dependent surface hopping model for charge transfer in DBs'''

    # machine precision
    MTR     = 1e-16         # for tickrates division

    # energy parameters
    debye   = 50            # debye screening length, angstroms
    eps0    = 8.854e-12     # F/m
    q0      = 1.602e-19     # C
    epsr    = 11.7          # relative permittivity

    Kc = 1e10*q0/(2*np.pi*epsr*eps0)    # Coulomb strength, eV.angstrom

    # lattice parameters
    a = 3.84    # lattice vector in x, angstroms    (intra dimer row)
    b = 7.68    # lattice vector in y, angstroms    (inter dimer row)

    # general settings
    fixed_pop = True        # fixed number of electrons
    fixed_rho = 0.5         # filling density if fixed_pop (Nel = round(N*fixed_rho))
    burn_time = 500         # initial burn time, seconds

    # useful lambdas
    rebirth = np.random.exponential     # reset for hopping lifetimes

    def __init__(self, X, Y=None, model='VRH', **kwargs):
        '''Construct a HoppingModel for a DB arrangement with the given x and
        optinoal y coordinates in unit of the lattice vectors. For now, assume
        only the top site of each dimer pair can be a DB.

        inputs:
            X       : Dimer column for each DB
            Y       : Dimer row for each DB
            model   : Type of hopping rate model

        optional key-val arguments:
            None
        '''

        # format and store db locations and number
        self.X = np.array(X).reshape([-1,])
        self.N = len(self.X)
        self.Y = np.zeros(self.N) if Y is None else np.array(Y).reshape([-1,])

        assert self.X.shape == self.Y.shape, 'X,Y shape mismatch'

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
        self.Nel = int(round(self.N*self.fixed_rho))

        # setup model
        if model not in models:
            raise KeyError('Invalid model type. Choose from [{0}]'.format(
                                ', '.join(models.keys())))
        self.model = models[model]()
        self.model.setup(dX, dY)

    def setElectronCount(self, n):
        '''Set the number of electrons in the system'''

        if n<0:
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

        self.update()

        self.lifetimes = [self.rebirth() for _ in range(self.Nel)]
        self.energy = self.computeEnergy()

        if charges is None:
            # burn off random initial state
            self.run(self.burn_time)

    def update(self):
        '''Update the energy deltas, tunneling rates, and tick rates'''

        occ, nocc = self.state[:self.Nel], self.state[self.Nel:]

        # effective bias at each location
        beff = self.bias+self.dbias-np.dot(self.V, self.charge)

        # update parameters.... magic math
        self.dG = beff[occ].reshape(-1,1) - beff[nocc] - self.V[occ,:][:,nocc]

        self.trates = self.model.rates(self.dG, occ, nocc)
        self.tickrates = np.sum(self.trates, axis=1)

    def peek(self):
        '''Return the time before the next tunneling event and the index of the
        event'''

        tdeltas = self.lifetimes/(self.tickrates+self.MTR)
        ind = np.argmin(tdeltas)

        return tdeltas[ind], ind

    def hop(self, ind):
        '''Perform a hop with the given electron'''

        src = self.state[ind]

        # determine target
        P = np.copy(self.trates[ind])
        t_ind = np.random.choice(range(self.Nel, self.N),p=P/P.sum())
        target = self.state[t_ind]

        # update
        self.energy += self.dG[ind, t_ind-self.Nel]
        self.charge[src], self.charge[target] = 0, 1
        self.state[ind], self.state[t_ind] = target, src
        self.lifetimes[ind] = self.rebirth()
        self.update()

    def run(self, dt):
        '''Run the inherent dynamics for the given number of seconds'''

        while dt>0:
            tick, ind = self.peek()
            mtick = min(dt, tick)
            self.lifetimes -= mtick*self.tickrates
            if tick<=dt:
                self.hop(ind)
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
