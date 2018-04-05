#!/usr/bin/env python
# encoding: utf-8

'''
Collection of different hopping models for calculating hopping rates
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-04-05'  # last update

import numpy as np

# Virtual base class for hopping models

class BaseModel(object):
    '''Virtual base class for all tunneling rate models'''

    # shared physical constants
    hbar    = 6.582e-16     # eV.s

    alph    = 10.   # inverse attenuation length, 1/angstroms
    prefact = 1e-3  # scaling prefactor for rates
    ktf     = 1.    # thermal broadening factor
    lamb    = 0.04  # reorganization energy, eV

    def __init__(self):
        '''Initialise a hopping rate model'''
        pass

    def setup(self, X, Y, kt):
        '''Setup the model for the given DB arrangement

        inputs:
            X  : matrix of x positions, angstroms
            Y  : matrix of y positions, angstroms
            kt : thermal energy of surface, eV
        '''
        self.beta = 1./kt
        dX, dY = X-X.reshape(-1,1), Y-Y.reshape(-1,1)
        self.R = np.sqrt(dX**2+dY**2)
        self._spatial_rates()

    def rates(self, dG, occ, nocc):
        '''Compute the hopping rates from the energy deltas

        inputs:
            dG      : matrix of energy deltas with dG[i,j] the change for a
                      hop from site occ[i] to site nocc[j].
            occ     : list of occupied sites, possible sources
            nocc    : list of unoccupied sites, possible targets
        '''
        raise NotImplementedError()

    def cohopping_rate(self, dG, i,j,k,l):
        '''Compute the cohopping rate from sites i,j to sites k,l with the given
        energy delta

        inputs:
            dG  : energy delta associated with the cohopping event
            i,j : electron source
            k,l : db targets
        '''
        raise NotImplementedError()

    def setAttenuation(self, alph):
        self.alph = alph
        self._spatial_rates()

    def setPrefactor(self, fact):
        self.prefact = fact
        self._spatial_rates()

    def setLambda(self, lamb):
        self.lamb = lamb

    # internal methods

    def _spatial_rates(self):
        '''compute the spatial decay prefactors'''
        self.T0 = self.prefact*np.exp(-2*self.R/self.alph)


# derived models

class VRHModel(BaseModel):
    '''Variable-Range Hopping model for hopping rates'''

    # model-specific parameters
    prefact = 1.e11     # scaling prefactor for rates

    lamb    = 0.01      # self-trapping energy, eV
    expmax  = 1e2       # maximum argument for exp(x)

    def __init__(self):
        super(VRHModel, self).__init__()

    def setup(self, X, Y, kt):
        super(VRHModel, self).setup(X, Y, kt)

    def rates(self, dG, occ, nocc):
        arg = -self.beta*(dG+self.lamb)/self.ktf
        return self.T0[occ,:][:,nocc]*self._exp(arg)

    def _exp(self, arg):
        '''Thresholded version of exp'''
        mask = arg <= self.expmax
        arg = arg*mask + self.expmax*(1-mask)
        return np.exp(arg)


class MarcusModel(BaseModel):
    '''Marcus Theory model for hopping rates'''

    # model-specific parameters
    lamb    = 0.04      # reorganization energy, eV

    # transfer integral parameters
    prefact = 1e-3      # prefactor

    # cohopping parameters
    cohop_lamb = lamb   # cohopping reorganization energy, eV
    cohop_alph = 1e-2   # cohopping inverse attenuation length, 1/angstroms

    def __init__(self):
        super(MarcusModel, self).__init__()

    # inherited methods

    def setup(self, X, Y, kt):
        super(MarcusModel, self).setup(X, Y, kt)
        self.setLambda(self.lamb)

    def setLambda(self, lamb):
        self.lamb = lamb
        self.lbeta = np.inf if lamb == 0 else self.beta/lamb/self.ktf
        self.Tp = self.T0*np.sqrt(self.lbeta*np.pi)/self.hbar

    def rates(self, dG, occ, nocc):
        return self.Tp[occ,:][:,nocc]*np.exp(-.25*self.lbeta*(dG+self.lamb)**2)

    def cohopping_rate(self, dG, i, j, k, l):
        return self.cohop_tint(i,j,k,l)*np.exp(-.25*self.lbeta*(dG+self.cohop_lamb)**2)


    # specific methods

    def cohop_tint(self, i,j,k,l):
        return np.sqrt(self.Tp[i,k]*self.Tp[j,l]+self.Tp[i,l]*self.Tp[j,k])*np.exp(-self.cohop_alph*self.R[i,j])

models = {  'marcus':   MarcusModel,
            'VRH':      VRHModel }
