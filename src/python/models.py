#!/usr/bin/env python
# encoding: utf-8

'''
Collection of different hopping models for calculating hopping rates
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-02-14'  # last update

import numpy as np

# Virtual base class for hopping models

class BaseModel(object):
    '''Virtual base class for all tunneling rate models'''

    # shared physical constants
    hbar    = 6.582e-16     # eV.s
    kb      = 8.617e-05     # eV/K
    T       = 4.            # K

    def __init__(self):
        ''' '''
        pass

    def setup(self, dX, dY):
        '''Setup the model for the given DB arrangement

        inputs:
            dX  : matrix of x differences, angstroms
            dY  : matrix of y differences, angstroms
        '''
        raise NotImplementedError()

    def rates(self, dG, occ, nocc):
        '''Compute the hopping rates from the energy deltas

        inputs:
            dG      : matrix of energy deltas with dG[i,j] the change for a
                      hop from site occ[i] to site nocc[j].
            occ     : list of occupied sites, possible sources
            nocc    : list of unoccupied sites, possible targets
        '''
        raise NotImplementedError()


# derived models

class VRHModel(BaseModel):
    '''Variable-Range Hopping model for hopping rates'''

    # model-specific parameters
    alph    = 1e-2     # inverse attenuation length, 1/angstroms
    r0      = 1.e24    # scaling prefactor for rates
    lamb    = 0.02     # self-trapping energy, eV

    def __init__(self):
        super(VRHModel, self).__init__()
        self.beta = 1./(self.kb*self.T)

    def setup(self, dX, dY):
        R = np.sqrt(dX**2+dY**2)
        self.T0 = self.r0*np.exp(-2*self.alph*R)
        print(self.T0)

    # TODO: problem with exp overflow here, decreases in energy beyond lamb
    #       cause essentially instantaneous hops
    def rates(self, dG, occ, nocc):
        return self.T0[occ,:][:,nocc]*np.exp(-self.beta*(dG+self.lamb))


class MarcusModel(BaseModel):
    '''Marcus Theory model for hopping rates'''

    # model-specific parameters
    lamb    = 0.11      # reorganization energy, eV

    # transfer integral parameters
    t0      = 1e8       # prefactor
    alph    = 1e-2      # inverse attenuation length, 1/angstroms

    a = 3.84      # length scale

    def __init__(self):
        super(MarcusModel, self).__init__()
        self.lbeta = 1./(self.lamb*self.kb*self.T)

    # inheritted methods

    def setup(self, dX, dY):
        R = np.sqrt(dX**2+dY**2)
        self.Tp = np.abs(self.tint(R))**2/self.hbar*np.sqrt(self.lbeta*np.pi)

    def rates(self, dG, occ, nocc):
        return self.Tp[occ,:][:,nocc]*np.exp(-.25*self.lbeta*(dG+self.lamb)**2)

    # specific methods

    def tint(self, R):
        tij = self.t0*np.exp(-self.alph*R)
        return tij

models = {  'marcus':   MarcusModel,
            'VRH':      VRHModel }
