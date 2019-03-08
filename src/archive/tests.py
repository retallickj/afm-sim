#!/usr/bin/env python
# encoding: utf-8


__author__      = 'Jake Retallick'
__copyright__   = 'Apache License 2.0'
__version__     = '1.1'
__data__        = '2018-03-19'

'''
Testing Tools for the Hopping Model simulations
'''

import numpy as np
import matplotlib.pyplot as plt

from hopper import HoppingModel

class EnergySweep:
    '''Calculate the energy of each state over a sweep of tip positions'''

    def __init__(self, model):
        ''' '''

        self.model = model

        self.tip = model.getChannel('tip')
        assert self.tip is not None, 'HoppingModel is missing a tip channel'

        if not self.model.initialised:
            self.model.initialise()

    def sweep(self, start, end, nsteps):
        '''Compute the energies for all configurations over a linear trajectory
        of the tip.

        inputs:
            start   : tip starting position: (x,y)
            end     : tip ending position: (x,y)
            nsteps  : number of increments between the start and end inclusive
        '''

        X = np.linspace(start[0], end[0], nsteps)
        Y = np.linspace(start[1], end[1], nsteps)

        nto_bin = lambda n: format(n, '0{0}b'.format(int(self.model.N)))
        nto_array = lambda n: np.array([c=='1' for c in nto_bin(n)], dtype=int)
        to_occ = lambda ar: list(np.nonzero(ar)[0])
        states = [nto_array(n) for n in range(2**self.model.N)]
        occs = [to_occ(s) for s in states]

        E = np.zeros([nsteps, len(occs)], dtype=float)
        for i, (x,y) in enumerate(zip(X,Y)):
            self.tip.setPos(x, y)
            for j, occ in enumerate(occs):
                E[i,j] = self.model.computeEnergy(occ=occ)

        # pick the low energy confiurations
        mins = np.min(E, axis=0)
        _, order = zip(*sorted((m,n) for n,m in enumerate(mins)))
        E = E[:,order[:20]]

        plt.plot(E)
        plt.show(block=True)


if __name__ == '__main__':

    line = [0, 7, 9, 14, 16, 23]
    device = line

    model = HoppingModel(device, model='marcus')
    tip = model.addChannel('tip')
    tip.setHeight(.4)

    test = EnergySweep(model)

    tip.computeDeltas()

    pad, fact = 2, model.a*.1
    xlo, xhi = (line[0]-pad)*fact, (line[-1]+pad)*fact

    test.sweep([xlo,0], [xhi,0], 240)
