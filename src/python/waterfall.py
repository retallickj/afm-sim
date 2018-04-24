#!/usr/bin/env python
# encoding: utf-8

'''
Generation of waterfall figures
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-04-17'  # last update

from hopper import HoppingModel
import os

import numpy as np
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class Waterfall:

    log_fn = os.path.join('.', '.temp', 'waterfall.log')
    img_fn = os.path.join('.', 'img', 'waterfall.pdf')

    ppnm = 50   # pixels per nm
    sig = .8    # 'atom diameter'

    lo, hi = 1., 2.     # amplitudes of unoccupied/occupied DBs
    portrait = True     # portrait style plotting
    xticks = False

    # coloring for plots
    cdict = {'red':    ((0, .48, .48),
                        (.5, .14, .14),
                        (1., 0, 0)),
             'green':  ((0,.82,.82),
                        (.5,.55,.55),
                        (1., 0, 0)),
             'blue':   ((0,.89,.89),
                        (.5,.70,.70),
                        (1., 0, 0))
             }

    cm = LinearSegmentedColormap('cm', cdict)

    def __init__(self, hopper):
        '''Initialise a waterfall generator. For now, assumes the
        device being simulated is linear in the x direction'''

        self.hopper = hopper
        self.X = .1*self.hopper.a*self.hopper.X
        self.N = self.hopper.N



    def generate(self, nscans=100, srate=10.0, pad=1.0, mu=.25, lamb=0.04):
        '''Generate the waterfall image.

        inputs:
            nscans  : number of line scans
            srate   : tip scan rate, in nm/s
            pad     : padding on either side of device, in nm
            mu      : Ef-DB- difference, eV
            lamb    : self-trapping energy
        '''

        if not self.hopper.initialised:
            self.hopper.initialise()

        bulk = self.hopper.getChannel('bulk')
        if bulk is not None:
            bulk.mu = mu

        self.hopper.model.setLambda(lamb)

        # burn
        self.hopper.burn(10, per=True)

        # determine run time of scan
        xlo, xhi = np.min(self.X)-pad, np.max(self.X)+pad

        self.T = nscans*(xhi-xlo)/srate

        # populate hopping log
        self.hopper.startLog(self.log_fn)
        self.hopper.run(self.T)
        self.hopper.endLog()

        # simulate scan process
        self._show(nscans, srate, xlo, xhi)

    def _parser(self):
        '''Parse the time and state information from the log file'''

        with open(self.log_fn, 'r') as fp:
            state = self.hopper.charge
            for line in fp:
                t, s = line.split(' :: ')
                t = float(t)
                s = format(int(s, base=16), '0{0}b'.format(self.N))
                state = [int(x) for x in s]
                yield t, state
            yield np.inf, state     # hold on last state


    def _show(self, nscans, srate, xlo, xhi):
        '''Generate the waterfall from the hopping log'''

        xx = np.linspace(xlo, xhi, 1+int((xhi-xlo)*self.ppnm))
        kernel = np.exp(-np.linspace(-2,2, 1+int(self.sig*self.ppnm))**2)
        nx = len(xx)

        # index array, nearest db at each position
        D = np.abs(xx.reshape(-1,1) - self.X.reshape(1,-1))
        ind = np.argmin(D, axis=1)

        # impulse array
        imp = np.zeros(xx.shape, dtype=float)
        for x in self.X:
            n, r = divmod((x-xlo)*self.ppnm, 1)
            imp[int(n):int(n)+2] = 1-r, r

        lo_val = np.convolve(imp, self.lo*kernel, 'same')
        hi_val = np.convolve(imp, self.hi*kernel, 'same')

        # generate waterfall
        self.state_gen = self._parser()
        self.t, self.state = 0., None   # cache

        n, dn, dt = 0, 1, (xhi-xlo)/(srate*nx)
        data = np.zeros([nscans, nx], dtype=float)
        for m in range(nscans):
            for _ in range(nx):
                data[m,n] = self._integrate(ind[n], dt)
                n += dn
            data[m,:] = lo_val + data[m,:]*(hi_val-lo_val)
            n, dn = n - dn, -dn

        self._plot_handler(xlo, xhi, data)



    def _plot_handler(self, xlo, xhi, data):
        '''Create the waterfall plot'''

        # formatting options
        FS = 20
        TFS = 18
        size, asp = 4, 1.9

        if self.portrait:
            fig, (ax, cax) = plt.subplots(nrows=2, figsize=(size, size*asp),
                        gridspec_kw={'height_ratios': [1, .02]})
            dat = data
            xext, yext = xhi-xlo, self.T/60
            xlab, tlab = ax.set_xlabel, ax.set_ylabel
            cor = 'horizontal'
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
        else:
            fig, (ax, cax) = plt.subplots(ncols=2, figsize=(size*asp, size),
                        gridspec_kw={'width_ratios': [1, .05]})
            dat = data.T
            xext, yext = self.T/60, xhi-xlo
            xlab, tlab = ax.set_ylabel, ax.set_xlabel
            cor = 'vertical'

        im = ax.imshow(dat, interpolation='None', aspect='auto', cmap=self.cm,
                        extent=[0, xext, 0, yext])

        # ticks
        tick_fill = lambda f, ext: f([round(x*ext,1) for x in [0, .5, 1.]])
        if self.xticks:
            tick_fill(ax.set_xticks, xext)
            tick_fill(ax.set_yticks, yext)
            xlab('x (nm)', fontsize=FS)
            tlab('Time (min)', fontsize=FS)
        else:
            if self.portrait:
                ax.set_xticks([])
                tick_fill(ax.set_yticks, yext)
            else:
                tick_fill(ax.set_xticks, xext)
                ax.set_yticks([])
            tlab('Time (min)', fontsize=FS)
        ax.tick_params(axis='both', which='major', labelsize=TFS)

        # colorbar
        cbar = fig.colorbar(im, cax=cax, ticks=[0,1,2], orientation=cor)
        cbar.set_label('Charges', fontsize=FS)
        cbar.ax.tick_params(labelsize=TFS)


        plt.tight_layout(pad=0, h_pad=0, w_pad=0)

        fname = self.img_fn
        direc = os.path.dirname(fname)
        if not os.path.isdir(direc):
            os.makedirs(direc)

        plt.savefig(fname, bbox_inches='tight')
        plt.show()



    def _integrate(self, n, dt):
        '''Integrate the charge of the n^th DB for dt seconds'''

        charge, norm = 0., 1./dt
        while dt > 0:
            if self.t > 0:
                t, state = self.t, self.state
                self.t = 0.
            else:
                t, state = next(self.state_gen)
            if t<dt:
                charge += t*state[n]
                dt -= t
            else:
                charge += dt*state[n]
                self.t, self.state = t-dt, list(state)
                dt = 0

        return charge*norm


if __name__ == '__main__':

    d1221 = [8,10,15,17]
    d1221.insert(0, d1221[0]-7)
    d1221.append(d1221[-1]+7)

    device = d1221

    model = HoppingModel(device, model='marcus')
    model.addChannel('bulk')
    waterfall = Waterfall(model)
    waterfall.generate(nscans=800, srate=8.9, pad = 1.6, mu=0.1)
