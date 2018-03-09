#!/usr/bin/env python
# encoding: utf-8


'''
Model for computing the influence of the AFM tip on the DBs
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-03-08'  # last update

import numpy as np
import scipy.constants as const

import os.path


class TipModel:
    '''Tip Model'''

    # physics constants
    q0   = const.e
    Kc   = 1e9*const.e/(4*const.pi*const.epsilon_0)   # Coulomb strength, eV.[nm]
    epsr = 6.35     # dielectric constant for the surface
    epsi = 9.0      # dielectric constant for surface-tip

    # empirical parameters
    tunrate0 = 43.3e9       # lateral decay of the tunneling rate tip-DB, Hz
    k_tip_DBm = 6.28        # spatial decay of the tunneling rate, 1/nm
    E_DB_m_bulkCB = -0.32   # negative DB level in bulk wrt CBM, eV

    # experimental fit parameters
    tipR    = 7.    # tip radius in nm
    tipDW   = 0.65  # actual difference in tip-sample work-functions

    # calibration data
    TIBBvH_fname = os.path.join('.', 'TIBB_vs_H.dat')
    TIBBvR_fname = os.path.join('.', 'TIBB_vs_R_d200pm.dat')
    tipR0   = 5.0   # tip radius in FEM calculations, nm
    tipH0   = 0.2   # tip height in FEM calculations, nm
    tipDW0  = 0.9   # tip-sample work-function difference in FEM calculations, eV

    # useful lambdas


    def __init__(self):
        '''Initialise the model parametrization'''

        self._TIBBvH_calib()
        self._TIBBvR_calib()

    def setup(self, X, Y):
        '''Setup the tip model for the given DB locations'''

        self.X = np.array(X).reshape([-1,])
        self.Y = np.array(Y).reshape([-1,])

    def setPos(self, x, y):
        '''Update the tip position, make the appropriate pre-calcs'''
        self.x, self.y = x, y


    def setRadius(self, R):
        '''Set the tip radius in nm, make appropriate pre-calcs'''
        self.tipR = R


    def setHeight(self, H):
        '''Update the tip height, make appropriate pre-calcs'''
        self.tipH = H

    def energy_shifts(self, occ):
        '''Calculate the tip induced level shifts for each DB when the given
        DBs are occupied'''
        pass


    # internal methods

    def _updateTIBB(self):
        '''update the TIBB interpolator for new tip radius or height'''
        self.TIBB = lambda r: None

        R, H, R0, femR, P = self.tipR, self.tipH, self.tipR0, self.femR, self.P
        factR = ((R+H)/(R0+H))**P[0]*(1.-(R-R0)/(R+H)*np.exp(-(P[3]*femR/R)**P[1]))
        TIBB = self.femTIBB*self.TIBBvH_fit(H) ... #NB

    def _TIBBvH_calib(self):
        '''Calibrate the H dependence of the TIBB'''

        data = np.genfromtxt(self.TIBBvH_fname, delimiter=None)
        assert data.shape[1]==3, 'TIBBvH calibration data must be 3-column'
        Htip, TIBB, ICIBB = data.T
        coefs = np.polyfit(Htip, TIBB*1e-3, 3)
        self.TIBBvH_fit = lambda H, np.polyval(coefs, H)


    def _TIBBvR_calib(self):
        '''Calibrate the distance, R, dependence of the TIBB for a specific
        height and tip radius'''

        data = np.genfromtxt(self.TIBBvR_fname, delimiter=None)
        assert data.shape[1]==2, 'TIBBvR calibration data must be 2-column'
        self.femR, self.femTIBB = data.T
        #self.femTIBB = interp1d(R, TIBB, fill_value='extrapolate')


if __name__ == '__main__':
    tip = TipModel()
    print(tip.coefs)
