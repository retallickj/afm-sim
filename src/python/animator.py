#!/usr/bin/env python
# encoding: utf-8

'''
Real-time animation of DB arrangements
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-02-15'  # last update

import numpy as np
from itertools import product

from PyQt5.QtCore import (Qt, QTimer, QThread)
from PyQt5.QtGui import (QPen, QBrush, QColor, QPainter)
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QGraphicsEllipseItem)

from hopper import HoppingModel


_SF = 10     # scale factor

class Thread(QThread):
    def __init__(self, func):
        super(Thread, self).__init__()
        self.func = func

    def run(self):
        ''' '''
        self.func()
        self.exec_()

class DB(QGraphicsEllipseItem):

    pen     = QPen(Qt.white, 3)     # DB edge pen
    bgpen   = QPen(Qt.darkGray, 1, Qt.DotLine)
    fill    = QBrush(Qt.green)      # charged DB fill color
    nofill  = QBrush(Qt.NoBrush)    # uncharged DB fill color

    D = 1.5*_SF              # dot diameter

    def __init__(self, x, y, bg=False, parent=None):
        super(DB, self).__init__(_SF*x, _SF*y, self.D, self.D, parent=parent)
        self.setPen(self.bgpen if bg else self.pen)
        self.setCharge(False)
        if not bg:
            self.setZValue(1)

    def setCharge(self, charged):
        '''Set the charge state of the DB'''

        self.setBrush(self.fill if charged else self.nofill)


class HoppingAnimator(QGraphicsView):
    ''' '''

    # lattice parameters
    a = 3.84    # lattice vector in x, angstroms    (intra dimer row)
    b = 7.68    # lattice vector in y, angstroms    (inter dimer row)
    c = 2.25    # dimer pair separation, angstroms

    WINX = 1000  # window width
    WINY = 400

    rate = 100  # millis/second

    bgcol = QColor(29, 35, 56)  # background color

    def __init__(self, model):
        '''Initialise the HoppingAnimator instance for the given DB positions.
        X should be formatted as for HoppingModel'''

        super(HoppingAnimator, self).__init__()

        assert isinstance(model, HoppingModel), 'Invalid model type'

        self.model = model
        self.X, self.Y = self.model.X, self.model.Y

        self._initGUI()

        self.model.initialise()
        self.model.burn(100)

    def _initGUI(self):
        '''Initialise the animator window'''

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._drawDBs()

        self.setGeometry(0, 0, self.WINX, self.WINY)
        self.setBackgroundBrush(QBrush(self.bgcol, Qt.SolidPattern))
        self.setWindowTitle('Hopping Animator')

    def _drawDBs(self):
        '''Draw all the DBs for the animator'''

        # background
        X = np.arange(np.min(self.X)-2, np.max(self.X)+3)
        Y = np.arange(round(np.min(self.Y)), round(np.max(self.Y))+1)

        f = self.c/self.b
        for x,y in product(X,Y):
            self.scene.addItem(DB(self.a*x, self.b*y, bg=True))
            self.scene.addItem(DB(self.a*x, self.b*(y+f), bg=True))

        # foreground
        self.dbs = []
        for x,y in zip(self.X, self.Y):
            self.dbs.append(DB(self.a*x,self.b*y))
            self.scene.addItem(self.dbs[-1])

    def tick(self):
        ''' '''

        for i,c in enumerate(self.model.charge):
            self.dbs[i].setCharge(c)

        dt = self.model.peek()[0]
        self.model.run(dt)

        millis = int(self.rate*dt)
        print(dt, millis)
        if millis>=1:
            self.timer = QTimer()
            self.timer.timeout.connect(self.tick)
            self.timer.start(min(millis, 10000))
        else:
            self.tick()



if __name__ == '__main__':

    import sys

    line = [1, 8, 10, 15, 17, 24]
    _or = [(0,0,1),(2,1,1),(6,1,1),(8,0,1),(4,3,0),(4,4,1),(4,6,0)]
    _or.append((-2,-1,1))
    _or.append((10,-1,1))

    device = _or

    model = HoppingModel(device, model='marcus')
    model.setElectronCount(5)

    app = QApplication(sys.argv)
    animator = HoppingAnimator(model)

    thread = Thread(animator.tick)
    thread.start()

    animator.show()
    sys.exit(app.exec_())