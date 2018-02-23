#!/usr/bin/env python
# encoding: utf-8

'''
Real-time animation of DB arrangements
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-02-15'  # last update

import shutil, os
import numpy as np
from itertools import product

from PyQt5.QtCore import (Qt, QTimer, QThread)
from PyQt5.QtGui import (QPen, QBrush, QColor, QPainter, QImage)
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
    WINY = 800

    rate = 100  # millis/second

    bgcol = QColor(29, 35, 56)  # background color
    record_dir = './.temp_rec/'

    def __init__(self, model, record=False, fps=30):
        '''Initialise the HoppingAnimator instance for the given DB positions.
        X should be formatted as for HoppingModel'''

        super(HoppingAnimator, self).__init__()

        assert isinstance(model, HoppingModel), 'Invalid model type'

        self.model = model
        self.X, self.Y = self.model.X, self.model.Y

        self._initGUI()

        self.model.initialise()

        # setup threads
        self.threads = []
        self.threads.append(Thread(self.tick))

        # setup recording
        self.recording = record
        if record:
            # force clean directory
            if os.path.exists(self.record_dir):
                shutil.rmtree(self.record_dir)
            os.makedirs(self.record_dir)
            self.rind = 0   # record index
            self.fps = fps

            # setup threads
            self.threads.append(Thread(self.record))

    def _initGUI(self):
        '''Initialise the animator window'''

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._drawDBs()

        self.setGeometry(100, 100, self.WINX, self.WINY)
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

    def record(self):
        '''Record the QGraphicsScene at the given fps'''

        assert self.fps>0 and self.fps<=1000, 'Invalid fps'

        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        image = QImage(self.scene.sceneRect().size().toSize(), QImage.Format_ARGB32)
        image.fill(self.bgcol)

        painter = QPainter(image)
        self.scene.render(painter)
        image.save(os.path.join(self.record_dir, 'grab{0:06d}.png'.format(self.rind)))
        painter.end()
        self.rind += 1

        self.rec_timer = QTimer()
        self.rec_timer.timeout.connect(self.record)
        self.rec_timer.start(int(1000./self.fps))

    def compile(self):
        '''compile the recording directory into a video'''

        os.chdir(self.record_dir)
        os.system("ffmpeg -r {0} -f image2 -i grab%06d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p ../rec.mp4".format(int(self.fps)))
        os.chdir('..')
        shutil.rmtree(self.record_dir)

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

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            if self.recording:
                self.compile()
            self.close()


if __name__ == '__main__':

    import sys
    sys.setrecursionlimit(50)

    line = [8, 10, 15, 17, 22, 24, 29, 31, 36, 38]
    #line.insert(0, line[0]-7)
    line.append(line[-1]+7)

    _or = [(0,0,0),(2,1,0),(6,1,0),(8,0,0),(4,3,0),(4,4,1),(4,6,0)]
    _or.append((-2,-1,0))
    _or.append((10,-1,0))

    def QCA(N):
        qca = []
        for n in range(N):
            x0 = 10*n
            qca += [(x0,0,1), (x0+3,0,1), (x0,2,0), (x0+3,2,0)]
        qca.append((-4,0,1))
        return qca

    device = QCA(7)

    # NOTE: recording starts immediately if record==True. Press 'Q' to quit and
    #       compile temp files into an animation ::'./rec.mp4'
    # model = HoppingModel(device, model='marcus', record=True)
    model = HoppingModel(device, model='marcus')
    # model.fixElectronCount(5)
    #model.addChannel('bulk')

    app = QApplication(sys.argv)
    animator = HoppingAnimator(model)

    for thread in animator.threads:
        thread.start()

    animator.show()
    sys.exit(app.exec_())
