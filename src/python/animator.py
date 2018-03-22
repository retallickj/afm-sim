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

from PyQt5.QtCore import (Qt, QTimer, QThread, pyqtSignal, QDateTime)
from PyQt5.QtGui import (QPen, QBrush, QColor, QPainter, QImage)
from PyQt5.QtWidgets import *

from hopper import HoppingModel


_SF = 50     # scale factor

class Thread(QThread):
    def __init__(self, func):
        super(Thread, self).__init__()
        self.func = func

    def run(self):
        ''' '''
        self.func()
        self.exec_()


class DB(QGraphicsEllipseItem):

    pen     = QPen(QColor("white"), .2*_SF)     # DB edge pen
    bgpen   = QPen(Qt.darkGray, .1*_SF, Qt.DotLine)

    pfill   = QBrush(QColor("orange"))     # charged DB for fixed perturbers
    fill    = QBrush(Qt.green)      # charged DB fill color
    nofill  = QBrush(Qt.NoBrush)    # uncharged DB fill color

    D = 1.8*_SF              # dot diameter

    def __init__(self, x, y, n=-1, bg=False, parent=None):
        super(DB, self).__init__(_SF*x, _SF*y, self.D, self.D, parent=parent)
        self.xx, self.yy, self.n = x, y, n
        self.setPen(self.bgpen if bg else self.pen)
        self.setCharge(False)
        self.bg = bg
        if not bg:
            self.setZValue(2)

    def setCharge(self, charged):
        '''Set the charge state of the DB'''
        self.charged = charged
        if charged:
            brush = self.pfill if self.bg else self.fill
        else:
            brush = self.nofill

        self.setBrush(brush)



class Tracker(QGraphicsEllipseItem):

    pen = QPen(Qt.red, .2*_SF, Qt.DotLine)
    D = DB.D*1.5
    dd = .5*(D-DB.D)

    def __init__(self, parent=None):
        super(Tracker, self).__init__(0, 0, self.D, self.D, parent=parent)
        self.setZValue(1)
        self.setPen(self.pen)
        self.setBrush(QBrush(Qt.NoBrush))
        self.hide()

    def track(self, db):
        self.setPos(_SF*db.xx - self.dd , _SF*db.yy - self.dd)
        self.show()



class Tip(QGraphicsEllipseItem):

    pen     = QPen(QColor("blue"), .2*_SF)
    fill    = QBrush(QColor("blue"))

    D = 2*_SF

    def __init__(self, parent=None):
        super(Tip, self).__init__(0, 0, self.D, self.D, parent=parent)
        self.setPen(self.pen)
        self.setBrush(self.fill)
        self.setZValue(3)
        #self.hide()



class FieldSlider(QHBoxLayout):
    '''Container for parameter selected by a QSlider'''

    def __init__(self, txt, parent=None):
        super(FieldSlider, self).__init__(parent)

        self.txt = QLabel(txt)
        self.out = QLabel()
        self.fval = lambda n: n
        self.func = lambda x: None

        self.initGUI()

    def initGUI(self):

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.valueChanged)
        self.slider.sliderReleased.connect(self.sliderReleased)

        self.addWidget(self.txt, stretch=4)
        self.addWidget(self.slider, stretch=40)
        self.addWidget(self.out, stretch=4)

    def setBounds(self, lo, hi, inc, val):

        self.lo, self.hi, self.inc = lo, hi, inc
        self.fval = lambda n: lo+n*self.inc
        self.val = val

        self.slider.setMinimum(0)
        self.slider.setMaximum(round((hi-lo)*1./inc))
        self.slider.setValue(round((val-lo)/inc))

    def setValue(self, val):
        self.val = val
        self.setValue(round((val-self.lo)/self.inc))

    def setToolTip(self, txt):
        self.txt.setToolTip(txt)

    # event handling
    def valueChanged(self):
        self.val = self.fval(self.slider.value())
        self.out.setText('{0:.3f}'.format(self.val))

    def sliderReleased(self):
        self.func(self.val)


class FieldEdit(QHBoxLayout):
    '''Container for parameter selected by a QLineEdit'''

    def __init__(self, parent=None):
        super(FieldEdit, self).__init__(parent)


class DockWidget(QDockWidget):
    ''' '''

    WIDTH = 200

    def __init__(self, parent=None):
        super(DockWidget, self).__init__(parent)

        self.initGUI()

    def initGUI(self):

        self.setMinimumWidth(self.WIDTH)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        widget = QWidget(self)
        self.vbox = QVBoxLayout(widget)
        self.vbox.setAlignment(Qt.AlignTop)
        self.setWidget(widget)

        self.hide()

    def addSeparator(self):
        '''Add a horizonal separator line to the dock layout'''

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self.vbox.addWidget(sep)

    def addText(self, txt):
        '''Add a line of text to the dock'''

        label = QLabel(txt)
        self.vbox.addWidget(label)

    def addSlider(self, txt, lo, hi, inc, val, func, ttip=''):
        '''Add a slider controlled parameter to the Dock Widget

        inputs:
            txt     : Label of the slider
            lo      : Lowest value of the slider
            hi      : Highest value of the slider
            inc     : Increment between slider ticks
            func    : When slider is updated to x, func(x) called
        '''

        slider = FieldSlider(txt)
        slider.setBounds(lo, hi, inc, val)
        slider.setToolTip(ttip)
        slider.func = func

        self.vbox.addLayout(slider)
        return slider

    def addWidget(self, widget, stretch=-1):
        self.vbox.addWidget(widget, stretch=stretch)

    def addLayout(self, layout, stretch=-1):
        self.vbox.addLayout(layout, stretch=stretch)








class HoppingAnimator(QGraphicsView):
    ''' '''

    # lattice parameters
    a = 3.84    # lattice vector in x, angstroms    (intra dimer row)
    b = 7.68    # lattice vector in y, angstroms    (inter dimer row)
    c = 2.25    # dimer pair separation, angstroms

    rate = 1   # speed-up factor

    xpad, ypad = 6, 3

    bgcol = QColor(29, 35, 56)  # background color
    record_dir = './.temp_rec/'

    signal_tick = pyqtSignal()
    signal_dbtrack = pyqtSignal(int)

    def __init__(self, model, record=False, fps=30):
        '''Initialise the HoppingAnimator instance for the given DB positions.
        X should be formatted as for HoppingModel'''

        super(HoppingAnimator, self).__init__()

        assert isinstance(model, HoppingModel), 'Invalid model type'

        self.model = model
        self.X, self.Y = self.model.X, self.model.Y

        self.bulk = self.model.getChannel('bulk')
        self.tip = self.model.getChannel('tip')

        self._initGUI()

        self.model.initialise()

        # setup threads
        self.timers = []
        self.threads = []
        self.threads.append(Thread(self.tick))

        self.tick_timer = QTimer()
        self.tick_timer.timeout.connect(self.tick)
        self.timers.append(self.tick_timer)

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
            self.record_timer = QTimer()
            self.record_timer.timeout.connect(self.record)

        self.paused = False
        self.rtimes = [0,]*len(self.timers)

        self.panning = False
        self.panx, self.pany = 0., 0.
        self.path = []

    def _initGUI(self):
        '''Initialise the animator window'''

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._drawDBs()
        self.tracker = Tracker()
        self.scene.addItem(self.tracker)

        if self.tip is not None:
            self.tip_item = Tip()
            self.scene.addItem(self.tip_item)

        self.setBackgroundBrush(QBrush(self.bgcol, Qt.SolidPattern))
        self.setWindowTitle('Hopping Animator')

        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

    def passControls(self, dock):
        '''Add control fields to the given DockWidget'''


        # hopping controls

        dock.addSeparator()
        dock.addText('Hopping Model:')

        if self.model.fixed_pop:
            val = self.model.Nel
            func = lambda N: self.setParFunc(self.model.fixElectronCount, N)
            dock.addSlider('N', 0, self.model.N, 1, val, func,
            'Number of electrons in the surface.')

        # hopping model parameters

        mdl = self.model.model
        if mdl.__class__.__name__ == 'MarcusModel':
            val = self.model.model.lamb
            func = lambda v: self.setParFunc(self.model.model.setLambda, v)
            dock.addSlider('lambda', 0.01, 0.2, .001, val, func,
                'Reorganization Energy: self-trapping for DB occupation')

        elif mdl.__class__.__name__ == 'VRHModel':
            val, func = mdl.lamb, lambda v: self.setPar(mdl, 'lamb', v)
            dock.addSlider('lambda', 0.01, 0.2, .001, val, func,
                'Reorganization Energy: self-trapping for DB occupation')

            val = np.log10(mdl.nu)
            func = lambda v: self.setParFunc(mdl.setNu, 10**v)
            dock.addSlider('log(nu)', -1, 5, .5, val, func,
                'Hopping rate prefactor')

            val, func = mdl.ktd, lambda v: self.setPar(mdl, 'ktd', v)
            dock.addSlider('ktd', 0.01, 1, .01, val, func,
                'Inverse kT attenuation, lower increases effective temperature')


        # bulk controls
        if self.bulk is not None:
            dock.addSeparator()
            dock.addText('Bulk properties')

            val, func = self.bulk.lamb, lambda v: self.setPar(self.bulk, 'lamb', v)
            dock.addSlider('lambda', .01, .4, .005, val, func,
                'Self-Trapping energy for surface-bulk hopping')

            val, func = self.bulk.mu, lambda v: self.setPar(self.bulk, 'mu', v)
            dock.addSlider('mu', -.3, .3, .01, val, func,
                'Chemical Potential: local potential at which charges will hop \
                between the bulk and the surface')

            val = np.log10(self.bulk.nu)
            func = lambda v: self.setPar(self.bulk, 'nu', 10**v)
            dock.addSlider('log(nu)', -1, 5, .5, val, func,
                'Maximum hopping rate between the bulk and the surface')

        # tip controls
        if self.tip is not None:
            dock.addSeparator()
            dock.addText('Tip properties')

            val, func = self.tip.scale, lambda v: self.setPar(self.tip, 'scale', v)
            dock.addSlider('scale', 0., 1., .01, val, func,
                'Attenuation for tip contribution to the energy calculation')

            val, func = self.tip.epsr, lambda v: self.setPar(self.tip, 'epsr', v)
            dock.addSlider('epsr', 1., 10., .2, val, func,
                'Relative permittivity for image charge interactions')

            val, func = self.tip.lamb, lambda v: self.setPar(self.tip, 'lamb', v)
            dock.addSlider('lambda', 0.01, .4, .005, val, func,
                'Self-Trapping energy for surface-tip hopping')

            val, func = self.tip.mu, lambda v: self.setPar(self.tip, 'mu', v)
            dock.addSlider('mu', -1., 1., .01, val, func,
                'Chemical Potential: local potential at which charges will hop \
                between the tip and surface')

            val = np.log10(self.tip.TR0)
            func = lambda v: self.setPar(self.tip, 'TR0', 10**v)
            dock.addSlider('log(nu)', -1, 5, .5, val, func,
                'Maximum hopping rate between the tip and surface')


            val = 1e3*self.tip.tipH
            func = lambda h: self.setParFunc(self.tip.setHeight, 1e-3*h)
            dock.addSlider('H', 100, 1000, 10, val, func,
                'Tip height in pm')

            val = self.tip.tipR
            func = lambda R: self.setParFunc(self.tip.setRadius, R)
            dock.addSlider('R', 1, 50, 1, val, func,
                'Tip radius in nm')

            val, func = self.tip.rate, lambda v: self.setPar(self.tip, 'rate', v)
            dock.addSlider('rate', 1., 50., .5, val, func,
                'Tip scan rate in nm/s')

        # animator controls
        if True:
            dock.addSeparator()
            dock.addText('Animation controls')

            val = np.log10(self.rate)
            func = lambda v: self.setPar(self, 'rate', 10**v, tc=2)
            dock.addSlider('log(rate)', -3., 3., .5, val, func,
                'Speed-up factor for the animation.')

        # functionality
        if self.tip is not None:

            dock.addSeparator()
            dock.addText('Tip Programs:')

            self.pad_edit = QLineEdit('2')
            self.pad_edit.setToolTip('Padding size, in angstroms')
            self.nline_edit = QLineEdit('200')
            self.nline_edit.setToolTip('Number of lines in the 2D scan')

            hb = QHBoxLayout()
            hb.addWidget(QLabel('Padding:'), stretch=1)
            hb.addWidget(self.pad_edit, stretch=2)
            dock.addLayout(hb)

            # full 2D scan
            hb = QHBoxLayout()
            hb.addWidget(QLabel('Lines:'), stretch=1)
            hb.addWidget(self.nline_edit, stretch=2)
            dock.addLayout(hb)

            def newButton(slot, txt='Run', ttip=''):
                button = QPushButton(txt)
                button.clicked.connect(slot)
                button.setToolTip(ttip)
                return button

            hb = QHBoxLayout()
            hb.addWidget(newButton(self.lineScan, 'Line',
                'Line scan path at nearest DB row'), stretch=1)
            hb.addWidget(newButton(self.fullScan, 'Full',
                'Full 2D scan with the given number of lines'), stretch=1)
            dock.addLayout(hb)



    def setPar(self, obj, attr, val, tc=1):
        setattr(obj, attr, val)
        for _ in range(tc):
            self.tick()

    def setParFunc(self, func, val, tc=1):
        '''Set a parameter through an accessor function'''
        func(val)
        self.tick()

    def setTipHeight(self, H):
        self.tip.setHeight(H)
        self.tick()


    def lineScan(self):
        '''Start a line scan at the db row closest to the current tip position'''

        f = .586
        pad = float(self.pad_edit.text())
        lo_x, hi_x = self.a*np.min(self.X), self.a*np.max(self.X)
        y0 = round(self.tip.tipY*10/self.a,1)
        n0, d = divmod(y0+.7, 2)

        y = .1*(n0*self.b+(d>1)*self.c)
        path = [(.1*(lo_x-pad), y), (.1*(hi_x+pad), y)]
        self.tip.setScan(path, loop=True)
        self.path = []
        self.tick()

    def fullScan(self):

        # get scan bounds

        pad = float(self.pad_edit.text())

        lo_x, hi_x = self.a*np.min(self.X), self.a*np.max(self.X)
        lo_y, hi_y = self.b*np.min(self.Y), self.b*np.max(self.Y)

        lo_x, lo_y = .1*(lo_x-pad), .1*(lo_y-pad)
        hi_x, hi_y = .1*(hi_x+pad), .1*(hi_y+pad)
        print(lo_x, hi_x, lo_y, hi_y)

        nlines = int(self.nline_edit.text())
        assert nlines>1, '2D scan must contain at least 2 lines'
        path = []
        dy, y = (hi_y-lo_y)/(nlines-1), lo_y
        for line in range(nlines):
            path += [(lo_x,y), (hi_x,y), (hi_x, y+dy)]
            y += dy
            lo_x, hi_x = hi_x, lo_x
        self.tip.setScan(path, loop=True)
        self.path = []
        self.tick()





    def _drawDBs(self):
        '''Draw all the DBs for the animator'''

        # background
        X = np.arange(np.min(self.X)-self.xpad, np.max(self.X)+self.xpad+1)
        Y = np.arange(round(np.min(self.Y))-self.ypad, round(np.max(self.Y))+self.ypad+1)

        f = self.c/self.b
        for x,y in product(X,Y):
            self.scene.addItem(DB(self.a*x, self.b*y, bg=True))
            self.scene.addItem(DB(self.a*x, self.b*(y+f), bg=True))

        # foreground
        self.dbs = []
        for n, (x,y) in enumerate(zip(self.X, self.Y)):
            self.dbs.append(DB(self.a*x,self.b*y, n=n))
            self.scene.addItem(self.dbs[-1])


    def screencapture(self, fname):
        '''Save a screenshot of the QGraphicsScene and save it to the given
        filename'''

        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        image = QImage(self.scene.sceneRect().size().toSize(), QImage.Format_ARGB32)
        image.fill(self.bgcol)

        painter = QPainter(image)
        self.scene.render(painter)
        image.save(fname)
        painter.end()


    def record(self):
        '''Record the QGraphicsScene at the given fps'''

        assert self.fps>0 and self.fps<=1000, 'Invalid fps'

        fname = os.path.join(self.record_dir, 'grab{0:06d}.png'.format(self.rind))
        self.screencapture(fname)

        self.rind += 1
        self.record_timer.start(int(1000./self.fps))

    def compile(self):
        '''compile the recording directory into a video'''

        os.chdir(self.record_dir)
        os.system("ffmpeg -r {0} -f image2 -i grab%06d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p ../rec.mp4".format(int(self.fps)))
        os.chdir('..')
        shutil.rmtree(self.record_dir)

    def updateTip(self):
        '''update the location of the tip graphic'''
        self.tip_item.setPos(self.tip.tipX*10*_SF, self.tip.tipY*10*_SF)
        self.update()

    def setTipTarget(self, x, y):
        if self.tip is not None:
            self.tip.setTarget(.1*x, .1*y)
            self.tick()

    def pause(self):
        '''Pause/Resume all timers'''
        for i, t in enumerate(self.timers):
            if self.paused:
                t.start(self.rtimes[i])
            else:
                self.rtimes[i] = t.remainingTime()
                t.stop()
        self.paused = not self.paused

    def tick(self):
        ''' '''

        if not self.paused:
            # draw last state
            for i,c in enumerate(self.model.charge):
                self.dbs[i].setCharge(c)

            if self.tip is not None:
                self.updateTip()

            self.update()
            self.signal_tick.emit()

            # update hopper state
            milli = 1.
            while milli>0:
                dt = self.model.step()
                millis = dt*1000./self.rate
                milli -= millis

            self.tick_timer.start(min(max(int(millis),1), 10000))


    def zoomExtents(self):
        '''Scale view to contain all items in the scene'''
        rect = self.scene.itemsBoundingRect()
        self.fitInView(rect, Qt.KeepAspectRatio)
        self.scale(2,2)

    def mousePressEvent(self, e):
        super(HoppingAnimator, self).mousePressEvent(e)

        if e.button() == Qt.MiddleButton:
            self.panning = True
            self.panx, self.pany = e.x(), e.y()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()

        elif e.button() == Qt.LeftButton:

            # path lists
            if self.tip is not None:
                if e.modifiers() & Qt.ShiftModifier:
                    pos, dp = self.mapToScene(e.pos())/_SF, .5*self.tip_item.D/_SF
                    if e.modifiers() & Qt.ControlModifier:
                        self.path.append((.1*(pos.x()-dp), .1*(pos.y()-dp)))
                    else:
                        self.setTipTarget(pos.x()-dp, pos.y()-dp)
                    e.accept()
                    return

            item = self.itemAt(e.pos())
            if isinstance(item, DB) and item.bg:
                item.setCharge(not item.charged)
                self.model.addCharge(item.xx, item.yy, pos=item.charged)

        elif e.button() == Qt.RightButton:
            item = self.itemAt(e.pos())
            if isinstance(item, DB):
                self.signal_dbtrack.emit(item.n)

    def mouseMoveEvent(self, e):
        if self.panning:
            hsb, vsb = self.horizontalScrollBar(), self.verticalScrollBar()
            hsb.setValue(hsb.value()+self.panx-e.x())
            vsb.setValue(vsb.value()+self.pany-e.y())
            self.panx, self.pany = e.x(), e.y()
            e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.setCursor(Qt.ArrowCursor)
            e.accept()

    def mouseDoubleClickEvent(self, e):
        self.mousePressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Control:
            if self.path:
                self.tip.setScan(self.path, loop=True)
                self.path = []
                self.tick()




class MainWindow(QMainWindow):
    ''' '''

    WINX = 1400     # window width
    WINY = 1000      # window height

    ZOOM = .1

    def __init__(self, model, record=False, fps=30):
        ''' '''
        super(MainWindow, self).__init__()

        self.record = record
        self.fps = fps

        self.dbn = -1       # tracked db index
        self.model = model

        self.animator = HoppingAnimator(model, record=record, fps=fps)
        self.animator.signal_tick.connect(self.tickSlot)
        self.animator.signal_dbtrack.connect(self.trackDB)

        self.initGUI()
        self.createDock()

        for thread in self.animator.threads:
            thread.start()

    def initGUI(self):
        ''' '''

        self.setGeometry(100, 100, self.WINX, self.WINY)
        self.setCentralWidget(self.animator)
        self.animator.zoomExtents()

    def createDock(self):
        '''Create the dock widget for simulation options'''

        self.dock = DockWidget(self)

        self.beff = QLabel()
        self.dock.addWidget(self.beff)

        self.ltime = QLabel()
        self.dock.addWidget(self.ltime)

        self.ecount = QLabel()
        self.dock.addWidget(self.ecount)

        self.animator.passControls(self.dock)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

    def tickSlot(self):
        self.ecount.setText('Number of Electrons: {0}'.format(self.model.Nel))
        self.echoDB()

    def trackDB(self, n):
        if self.dbn == n:
            n = -1
        self.dbn = n
        if n<0:
            self.animator.tracker.hide()
        else:
            self.animator.tracker.track(self.animator.dbs[n])
        self.echoDB()


    def echoDB(self):
        if self.dbn < 0:
            self.beff.setText('')
            self.ltime.setText('')
        else:
            self.beff.setText('DB-Beff: {0:.3f}'.format(self.model.beff[self.dbn]))
            self.ltime.setText('Lifetime: {0:.3f}'.format(
                                    self.model.lifetimes[self.dbn]))


    def keyPressEvent(self, e):

        if e.key() == Qt.Key_Q:
            if self.record:
                self.animator.compile()
            self.close()
        elif e.key() == Qt.Key_O:
            self.dock.setVisible(not self.dock.isVisible())
        elif e.key() == Qt.Key_Space:
            self.animator.tick()
        elif e.key() in [Qt.Key_Plus, Qt.Key_Equal]:
            zfact = 1+self.ZOOM
            self.animator.scale(zfact,zfact)
        elif e.key() == Qt.Key_Minus:
            zfact = 1-self.ZOOM
            self.animator.scale(zfact, zfact)
        elif e.key() == Qt.Key_S:
            fname = QDateTime.currentDateTime().toString('yyyyMMdd-hhmmss.png')
            fname = os.path.join('.', fname)
            print('Screenshot saved to: {0}'.format(os.path.normpath(fname)))
            self.animator.screencapture(fname)
        elif e.key() == Qt.Key_P:
            self.animator.pause()
        elif e.key() == Qt.Key_L:
            self.animator.lineScan()




if __name__ == '__main__':

    import sys

    line = [8, 10, 15, 17]
    line.insert(0, line[0]-7)
    line.append(line[-1]+7)

    pair = lambda n: [0, n]

    _or = [(0,0,0),(2,1,0),(6,1,0),(8,0,0),(4,3,0),(4,4,1)]
    _or.append((4,6,0))
    # _or.append((-2,-1,0))
    # _or.append((10,-1,0))

    def QCA(N):
        qca = []
        for n in range(N):
            x0 = 10*n
            qca += [(x0,0,1), (x0+3,0,1), (x0,2,0), (x0+3,2,0)]
        qca.append((-4,0,1))
        return qca

    def wire(N):
        wire = []
        dx, dp, x = 2, 6, 0
        for n in range(N):
            wire += [(x,0,0), (x+dx,0,0)]
            x += dp
        # perturbers
        return wire

    device = line

    # NOTE: recording starts immediately if record==True. Press 'Q' to quit and
    #       compile temp files into an animation ::'./rec.mp4'
    # model = HoppingModel(device, model='marcus', record=True)
    model = HoppingModel(device, model='VRH')
    model.addChannel('bulk')
    model.addChannel('tip')
    model.fixElectronCount(3)

    app = QApplication(sys.argv)
    mw = MainWindow(model)

    mw.show()
    sys.exit(app.exec_())
