#!/usr/bin/env python
# encoding: utf-8

'''
Real-time animation of DB arrangements
'''

__author__      = 'Jake Retallick'
__copyright__   = 'MIT License'
__version__     = '1.2'
__date__        = '2018-03-26'  # last update

import shutil, os
import numpy as np
from itertools import product
import json
from subprocess import Popen

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtPrintSupport import QPrinter

from hopper import HoppingModel

from timeit import default_timer as wall


_SF = 50     # scale factor


class Thread(QThread):
    def __init__(self, func):
        super(Thread, self).__init__()
        self.func = func

    def run(self):
        ''' '''
        self.func()
        self.exec_()


class Logger(object):
    ''' '''

    _log_file = 'log.dat'
    _temp_file = '.temp_log'

    # TODO: start the Viewer on a key press

    def __init__(self, root, view=True):
        ''' '''

        print('Logger started')
        self.root = os.path.abspath(root)
        self.log_fn = os.path.join(self.root, self._log_file)

        if not os.path.exists(self.root):
            print('Logger creating directory: {0}'.format(self.root))
            os.makedirs(self.root)

        self.temp_fn = os.path.join(self.root, self._temp_file)
        self.view, self.viewing = view, False
        self.P = None

    def cleanup(self):

        print('Logger being closed')

        # close viewer log if open
        if hasattr(self, 'view_fp'):
            self.view_fp.close()

        # close the Viewer process if open
        if self.P is None:
            self.P.terminate()

    def log(self, data):
        '''Log the given dictionary to the next file'''

        with open(self.temp_fn, 'w') as fp:
            json.dump(data, fp, indent=1)

        # keep trying to remove file if it exists
        while True:
            if os.path.exists(self.log_fn):
                try:
                    os.remove(self.log_fn)
                    break
                except:
                    pass

        os.rename(self.temp_fn, self.log_fn)

        if self.view and not self.viewing:
            self.startViewer()

    def startViewer(self):
        ''' '''
        print('Starting Viewer')
        self.viewing = True
        self.view_fp = open(os.path.join(self.root, 'stdout'), 'a')
        self.P = Popen(['python3', 'lineview.py', self.log_fn],
                    stdout=self.view_fp, stderr=self.view_fp)


class ClockGradient(QLinearGradient):

    colors = [ QColor(0,0,255,50),    # minimum clocking field color shift
             QColor(0,0,0,0),     # mid value color shift
             QColor(255,0,0,50)     # maximum clocking field color shift
            ]

    steps = 30
    mixfact = .1

    nwaves = 1

    def __init__(self, clock, rect):
        '''Initialise the clocking gradient display.

        inputs:
            clock   : object with clock.length and clock.waveform(x)
            rect    : bounding rect of object to display gradient
        '''
        super(ClockGradient, self).__init__()

        self.clock = clock
        self.setSpread(QGradient.RepeatSpread)

        self.setStart(rect.left(),0)
        self.setFinalStop(rect.right(),0)

        self.prepareClock()


    def prepareClock(self):
        '''Set up the clocking waveform gradient stops'''

        xx = np.linspace(0,1,self.steps)
        yy = self.clock.waveform(self.nwaves*xx*self.clock.length, 0)

        mn, mx = np.min(yy), np.max(yy)
        interp = lambda v: self._interpolate((v-mn)/(mx-mn))

        for x,y in zip(xx,yy):
            self.setColorAt(x, interp(y))

    def updateColors(self):
        '''Track the clock by adjusting the stop coordinates'''

        # for now, assume waveform extrema at +- wlength/4 at t=0
        xx = self.clock.length*np.array([0,self.nwaves])

        # advance minimum position by current time
        xx += self.clock.length*self.clock.wf_f*self.clock.t
        if xx[0] > self.clock.length:
            xx -= self.clock.length

        self.setStart(xx[0]*_SF, 0)
        self.setFinalStop(xx[1]*_SF, 0)

    def _interpolate(self,v):
        '''Interpolate the gradient color for a value between 0 and 1'''

        if v<.5:
            c1,c2 = self.colors[0], self.colors[1]
            f1,f2 = .5-v, v
        else:
            c1,c2 = self.colors[1], self.colors[2]
            f1,f2 = 1-v, v-.5

        c = 2*(f1*np.array(c1.getRgb())+f2*np.array(c2.getRgb()))
        return QColor(*c)


class ClockPublish:
    '''Handler for creating formatted screenshots including the
    clocking fields for publishing'''

    def __init__(self, fname, clock):
        '''
        inputs:
            fname : name of svg screenshot of
            clock :
        '''
        pass




class DB(QGraphicsEllipseItem):

    D = 1.8*_SF              # dot diameter

    normal = {
     # DB edge pens
     'pen':     QPen(QColor("white"), .2*_SF),   # DB edge pen
     'bgpen':   QPen(QColor(255,255,255,50), .2*_SF, Qt.DotLine),

     # fill color
     'pfill':   QBrush(QColor("orange")),   # fixed perturber fill
     'fill':    QBrush(Qt.green),           # charged DB fill
     'nofill':  QBrush(Qt.NoBrush),         # uncharged DB fill

     # diameters
     'dbD':     D,      # DB diameter
     'latD':    D       # lattice dot diameter
    }

    capture = {
     # DB edge pens
     'pen':     QPen(QColor("black"), .2*_SF),   # DB edge pen
     'bgpen':   QPen(QColor("gray"), .2*_SF, Qt.DotLine),

     # fill color
     'pfill':   QBrush(QColor("orange")),   # fixed perturber fill
     'fill':    QBrush(Qt.red),             # charged DB fill
     'nofill':  QBrush(Qt.NoBrush),         # uncharged DB fill

     # diameters
     'dbD':     1.5*D,  # DB diameter
     'latD':    0.7*D   # lattice dot diameter
    }


    def __init__(self, x, y, n=-1, bg=False, parent=None):
        super(DB, self).__init__(_SF*x, _SF*y, self.D, self.D, parent=parent)
        self.xx, self.yy, self.n = x, y, n
        self.setPen(self.normal['bgpen'] if bg else self.normal['pen'])
        self.setCharge(False)
        self.bg = bg
        self._setDiameter(self.normal['latD' if not bg else 'dbD'])
        if not bg:
            self.setZValue(2)
        self.state = {}

    def setCharge(self, charged):
        '''Set the charge state of the DB'''
        self.charged = charged
        if charged:
            brush = self.normal['pfill'] if self.bg else self.normal['fill']
        else:
            brush = self.normal['nofill']

        self.setBrush(brush)

    def setCapture(self):
        '''Set paint parameters for capture mode'''

        D = self.capture['dbD' if (not self.bg or self.charged) else 'latD']
        self.state['D'] = self._setDiameter(D)
        self.state['pen'] = self.pen()
        self.state['brush'] = self.brush()

        pen = self.capture['bgpen' if self.bg else 'pen']
        self.setPen(pen)

        if self.charged:
            brush = self.capture['pfill' if self.bg else 'fill']
        else:
            brush = self.capture['nofill']
        self.setBrush(brush)

    def unsetCapture(self):
        '''Restore paint state after capture mode'''

        self._setDiameter(self.state['D'])
        self.setPen(self.state['pen'])
        self.setBrush(self.state['brush'])

    def _setDiameter(self, D):
        '''Update the dot diameter and return the old one'''

        rect = self.boundingRect()
        d = rect.width() - self.pen().width()
        center = rect.center()

        rect.setSize(QSizeF(D,D))
        rect.moveCenter(center)
        self.setRect(rect)

        return d



class SnapTarget(QGraphicsEllipseItem):

    pen     = QPen(Qt.NoPen)
    fill    = QBrush(Qt.yellow)

    D = .6*DB.D
    dd = .5*(DB.D-D)
    r0 = .2*_SF

    def __init__(self, parent=None):
        super(SnapTarget, self).__init__(0,0,self.D, self.D, parent=parent)

        self.setZValue(2)
        self.setPen(self.pen)
        self.setBrush(self.fill)
        self.hide()

        self.target = None
        self.state = {}

    def setTarget(self, target=None):
        if target is None:
            self.hide()
        elif isinstance(target, DB):
            self.setPos(_SF*target.xx + self.dd, _SF*target.yy + self.dd)
            self.show()
        else:
            raise KeyError('Snap: Invalid target type...')
        rect = self.target.boundingRect() if self.target else None
        self.target = target
        return rect

    def setCapture(self):
        self.state['vis'] = self.isVisible()
        self.hide()

    def unsetCapture(self):
        if self.state['vis']:
            self.show()


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
        self.state = {}

    def track(self, db):
        self.setPos(_SF*db.xx - self.dd , _SF*db.yy - self.dd)
        self.show()

    def setCapture(self):
        self.state['vis'] = self.isVisible()
        self.hide()

    def unsetCapture(self):
        if self.state['vis']:
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

    xpad, ypad = 8, 4

    bgcol = QColor(29, 35, 56)  # background color
    record_dir = os.path.join('.', '.temp_rec/')

    logging = False     # set True for LineView animation
    log_dir = os.path.join('.', '.temp/')

    zoom_rate = .1
    zoom_bounds = [.01, 5.]

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
        self.clock = self.model.getChannel('clock')

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

        self.logger = Logger(self.log_dir) if self.logging else None

        self.panning = False
        self.panx, self.pany = 0., 0.
        self.path = []

        self.dbn = -1
        self.state = {}

    def cleanup(self):
        #super(HoppingAnimator, self).__del__()
        if self.logger:
            self.logger.cleanup()
        self.model.cleanup()

    def _initGUI(self):
        '''Initialise the animator window'''

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setMouseTracking(True)

        self._drawDBs()
        self.tracker = Tracker()
        self.snapper = SnapTarget()
        self.scene.addItem(self.tracker)
        self.scene.addItem(self.snapper)

        self.setStyleSheet('QScrollBar{background:white;}\
        QScrollBar::handle{background:gray}')

        self.old_pos = QPoint(0,0)

        if self.tip is not None:
            self.tip_item = Tip()
            self.scene.addItem(self.tip_item)

        if self.clock is not None:
            self.gradient = ClockGradient(self.clock, self.scene.sceneRect())
            self.scene.setForegroundBrush(self.gradient)
        self.scene.setBackgroundBrush(QBrush(self.bgcol, Qt.SolidPattern))
        self.setWindowTitle('Hopping Animator')

        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)

        self.zoomExtents()
        self.update()

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

        val = mdl.lamb
        func = lambda v: self.setParFunc(mdl.setLambda, v)
        dock.addSlider('lambda', 0.001, 0.3, .001, val, func,
            'Reoranization Energy: thermal barrier for hopping')

        val = np.log10(mdl.prefact)
        func = lambda v: self.setParFunc(mdl.setPrefactor, 10**v)
        dock.addSlider('log(nu)', -5, 15, .2, val, func,
            'Hopping rate prefactor')

        val = mdl.alph
        func = lambda v: self.setParFunc(mdl.setAttenuation, v)
        dock.addSlider('alpha', 1, 1e2, .1, val, func,
            'Attenuation length, in angstroms')

        val, func = mdl.ktf, lambda v: self.setPar(mdl, 'ktf', v)
        dock.addSlider('ktf', 1., 100., .1, val, func,
            'Thermal broadening factor, multiplier for thermal energy')

        # bulk controls
        if self.bulk is not None:
            dock.addSeparator()
            dock.addText('Bulk properties')

            val, func = self.bulk.lamb, lambda v: self.setPar(self.bulk, 'lamb', v)
            dock.addSlider('lambda', .01, .4, .005, val, func,
                'Self-Trapping energy for surface-bulk hopping')

            val, func = self.bulk.mu, lambda v: self.setPar(self.bulk, 'mu', v)
            dock.addSlider('mu', 0., .5, .01, val, func,
                'Chemical Potential: local potential at which charges will hop between the bulk and the surface')

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

            val = self.tip.tipR1key
            func = lambda R: self.setParFunc(
                        lambda r: self.tip.setRadius(icibb=r), R)
            dock.addSlider('ICIBB R', 1, 50, 1, val, func,
                'Tip radius in nm for ICIBB')

            val = self.tip.tipR2
            func = lambda R: self.setParFunc(
                        lambda r: self.tip.setRadius(tibb=r), R)
            dock.addSlider('TIBB R', 1, 200, 1, val, func,
                'Tip radius in nm for TIBB')

            val, func = self.tip.rate, lambda v: self.setPar(self.tip, 'rate', v)
            dock.addSlider('rate', 1., 50., .5, val, func,
                'Tip scan rate in nm/s')

        if self.clock is not None:

            dock.addSeparator()
            dock.addText('Clocking Field')

            val = np.log10(self.clock.length)-1
            func = lambda v: self.setPar(self.clock, 'length', 10**(v+1))
            dock.addSlider('log(length)', 0, 3, .1, val, func,
                'Wavelength, in nm')

            val = np.log10(self.clock.wf_f)
            func = lambda v: self.setPar(self.clock, 'wf_f', 10**v)
            dock.addSlider('Frequency', -2, 3, .1, val, func,
                'Frequency, in Hz')

            val, func = self.clock.wf_A, lambda v: self.setPar(self.clock, 'wf_A', v)
            dock.addSlider('Amplitude', 0, .5, .01, val, func,
                'Amplitude, in eV')

            val, func = self.clock.wf_0, lambda v: self.setPar(self.clock, 'wf_0', v)
            dock.addSlider('Offset', -.2, .2, .01, val, func,
                'Offset, in eV')

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
        '''Time stepping protocol'''

        if not self.paused:

            # print('\n\n Tick start:'); t = wall()

            # draw last state
            for i,c in enumerate(self.model.charge):
                self.dbs[i].setCharge(c)

            if self.tip is not None:
                self.updateTip()

            if self.clock is not None:
                self.gradient.updateColors()
                self.scene.setForegroundBrush(self.gradient)

            self.update()
            self.signal_tick.emit()

            # log current state
            if self.logging:
                self.log()

            # print('\tLogging: {0}'.format(wall()-t)); t = wall()

            # update hopper state
            mcount = 30
            milli = mcount
            while milli>0:
                dt = self.model.step()
                millis = dt*1000./self.rate
                milli -= millis

            # print('\tTick time: {0}'.format(wall()-t))

            self.tick_timer.start(min(max(int(millis),mcount), 10000))


    def setCapture(self):
        self.state['bg'] = self.scene.backgroundBrush()
        self.state['fg'] = self.scene.foregroundBrush()
        self.scene.setBackgroundBrush(QBrush(Qt.NoBrush))
        self.scene.setForegroundBrush(QBrush(Qt.NoBrush))

        for item in self.scene.items():
            if hasattr(item, 'setCapture'):
                item.setCapture()


    def unsetCapture(self):
        self.scene.setBackgroundBrush(self.state['bg'])
        self.scene.setForegroundBrush(self.state['fg'])

        for item in self.scene.items():
            if hasattr(item, 'unsetCapture'):
                item.unsetCapture()

    def extents(self):
        '''Get the bounding box for the design ignoring thje H sites'''

        ignore = lambda x: (isinstance(x, DB) and x.bg and not x.charged) or \
                                isinstance(x, SnapTarget)
        items = [x for x in self.scene.items() if not ignore(x)]

        # union of bounding boxes
        rect = items.pop().boundingRect()
        for item in items:
            rect |= item.boundingRect()
        dx, dy = _SF*self.a, _SF*self.c
        rect.adjust(-dx, -dy, dx, dy)

        return rect

    def zoomExtents(self):
        '''Scale view to contain all items in the scene.'''

        rect = self.extents()

        self.fitInView(rect, Qt.KeepAspectRatio)
        self.centerOn(rect.center())

        self.scale_fact = self.transform().m11()

    def screencapture(self, fname, vector=False, rect=None):
        '''Save a screenshot of the QGraphicsScene and save it to the given
        filename. Set vector to True if output format is SVG'''

        source = self.extents() if rect is None else rect
        target = source.translated(-source.topLeft())

        # create directory if non-existant
        direc = os.path.dirname(fname)
        if not os.path.isdir(direc):
            os.makedirs(direc)

        print(source.size())

        if vector:
            # SVG format
            if os.path.splitext(fname)[1].lower() == '.svg':
                canvas = QSvgGenerator()
                canvas.setFileName(fname)
                canvas.setSize(source.size().toSize())
                canvas.setViewBox(target)
            # PDF format
            else:
                canvas = QPrinter()
                canvas.setFullPage(True)
                canvas.setOutputFileName(fname)
                canvas.setPageSize(QPageSize(source.size(), QPageSize.Point))
                # recalibrate, because Qt
                size = QSizeF(source.width()**2/canvas.width(), source.height()**2/canvas.height())
                canvas.setPageSize(QPageSize(size, QPageSize.Point))
            self.setCapture()
        else:
            canvas = QImage(source.size().toSize(), QImage.Format_ARGB32)

        print(canvas.height(), canvas.width())
        painter = QPainter(canvas)
        self.scene.render(painter, target, source)
        painter.end()

        if not vector:
            canvas.save(fname)
        else:
            self.unsetCapture()

        print('Screenshot saved: {0}'.format(fname))

    def updateSnap(self):
        '''Update the snapping target'''

        d = 10*self.snapper.r0
        pos = self.old_pos
        rect = QRectF(pos.x()-.5*d, pos.y()-.5*d, d, d)

        items = self.scene.items(rect)
        if items:
            dist = lambda x: (x.pos()-pos).manhattanLength()
            cands =[x for x in items if not (isinstance(x, SnapTarget) or
                                                isinstance(x, Tracker))]
            target = min(cands, key = dist) if cands else None
        else:
            target = None

        rect = self.snapper.setTarget(target)
        if rect is not None:
            self.scene.update(rect)

    def log(self):
        ''' '''

        out = {'x': (self.a*self.X).tolist(),           # db positions, x
               'y': (self.b*self.Y).tolist(),           # db positions, y
               'charge': self.model.charge.tolist(),    # charge state
               'bias': self.model.beff.tolist(),        # db local biases
               }

        if hasattr(self.model.model, 'lamb'):
            out['lamb'] = self.model.model.lamb

        if self.bulk is not None:
            out['mu'] = self.bulk.mu

        if self.tip is not None:
            tipd = {'x': 10*self.tip.tipX,
                    'y': 10*self.tip.tipY,
                    'R1': 10*self.tip.tipR1,
                    'R2': 10*self.tip.tipR2,
                    'H': 10*self.tip.tipH}

            out['tip'] = tipd

        # tracker information
        out['dbn'] = self.dbn
        if self.dbn != -1 and self.model.charge[self.dbn]:
            tbias = self.model.getLevels(self.dbn)
            out['tbias'] =  {str(k):v for k,v in tbias.items()}

        self.logger.log(out)

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

            item = self.snapper.target
            if isinstance(item, DB) and item.bg:
                item.setCharge(not item.charged)
                self.model.addCharge(item.xx, item.yy, pos=item.charged)

        elif e.button() == Qt.RightButton:
            item = self.snapper.target
            if isinstance(item, DB):
                self.signal_dbtrack.emit(item.n)

    def mouseMoveEvent(self, e):

        scene_pos = self.mapToScene(e.pos())
        dist = (scene_pos-self.old_pos).manhattanLength()

        if self.panning:
            hsb, vsb = self.horizontalScrollBar(), self.verticalScrollBar()
            hsb.setValue(hsb.value()+self.panx-e.x())
            vsb.setValue(vsb.value()+self.pany-e.y())
            self.panx, self.pany = e.x(), e.y()
            e.accept()
        elif dist > self.snapper.r0:
            self.old_pos = scene_pos
            self.updateSnap()
            e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.setCursor(Qt.ArrowCursor)
            self.panning = False
            e.accept()

    def mouseDoubleClickEvent(self, e):
        self.mousePressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Control:
            if self.path:
                self.tip.setScan(self.path, loop=True)
                self.path = []
                self.tick()

    def wheelEvent(self, e):
        '''Scrolling behaviour'''

        zooming = e.modifiers() & Qt.ControlModifier
        if not zooming:
            super(HoppingAnimator, self).wheelEvent(e)
            self.update()
        else:
            self.wheelZoom(e)
        e.accept()

    def wheelZoom(self, e):
        '''Zoom in/out with scroll'''

        pixels = QPointF(e.pixelDelta())
        degrees = QPointF(e.angleDelta())/120
        steps = (pixels if pixels else degrees).y()

        scale = self.scale_fact * (1.+self.zoom_rate)**steps
        lo, hi = self.zoom_bounds
        scale = max(lo, min(hi, scale))

        old_pos = self.mapToScene(e.pos())
        self.scale(scale/self.scale_fact, scale/self.scale_fact)
        delta = self.mapToScene(e.pos())-old_pos
        self.translate(delta.x(), delta.y())
        self.scale_fact = scale

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
        for n, (x,y) in enumerate(zip(self.a*self.X, self.b*self.Y)):
            _x, _y, w = _SF*x, _SF*y, _SF*1
            item = self.scene.items(QRectF(_x, _y, w, w))[0]
            item.hide()
            self.dbs.append(DB(x, y, n=n))
            self.scene.addItem(self.dbs[-1])

        # set view rect
        rect = self.sceneRect()
        dx, dy = .5*rect.width(), .5*rect.height()
        self.setSceneRect(rect.adjusted(-dx, -dy, dx, dy))



class MainWindow(QMainWindow):
    ''' '''

    WINX = 1400     # window width
    WINY = 1000      # window height

    ZOOM = .1

    img_dir = os.path.join('.', 'img')  # directory for image saving

    use_svg = False  # use svg for vector graphics else pdf
    img_ext = { False: '.png',
                True:  '.svg' if use_svg else '.pdf'}

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
        self.animator.dbn = n
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

    def debug(self):

        from PyQt5.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
        import pdb, sys
        pyqtRemoveInputHook()
        try:
            dbg = pdb.Pdb()
            dbg.reset()
            dbg.do_next(None)
            dbg.interaction(sys._getframe().f_back, None)
        finally:
            pyqtRestoreInputHook()


    def keyPressEvent(self, e):

        if e.key() == Qt.Key_Q:
            if self.record:
                self.animator.compile()
            self.animator.cleanup()
            self.close()
        elif e.key() == Qt.Key_O:
            self.dock.setVisible(not self.dock.isVisible())
        elif e.key() == Qt.Key_Space:
            self.animator.tick()
        elif e.key() in [Qt.Key_Plus, Qt.Key_Equal]:
            zfact = 1+self.animator.zoom_rate
            self.animator.scale(zfact, zfact)
        elif e.key() == Qt.Key_Minus:
            zfact = 1-self.animator.zoom_rate
            self.animator.scale(zfact, zfact)
        elif e.key() == Qt.Key_D:
            self.debug()
        elif e.key() == Qt.Key_E:
            self.animator.zoomExtents()
        elif e.key() == Qt.Key_S:
            vector = bool(e.modifiers() & Qt.ShiftModifier)
            fname = QDateTime.currentDateTime().toString('yyyyMMdd-hhmmss')
            fname = os.path.join(self.img_dir, fname+self.img_ext[vector])
            self.animator.screencapture(fname, vector)
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
    # _or.append((4,6,1))
    # _or.append((-2,-1,0))
    # _or.append((10,-1,0))

    def QCA(N):
        qca = []
        for n in range(N):
            x0 = 10*n
            qca += [(x0,0,1), (x0+3,0,1), (x0,2,0), (x0+3,2,0)]
        #qca.append((-4,0,1))
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

    model = HoppingModel(device, model='marcus')
    model.addChannel('bulk')
    model.addChannel('clock')
    #model.addChannel('tip')
    #model.fixElectronCount(3)

    app = QApplication(sys.argv)
    mw = MainWindow(model)

    mw.show()
    sys.exit(app.exec_())
