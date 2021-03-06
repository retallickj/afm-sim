#!/usr/bin/env python
# encoding: utf-8

'''
Python connector that takes the output from C++ physics engine connector and
runs the simulation with the desired parameters
'''

__author__      = 'Samuel Ng'
__copyright__   = 'Apache License 2.0'
__version__     = '0.2'
__date__        = '2018-06-02'  # last update

from argparse import ArgumentParser
import os.path
import xml.etree.ElementTree as ET

from siqadtools import siqadconn

#from afm import AFMLine
from animator import (HoppingAnimator, MainWindow)
from hopper import HoppingModel

from qt_import import qt, import_qt_mod, import_qt_attr
import_qt_mod('QtCore', 'QtGui', 'QtWidgets', wc=globals())

class DBSimConnector:
    '''This class serves as a connector between the C++ physics engine connector and
    the AFMMarcus Python classes'''

    dbs = []        # list of tuples containing all dbs, (x, y)
    afmnodes = []   # list of tuples containing all afmnodes, (x, y, z)

    def parseCmlArguments(self):
        '''Parse command line arguments.'''

        def fileMustExist(fpath):
            '''Check if input file exists for argument parser'''
            if not os.path.exists(fpath):
                raise argparse.ArgumentTypeError("{0} does not exist".format(fpath))
            return fpath

        parser = ArgumentParser(description="This script takes the problem file "
                "and runs the AFM tip simulation with the AFM path and DB locations "
                "given in that file.")
        parser.add_argument(dest="in_file", type=fileMustExist,
                help="Path to the problem file.",
                metavar="IN_FILE")
        parser.add_argument(dest="out_file", help="Path to the output file.",
                metavar="OUT_FILE")
        parser.add_argument("--pot-json-import-path", dest="json_import_path",
                help="Path to the JSON DB potentials file.", 
                metavar="JSON_IMPORT_PATH")
        self.args = parser.parse_args()


    # Import problem parameters and design from SiQAD Connector
    def initProblem(self):
        self.sqconn = siqadconn.SiQADConnector("AFMMarcus", self.args.in_file,
            self.args.out_file)

        # retrieve DBs and convert to a format that hopping model takes
        for db in self.sqconn.dbCollection():
            self.dbs.append((db.n, db.m, db.l))

    # Run simulation
    def runSimulation(self):
        '''Run the simulation'''

        # check simulation type ('animation' or 'line_scan')
        if (self.sqconn.getParameter('simulation_type') == 'line_scan'):
            self.runLineScan()
        else:
            self.runAnimation()

    def runLineScan(self):
        # for now, only 1D line scan is supported, all y values will be discarded
        # TODO 2D support
        X = []
        for dbloc in self.dbs:
            X.append(dbloc[0])
        X.sort()
        print(X)

        # call the AFM simulation
        #self.afm = AFMLine(X)
        #self.afm.setScanType(int(self.sqconn.getParameter('scan_type')),
        #        float(self.sqconn.getParameter('write_strength')))
        #self.afm.setBias(float(self.sqconn.getParameter('bias')))
        #self.afm.run(Nel=int(self.sqconn.getParameter('num_electrons')),
        #        nscans=int(self.sqconn.getParameter('num_scans')),
        #        pad=[int(self.sqconn.getParameter('lattice_padding_l')),
        #            int(self.sqconn.getParameter('lattice_padding_r'))]
        #        )

    def runAnimation(self):
        import sys

        model = HoppingModel(self.dbs, self.sqconn.getParameter('hopping_model'))
        model.fixElectronCount(int(self.sqconn.getParameter('num_electrons')))

        model.addChannel('bulk')
        model.addChannel('clock', enable=False, fname=self.args.json_import_path)
        #model.addChannel('tip', enable=False)

        app = QApplication(sys.argv)
        mw = MainWindow(model)

        mw.show()
        mw.animator.start()
        sys.exit(app.exec_())

if __name__ == "__main__":
    # TODO maybe move this to animator.py
    connector = DBSimConnector()
    connector.parseCmlArguments()
    connector.initProblem()
    connector.runSimulation()
