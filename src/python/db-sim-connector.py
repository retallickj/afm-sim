#!/usr/bin/env python

import argparse
from afm import AFMLine
import xml.etree.ElementTree as ET

# parse command line arguments
parser = argparse.ArgumentParser(description='This script takes the minimal problem file produced by the C++ physics engine connector and runs the AFM tip simulation with the AFM path and DB locations given in that file.')
parser.add_argument('in_file', metavar='/path/to/input/file', help='Path to the minimum problem file')
parser.add_argument('out_file', metavar='/path/to/output/file', help='Path to the output file for C++ physics engine to parse and send back to the GUI')

args = parser.parse_args()


