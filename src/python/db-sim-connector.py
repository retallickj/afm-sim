#!/usr/bin/env python

from argparse import ArgumentParser
import os.path

from afm import AFMLine
import xml.etree.ElementTree as ET

class DBSimConnector:
    '''This class serves as a connector between the C++ physics engine connector and
    the AFMMarcus Python classes'''

    simparams = {}  # dictionary of simulation parameters
    dbs = []        # list of tuples containing all dbs, (x, y)
    afmnodes = []   # list of tuples containing all afmnodes, (x, y, z)

    def parseArguments(self):
        parser = ArgumentParser(description="This script takes the minimal problem file\
                produced by the C++ physics engine connector and runs the AFM tip\
                simulation with the AFM path and DB locations given in that file.")
        parser.add_argument("-i", "--input", dest="in_file", required=True, 
                type=self.fileMustExist, help="Path to the minimum problem file generated\
                by C++ physics engine connector.", metavar="IN_FILE")
        parser.add_argument("-o", "--output", dest="out_file", required=True,
                help="Path to the output file for C++ physics engine connector to\
                parse.", metavar="OUT_FILE")
        self.args = parser.parse_args()
        
    def fileMustExist(self, fpath):
        '''Check if input file exists for argument parser'''
        if not os.path.exists(fpath):
            raise argparse.ArgumentTypeError("{0} does not exist".format(fpath))
        return fpath

    
    # Deal with input XML
    def parseInputXml(self, fpath):
        '''Parse the input XML file input a form that AFMLine understands'''
        root = ET.parse(fpath).getroot()
        for child in root:
            if child.tag == "params":
                self.grabParams(child)
            elif child.tag == "dbs":
                self.grabDBs(child)
            elif child.tag == "afmnodes":
                self.grabAFMNodes(child)

    def grabParams(self, p_root):
        '''Grab simulation parameters from the params block'''
        for child in p_root:
            self.simparams[child.tag] = child.text
            print("simparams[%s] = %s" % (child.tag, child.text))

    def grabDBs(self, db_root):
        '''Grab DBs from the dbs block'''
        for child in db_root:
            if child.tag == "dbdot":
                self.dbs.append((int(child.attrib['x']), int(child.attrib['y']), int(child.attrib['b'])))
                print("dbdot: %s, %s, %s" % (child.attrib['x'], child.attrib['y'], child.attrib['b']))

    def grabAFMNodes(self, afm_root):
        '''Grab AFMnodes from the afmnodes block'''
        for child in afm_root:
            if child.tag == "afmnode":
                self.afmnodes.append((int(child.attrib['x']), int(child.attrib['y']), 
                        int(child.attrib['b']), float(child.attrib['z'])))
                print("afmnode: %s, %s, %s, %s" % (child.attrib['x'], child.attrib['y'], 
                        child.attrib['b'], child.attrib['z']))


    # Run simulation
    def runSimulation(self):
        '''Run the simulation with the parameters extracted from parseInputXml'''

        # for now, only 1D line scan is supported, all y values will be discarded
        # TODO 2D support
        X = []
        for dbloc in self.dbs:
            X.append(dbloc[0])
        X.sort()
        print(X)

        # call the AFM simulation
        self.afm = AFMLine(X)
        self.afm.setScanType(int(self.simparams['scan_type']), float(self.simparams['write_strength']))
        self.afm.setBias(float(self.simparams['bias']))
        self.afm.run(
                Nel=int(self.simparams['num_electrons']), 
                nscans=int(self.simparams['num_scans']),
                pad=[int(self.simparams['lattice_padding_l']), int(self.simparams['lattice_padding_r'])]
                )


    # Export the results
    def exportResults(self, fpath):
        '''Export the results to XML file for C++ physics engine connector to parse
        and return to GUI'''
        if os.path.exists(fpath):
            raise FileExistsError("There's already a file at the indicated output path, \
                                    aborting")

        node_root = ET.Element("afm_results")
        node_path = ET.SubElement(node_root, "afm_path", path_id="-1")
        # TODO change path_id to generic


        # the DBs encountered by this AFM path
        node_dbs_encountered = ET.SubElement(node_path, "dbs_encountered")
        for db in self.dbs: # TODO this isn't actually the encountered dbs, update
            ET.SubElement(node_dbs_encountered, "db", x=str(db[0]), y=str(db[1])).text = ""

        # line scan results by this AFM path
        node_scan_results = ET.SubElement(node_path, "scan_results")
        for line_scan in self.afm.charges:
            line_charge_str = ""
            for charge in line_scan:
                line_charge_str += str(charge)
            ET.SubElement(node_scan_results, "line_scan").text = line_charge_str

        # write to file
        self.indentXml(node_root)
        tree = ET.ElementTree(node_root)
        tree.write(fpath, encoding="utf-8", xml_declaration=True)


    def indentXml(self, elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indentXml(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

if __name__ == "__main__":
    # for now, only 1D line scan is supported, so all y values will be discarded.

    connector = DBSimConnector()
    connector.parseArguments()
    connector.parseInputXml(connector.args.in_file)
    connector.runSimulation()
    connector.exportResults(connector.args.out_file)
