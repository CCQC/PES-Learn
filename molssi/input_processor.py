"""
A class for extracting information from the main input of the user
"""

from . import regex 
import re
from . import molecule
import ast

class InputProcessor(object):
    """
    """
    def __init__(self, input_path):
        self.input_path = input_path
        self.zmat_string = self.extract_zmat_string()
        self.intcos_ranges = self.extract_intcos_ranges()

    def extract_zmat_string(self):
        with open(self.input_path, 'r') as f:
            self.full_string = f.read() 
        return re.findall(regex.intcoords_regex, self.full_string)[0] 

    def extract_intcos_ranges(self):
        """
        Find within the inputfile path internal coordinate range definitions
        """
        # create molecule object to obtain coordinate labels
        mol = molecule.Molecule(self.zmat_string)
        geomlabels = mol.geom_parameters 
        ranges = {}
        # for every geometry label look for its range identifer, e.g. R1 = [0.5, 1.2]
        for label in geomlabels:
            match = re.search(label+"\s*=\s*(\[.+\])", self.full_string).group(1)
            ranges[label] = ast.literal_eval(match)
         
        

