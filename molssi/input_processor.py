"""
A class for extracting information from the main input of the user
"""

from regex import intcoords_regex
import re
import molecule 
import ast

class InputProcessor(object):
    """
    """
    def __init__(self, input_path):
        self.input_path = input_path
        self.zmat_string = extract_zmat_string(self.input_path)

    def extract_zmat_string(self):
        with open(self.input_path, 'r') as f:
            self.full_string = f.read() 
        return re.findall(intcoords_regex, full_string)[0] 

    def extract_intcos_ranges(self):
        # create molecule object to obtain coordinate labels
        mol = Molecule(self.zmat_string)
        geomlabels = mol.geom_parameters 
        ranges = {}
        # for every geometry label look for its range identifer, e.g. R1 = [0.5, 1.2]
        for label in geomlabels:
            match = re.search(label+"\s*=\s*(\[.+\])", self.full_string).group()
            ranges[label] = ast.literal_eval(match)
         
        

# method: extract zmat string with regex
# method: extract internal coordinate scan ranges r1 = [0.7, 3.0], etc
# method(s): keep track of keyword options used  
