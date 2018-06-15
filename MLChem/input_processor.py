"""
A class for extracting information from the main input of the user
"""

from . import regex 
import re
from . import molecule
import collections
import numpy as np
import itertools as it
import ast

class InputProcessor(object):
    """
    A Class which handles information contained within an input file
    """
    def __init__(self, input_path):
        self.input_path = input_path
        with open(self.input_path, 'r') as f:
            self.full_string = f.read() 
        self.zmat_string = re.findall(regex.intcoords_regex, self.full_string)[0] 
        self.intcos_ranges = None 
        self.extract_intcos_ranges()
        self.keywords = self.get_keywords()
        
    def get_keywords(self):
        """
        Find keyword definitions within the input file
        """
        # keywords which have values that are strings, not other datatypes
        string_keywords = {'energy': None, 'energy_regex': None, 'energy_cclib': None,
        'gradient': None, 'gradient_header': None, 'gradient_footer': None, 'gradient_line': None,
         'remove_redundant_geometries': 'true'}
        for k in string_keywords:
            match = re.search(k+"\s*=\s*(.+)", self.full_string)
            # if the keyword is mentioned
            if match:
                value = str(match.group(1))
                try:
                    value = ast.literal_eval(value)
                    string_keywords[k] = value
                except:
                    raise Exception("\n'{}' is not a valid option for {}. Entry should be plain text or a string, surrounded by single or double quotes.".format(value,k))
                # make sure it isn't some other datatype like a list  
                if not isinstance(value, str):
                    raise Exception("\n'{}' is not a valid option for {}. Entry should be plain text or a string, surrounded by single or double quotes.".format(value,k))
        return string_keywords
        

    def extract_intcos_ranges(self):
        """
        Find within the inputfile path internal coordinate range definitions
        """
        # create molecule object to obtain coordinate labels
        mol = molecule.Molecule(self.zmat_string)
        geomlabels = mol.geom_parameters 
        ranges = collections.OrderedDict()
        # for every geometry label look for its range identifer, e.g. R1 = [0.5, 1.2, 25]
        for label in geomlabels:
            # check to make sure parameter isn't defined more than once
            if len(re.findall(label+"\s*=\s*", self.full_string)) > 1:
                raise Exception("Parameter {} defined more than once.".format(label))

            # if geom parameter has a geometry range, save it
            match = re.search(label+"\s*=\s*(\[.+\])", self.full_string)
            if match:
                try:
                    ranges[label] = ast.literal_eval(match.group(1))
                except: 
                    raise Exception("Something wrong with definition of parameter {} in input. Should be of the form [start, stop, # of points] or a fixed value".format(label))
            # if it has a fixed value, save it
            else:
                match = re.search(label+"\s*=\s*(-?\d+\.?\d*)", self.full_string)
                if not match:
                    raise Exception("\nDefinition of parameter {} not found in geometry input.      \
                                   \nThe definition is either missing or improperly formatted".format(label))
                ranges[label] = [float(match.group(1))]
        self.intcos_ranges = ranges
    
    def generate_displacements(self):
        d = self.intcos_ranges
        for key, value in d.items():
            if len(value) == 3:
                d[key] = np.linspace(value[0], value[1], value[2])
        #TODO fix cases when user inputs lists of length 2 or >3
        geom_values = list(it.product(*d.values()))

        disps = []
        for geom in geom_values:
            disp = collections.OrderedDict()
            for i, key in enumerate(d):
                disp[key] = geom[i]
            disps.append(disp)
        return disps

        
         
        

