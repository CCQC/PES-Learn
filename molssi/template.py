"""
Contains the TemplateProcessor class for handling template input files
"""

from . import regex
import re

class TemplateProcessor(object):
    """
    A class for handling template input files for electronic structure theory codes
        Parameters
        ----------
        template : str
            A file string of a template input file  
    """
    def __init__(self, template):
        self.template = template

    def extract_xyz(self):
        """
        Extracts an xyz-style geometry block from an input file 
    
        Parameters
        ----------
        selftemplate : str
            A file string
       
        Returns
        ------- 
        XYZ : str
            An xyz geometry of the form:
            atom_label  x_coord y_coord z_coord 
            atom_label  x_coord y_coord z_coord 
            ...
        """
        
        iter_matches = re.finditer(regex.xyz_block_regex, self.template, re.MULTILINE)
        matches = [match for match in iter_matches]
        if matches is None:
            raise Exception("No XYZ geometry found in template input file")

        # only find last xyz if there are multiple
        # grab string positions of xyz coordinates
        start = matches[-1].start() 
        end   = matches[-1].end() 

        xyz = self.template[start:end]
        return xyz

    def extract_internals():
        pass
