"""
Contains the OutputFile class for extracting information from outpute files producedby electronic structure theory codes
"""

from . import regex
import re
import cclib.io.ccio as ccio

class OutputFile(object):
    """
    A class for extracting information from output (log) files produced by electronic structure theory codes 
        Parameters
        ----------
        output : str
            A file string of a output (log) file  
    """
    def __init__(self, output):
        self.output = output

    def extract_energy_with_regex(self, energy_regex):
        """
        Finds the energy value (a float) in an output file according to a user supplied
        regular expression identifier.
    
        Example:
        Suppose your output file contains:
        
        FINAL ELECTRONIC ENERGY (Eh):   -2.3564983498

        One can obtain this floating point number with the regex identifier:
        \s*FINAL ELECTRONIC ENERGY \(Eh\):\s+(-\d+\.\d+)

        Checking ones regular expression is easy with online utilities such as pythex

        Parameters
        ---------
        energy_regex : string
            A string 

        Returns
        -------
        last_energy : float
            The last energy value matching the regular expression identifier 
        """
        last_energy = 0.0
        tmp = float(re.findall(energy_regex, output))
        if tmp is not None:
            last_energy = tmp[-1]
        return   last_energy

    def extract_energy_with_cclib(self):
        """
        Attempts to extract energies with cclib 
        """
        pass
