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

        Checking ones regular expression is easy with online utilities such as pythex (see pythex.org)

        Parameters
        ---------
        energy_regex : str
            A string containing the regex code for capturing an energy floating point number.
                e.g. "\s*FINAL ELECTRONIC ENERGY \(Eh\):\s+(-\d+\.\d+)"

        Returns
        -------
        last_energy : float
            The last energy value matching the regular expression identifier 
        """
        last_energy = 0.0
        tmp = float(re.findall(energy_regex, output))
        if tmp is not None:
            last_energy = tmp[-1]
        return last_energy

    def extract_energy_with_cclib(self, cclib_attribute, energy_index=-1):
        """
        Attempts to extract energies with cclib 
        Parameters
        ---------
        cclib_attribute : str
            The cclib target attribute. Valid options can be found in cclib documentation.
            Examples include "ccenergies", "scfenergies", or "mpenergies"
        energy_index : int
            Which energy to grab from the output file which matches the cclib_attribute. 
            Default is -1, the last energy printed in the output file which matches the cclib_attribute. 
        """
        pass
        
    
    def extract_cartesian_gradient_with_regex(self, header, grad_line_regex):
        """
        Extracts cartesian gradients according to a user supplied regular expression.
        A bit more tedious to use than the energy regex extractor as the size of the regular expression string will be quite long.
        Based on the assumption that most electronic structure theory codes print the cartesian gradient in a logical way:
        Example: 
        CARTESIAN GRADIENT:                 (header)
                                            (header)
        Atom 1 O 0.00000 0.23410 0.32398    (grad_line_regex)
        Atom 2 H 0.02101 0.09233 0.01342
        Atom 3 N 0.01531 0.04813 0.06118
    
        Parameters
        ----------
        header : str
            A regex identifier for text that comes immediately before the gradient data
        grad_line_regex : str
            A regex identifier for one line of gradient data. The regex must work for ALL lines of the gradient, so be sure
            to make it general enough. Must use capture groups () for the x, y, and z components
            For example, if the output file line for a gradient is 
            Atom 1 Cl 0.00000 0.23410 0.32398 
            A valid regex would be "\w+\s\d+\s\w+\s(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
            Again, this can easily be tested with online utilities such as pythex (see pythex.org)
        """
        pass
