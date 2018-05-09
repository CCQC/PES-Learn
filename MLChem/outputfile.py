"""
Contains the OutputFile class for extracting information from output files produced by electronic structure theory codes
"""

from . import regex
import re
import numpy as np
import cclib.io.ccio as ccio
from . import constants

class OutputFile(object):
    """
    A class for extracting information from output (log) files produced by electronic structure theory codes 
        Parameters
        ----------
        output_path : str
            A string represnting a file path to an output (log) file  
    """
    def __init__(self, output_path):
        self.output_path = output_path
        # save the output as a string 
        with open(output_path, "r") as f:  
            self.output_str = f.read()

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
        tmp = re.findall(energy_regex, self.output_str)
        if tmp:
            last_energy = float(tmp[-1])
            return last_energy
        # how do we handle cases when output files do not produce the energy?
        # we do not want to kill the program, but we also want to communicate that something went wrong during that computation 
        else:
            return None 

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
        cclib_outputobj = ccio.ccread(self.output_path) 
        e = None 
        # also, cclib does not handle things well if something fails to parse, needtry/except pairs 
        if cclib_attribute == "scfenergies":
            try:
                e = cclib_outputobj.scfenergies[-1] 
            except:
                e = None 
        if cclib_attribute == "mpenergies":
            try:
                e = cclib_outputobj.mpenergies[-1] 
            except:
                e = None 
        if cclib_attribute == "ccenergies":
            try:
                e = cclib_outputobj.ccenergies[-1] 
            except:
                e = None 
        # cclib puts energies into eV... ugh 
        if e: 
            e /= constants.hartree2ev
        return e 
    
    def extract_cartesian_gradient_with_regex(self, header, footer, grad_line_regex):
        """
        Extracts cartesian gradients according to user supplied regular expressions.
        A bit more tedious to use than the energy regex extractor as the size of the regular expressions may be quite long.
        Requires that the electronic structure theory code prints the cartesian gradient in a logical way.
        A "header" and "footer" identifier is needed so we don't accidentally parse things that *look like* gradients, like geometries

        Example: 
        CARTESIAN GRADIENT:                 (header)
        (optional extra text
         that does not match grad_line_regex)
        Atom 1 O 0.00000 0.23410 0.32398         (grad_line_regex)
        Atom 2 H 0.02101 0.09233 0.01342     
        Atom 3 N 0.01531 0.04813 0.06118
        (optional extra text that does not match grad_line_regex)
        (footer)
    
        Parameters
        ----------
        header : str
            A string of regular expressions which match unique text that is before and close to the gradient data
        footer : str
            A string of regular expressions for text that comes close to after the gradient data (does not need to be unique)
        grad_line_regex : str
            A regex identifier for one line of gradient data. The regex must work for ALL lines of the gradient, so be sure
            to make it general enough. Must use capture groups () for the x, y, and z components
            For example, if the output file gradient is 
            Atom 1 Cl 0.00000 0.23410 0.32398 
            Atom 2 H  0.02101 0.09233 0.01342     
            Atom 3 N  0.01531 0.04813 0.06118
            A valid argument for grad_line_regex would be "Atom\s+\d+\s+[A-Z,a-z]+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
            This can easily be tested with online utilities such as pythex (see pythex.org)
        Returns
        -------
        gradient : np.array
            A numpy array of floats representing the cartesian gradient
        """
        # warning: cartesian reorientation by quantum chemistry software may mess this up
        # grab all text after the header
        trimmed_str = re.split(header, self.output_str)[-1]
        # isolate gradient data using footer
        trimmed_str = re.split(footer, trimmed_str)[0] 
        # look for gradient line regex 
        gradient = re.findall(grad_line_regex, trimmed_str)
        #TODO add catch for when only some lines of the gradient are parsed but not all, check against number of atoms or something
        if gradient:
            # this gradient is a list of tuples, each tuple is an x, y, z for one atom
            gradient = np.asarray(gradient).astype(np.float)
            return gradient        
        else:
            return None

    def extract_cartesian_gradient_with_cclib(self, grad_index=-1):
        """
        Attempts to extract the cartesian gradient with cclib 
        Parameters
        ---------
        grad_index : int
            Which gradient to grab from the output file. 
            Default is -1, the last gradient printed in the output file which matches the cclib_attribute. 
        """
        # warning: cartesian reorientation by quantum chemistry software may mess this up
        cclib_outputobj = ccio.ccread(self.output_path) 
        if hasattr(cclib_outputobj, 'grads'):
            return cclib_outputobj.grads[-1]
        else:
            return None




