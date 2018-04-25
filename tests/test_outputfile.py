"""
Test the OutputFile class methods
"""
import numpy as np
import MLChem
import pytest
import re


# open an output file
path = 'tests/datafiles/molpro_water_gradient'
outputfile = MLChem.outputfile.OutputFile(path)

def test_extract_energy_with_regex():
    energy = outputfile.extract_energy_with_regex("!CCSD\(T\) total energy\s+(-?\d+.\d+)")
    assert energy == -76.241305026974

def test_extract_energy_with_cclib():
    energy = outputfile.extract_energy_with_cclib("ccenergies")
    #cclib converts energies to eV, so you lose precision
    assert energy.round(8) == -76.24130503

def test_extract_cartesian_gradient_with_regex():
    gradient = outputfile.extract_cartesian_gradient_with_regex("Atom\s+dE/dx\s+dE/dy.+", "Reading points", "\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
    test = np.array([[ 0.000000000  ,       0.000000025  ,       0.000000007 ], 
                     [ 0.000000000  ,      -0.000000005  ,       0.000000007 ],
                     [-0.000000000  ,      -0.000000019  ,      -0.000000013 ]])
    assert np.allclose(gradient,test)

#cclib currently has poor gradient support, only psi4, qchem, and gaussian well supported. Molpro works in fringe cases 
# have to test with qchem for now
path_2 = 'tests/datafiles/qchem_water_gradient'
outputfile_2 = MLChem.outputfile.OutputFile(path_2)
def test_extract_cartesian_gradient_with_cclib():
    gradient = outputfile_2.extract_cartesian_gradient_with_cclib()
    test = np.array([[ 0.0000000,  0.000000,   0.1273959],
                     [-0.0917225, -0.000000,  -0.063698 ],
                     [ 0.0917225,  0.000000,  -0.063698 ]])
    assert np.allclose(gradient, test)

