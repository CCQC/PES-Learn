"""
Test the TemplateProcessor class methods
"""

import molssi
import pytest

path = 'tests/datafiles/input_zmat_1'
input_object = molssi.input_processor.InputProcessor(path)

mol = molssi.molecule.Molecule(input_object.zmat_string)

def test_extract_intcos_ranges():
    x = input_object.intcos_ranges 
    y = [[0.7, 1.4, 8],[0.5, 1.8, 4.0],[1.0],[1],[-1],[180.0]]
    for key, value in x.items():
        assert value in y
