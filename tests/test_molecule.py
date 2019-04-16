"""
Test the Atom/Molecule classes 
"""

import peslearn
import pytest

path = 'tests/datafiles/input_zmat_1'
with open(path, 'r') as f:
    input_string = f.read()

input_obj = peslearn.InputProcessor(input_string)
mol = peslearn.datagen.Molecule(input_obj.zmat_string)

def test_extract_zmat():
    assert mol.n_atoms == 4
    assert mol.atom_labels == ['C','H','H','H']
    assert mol.geom_parameters == ['RCH1', 'r2', 'a1', 'r3', 'a2', 'D1']

def test_molecule_update_intcoords():
    newmol = peslearn.datagen.Molecule(input_obj.zmat_string)
    disp = {'RCH1': 2.0, 'r2': 1.0}
    newmol.update_intcoords(disp)
    assert newmol.atoms[1].intcoords['RCH1'] == 2.0
    assert newmol.atoms[2].intcoords['r2'] == 1.0

