"""
Test the TemplateProcessor class methods
"""

import peslearn
import pytest
import re

path = 'tests/datafiles/xyz_template'
template_object = peslearn.datagen.Template(path)

def test_extract_xyz():
    x = template_object.extract_xyz()
    assert re.match(peslearn.utils.regex.xyz_block_regex, x)
   
def test_header_xyz():
    x = template_object.header_xyz()
    assert x == '# a psi4 input file to text xyz extraction\n\nmolecule test {\n0 1\n'    

def test_footer_xyz():
    x = template_object.footer_xyz()
    assert x == "}\n\nset basis cc-pvdz\nset reference rhf\n\nenergy('hf')\n"

def test_parse_xyz():
    x, y  = template_object.parse_xyz()
    assert ((x == 64) and (y == 115))

