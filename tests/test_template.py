"""
Test the TemplateProcessor class methods
"""

import molssi
import pytest
import re

with open('tests/xyz_template.dat', 'r') as f:
    testfile = f.read()

template_object = molssi.template.TemplateProcessor(testfile)

def test_extract_xyz():
    x = template_object.extract_xyz()
    assert re.match(molssi.regex.xyz_block_regex, x)
    
    
