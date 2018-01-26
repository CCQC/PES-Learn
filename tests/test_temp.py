"""
A test file to see if pytest is working properly
"""

import molssi 
import pytest

def test_hello():
    a = molssi.temp.hello.func()
    assert a == 'Hello, world'
