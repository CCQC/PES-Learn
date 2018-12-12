"""
Data Generation Driver
"""
import timeit
import sys
import os
import json
# python 2 and 3 command line input compatibility
from six.moves import input
from collections import OrderedDict
# find MLChem module
#from .constants import package_directory 
sys.path.insert(0, "../../../")
import MLChem
import numpy as np
import pandas as pd

with open('input.dat', 'r') as f:
    input_string = f.read()

input_obj = MLChem.input_processor.InputProcessor(input_string)
mol = MLChem.molecule.Molecule(input_obj.zmat_string)
text = input("Do you want to 'generate' data, 'parse' data, or 'learn'?")
start = timeit.default_timer()

if text == 'generate':
    config = MLChem.configuration_space.ConfigurationSpace(mol, input_obj)
    template_obj = MLChem.template_processor.TemplateProcessor("./template.dat")
    config.generate_PES(template_obj)
    print("Data generation finished in {} seconds".format(round((timeit.default_timer() - start),2)))

if text == 'parse':
    MLChem.parsing_helper.parse(input_obj, mol)

if text == 'learn':
    if input_obj.keywords["ml_model"] == 'gp':
        gp = MLChem.gaussian_process.GaussianProcess(input_obj.keywords["pes_name"], input_obj, mol)
        gp.optimize_model()
    
stop = timeit.default_timer()
print("Total run time: {} seconds".format(stop - start))
