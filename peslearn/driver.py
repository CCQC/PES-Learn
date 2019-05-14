"""
Driver for PES-Learn
"""
import timeit
import sys
import os
import json
from six.moves import input
from collections import OrderedDict
import peslearn
import numpy as np
import pandas as pd

with open('input.dat', 'r') as f:
    input_string = f.read()

input_obj = peslearn.InputProcessor(input_string)

if input_obj.keywords['mode'] == None:
    text = input("Do you want to 'generate' data, 'parse' data, or 'learn'?")
    text = text.strip()

else:
    text = input_obj.keywords['mode']

start = timeit.default_timer()

if text == 'generate' or text == 'g':
    mol = peslearn.datagen.Molecule(input_obj.zmat_string)
    config = peslearn.datagen.ConfigurationSpace(mol, input_obj)
    template_obj = peslearn.datagen.Template("./template.dat")
    config.generate_PES(template_obj)
    print("Data generation finished in {} seconds".format(round((timeit.default_timer() - start),2)))

if text == 'parse' or text == 'p':
    mol = peslearn.datagen.Molecule(input_obj.zmat_string)
    peslearn.utils.parsing_helper.parse(input_obj, mol)

if text == 'learn' or text == 'l':
    if input_obj.keywords['use_pips'] == 'true':
        mol = peslearn.datagen.Molecule(input_obj.zmat_string)
    if input_obj.keywords["ml_model"] == 'gp':
        if input_obj.keywords['use_pips'] == 'true':
            gp = peslearn.ml.GaussianProcess(input_obj.keywords["pes_name"], input_obj, molecule_type=mol.molecule_type)
        else:
            gp = peslearn.ml.GaussianProcess(input_obj.keywords["pes_name"], input_obj)
        gp.optimize_model()
    
stop = timeit.default_timer()
print("Total run time: {} seconds".format(round(stop - start,2)))
