from keras.models import Sequential
from keras.layers.core import Dense

import pandas as pd
import numpy as np

import sys
import os
import json

# use molssi 
sys.path.insert(0, "../../")
import molssi

input_obj = molssi.input_processor.InputProcessor("./input.dat")
input_obj = molssi.input_processor.InputProcessor("./input.dat")
mol = molssi.molecule.Molecule(input_obj.zmat_string)
geom_labels = mol.geom_parameters
DATA = pd.DataFrame(columns = geom_labels)

os.chdir("./PES_data")
ndirs = sum(os.path.isdir(d) for d in os.listdir("."))
for i in range(1, ndirs+1):
    # get geometry data
    with open(str(i) + "/geom") as f:
        tmp = json.load(f) 
    new = []
    for l in geom_labels:
        new.append(tmp[l])
    # create row of dataframe for this geometry, energy
    df = pd.DataFrame([new], columns = geom_labels)
    path = str(i) + "/output.dat"
    # get output data (energies and/or gradients)
    output_obj = molssi.outputfile.OutputFile(path)
    energy = output_obj.extract_energy_with_regex("Final Energy:\s+(-\d+\.\d+)")
    DATA = DATA.append(df)


print(DATA)
