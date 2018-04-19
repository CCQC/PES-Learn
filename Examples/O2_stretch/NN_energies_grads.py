from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import pandas as pd
import numpy as np
import itertools as it

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
data = pd.DataFrame(columns = geom_labels)

os.chdir("./PES_data")
ndirs = sum(os.path.isdir(d) for d in os.listdir("."))


def cartesian_grad_to_idm(grad):
    """
    Convert cartesian gradient to interatomic distance gradient 
    """
    mat = np.zeros((len(grad),len(grad)))
    for i,j in it.combinations(range(len(grad)),2):
        R = np.linalg.norm(grad[i]-grad[j])
        mat[i,j] = R
        mat[j,i] = mat[i,j]
    return mat

E = []
G = []
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
    gradient = output_obj.extract_cartesian_gradient_with_cclib()
    gradient = cartesian_grad_to_idm(gradient)
    gradient = gradient[0,1]
    data = data.append(df)
    E.append(energy)
    G.append(gradient)


data['Energy'] = E
data['Gradient'] = G

# we have 100 datapoints of geometry, energy, and gradient
# take half of the data set for testing, one quarter for training, one quarter for validating, equally spaced along the surface
test_x = data.iloc[::2, :-2].values
test_y = data.iloc[::2, -2:].values

valid_x = data.iloc[1::4, :-2].values
valid_y = data.iloc[1::4, -2:].values

train_x = data.iloc[0::4, :-2].values
train_y = data.iloc[0::4, -2:].values

# scale the data
scaler = MinMaxScaler(feature_range=(0,1))
test_x = scaler.fit_transform((test_x).reshape(-1,1))
test_y = scaler.fit_transform(test_y)

train_x = scaler.fit_transform((train_x).reshape(-1,1))
train_y = scaler.fit_transform(train_y)

valid_x = scaler.fit_transform((valid_x).reshape(-1,1))
valid_y = scaler.fit_transform(valid_y)
valid_set = tuple([valid_x, valid_y])

in_dim = train_x.shape[1]
out_dim = train_y.shape[1]

model = Sequential([
Dense(units=20, input_shape=(1,), activation='tanh'),
Dense(units=20, activation='tanh'),
Dense(units=20, activation='tanh'),
Dense(units=out_dim, activation = 'tanh'),
])

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
print(model.summary())
model.fit(train_x,train_y,epochs=1000,validation_data=valid_set,verbose=2)

performance = model.evaluate(x=test_x,y=test_y)
print(performance)
