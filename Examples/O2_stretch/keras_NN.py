from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

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
data = pd.DataFrame(columns = geom_labels)

os.chdir("./PES_data")
ndirs = sum(os.path.isdir(d) for d in os.listdir("."))

E = []
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
    data = data.append(df)
    E.append(energy)


data['Energy'] = E
# shuffle the dataframe randomly so that validation set changes (lazy)
data = shuffle(data)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values


# scale the data
scaler = MinMaxScaler(feature_range=(0,1))
x = scaler.fit_transform((x).reshape(-1,1))
y = scaler.fit_transform((y).reshape(-1,1))


in_dim = x.shape[1]
out_dim = y.shape[1]


model = Sequential([
Dense(units=30, input_shape=(1,), activation='sigmoid'),
Dense(units=10, activation='sigmoid'),
Dense(units=out_dim, activation = 'sigmoid'),
])

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', 'mae'])
#print(model.summary())
model.fit(x,y,epochs=200,validation_split=0.3,verbose=2)


#test_x = np.array([1.0, 1.1, 1.2])
#test_y = np.array([-149.3567405256040388 , -149.4430195672399009, -149.4615051852128431])
# scale
#test_x = scaler.fit_transform((test_x).reshape(-1,1))
#test_y = scaler.fit_transform((test_y).reshape(-1,1))
#
#predictions = model.predict(test_x)
#for i,val in enumerate(predictions):
#    print(val)
#    print(test_y[i])
#

#
##loss, accuracy = model.evaluate(x, y, verbose=verbose)
#
