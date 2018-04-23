from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import pandas as pd
import numpy as np

import sys
import os
import json

# use molssi 
sys.path.insert(0, "../../../")
import molssi

input_obj = molssi.input_processor.InputProcessor("../input.dat")
mol = molssi.molecule.Molecule(input_obj.zmat_string)
geom_labels = mol.geom_parameters
data = pd.DataFrame(columns = geom_labels)

os.chdir("../PES_data")
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


os.chdir("../NNs")
data['Energy'] = E

# we have 100 datapoints of geometry, energy pairs

# validation set, take the 4th, 14th, 24th, etc
valid_x = data.iloc[4::10, :-1].values
valid_y = data.iloc[4::10, -1:].values

# training set, take the 0th, 10th, 20th, etc
train_x = data.iloc[::10, :-1].values
train_y = data.iloc[::10, -1:].values

# test set, take the 1st, 2nd, 3rd, 5th, 6th, 7th, etc
test_x = [i for i in data.iloc[:, :-1].values if (i not in valid_x and i not in train_x) ]
test_y = [i for i in data.iloc[:, -1:].values if (i not in valid_y and i not in train_y) ]
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)


# scale the data
scaler = MinMaxScaler(feature_range=(-1,1))
test_x = scaler.fit_transform((test_x).reshape(-1,1))
test_y = scaler.fit_transform((test_y).reshape(-1,1))

train_x = scaler.fit_transform((train_x).reshape(-1,1))
train_y = scaler.fit_transform((train_y).reshape(-1,1))

valid_x = scaler.fit_transform((valid_x).reshape(-1,1))
valid_y = scaler.fit_transform((valid_y).reshape(-1,1))
valid_set = tuple([valid_x, valid_y])

in_dim = train_x.shape[1]
out_dim = train_y.shape[1]

# train a fresh model 50 times. Save the best one.
models = []
MAE = []
for i in range(1):
    model = Sequential([
    Dense(units=100, input_shape=(1,), activation='softsign'),
    Dense(units=100, activation='softsign'),
    Dense(units=100, activation='softsign'),
    Dense(units=100, activation='softsign'),
    Dense(units=out_dim, activation = 'softsign'),
    ])
    
    model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    model.fit(x=train_x,y=train_y,epochs=1000,validation_data=valid_set,batch_size=25,verbose=1)
    
    performance = model.evaluate(x=test_x,y=test_y)
    models.append(model)
    MAE.append(performance[1])

p = model.predict(np.array(test_x))

predicted_y = scaler.inverse_transform(p.reshape(-1,1))
actual_y = scaler.inverse_transform(test_y.reshape(-1,1))
print(np.hstack((predicted_y,actual_y)))

print("Mean absolute error of the actual energies: ", np.sum(np.absolute((predicted_y - actual_y))) / len(predicted_y))
print("Percent error of actual energies is: ", np.mean((predicted_y - actual_y) / actual_y))

