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

# use MLChem 
sys.path.insert(0, "../../../")
import MLChem

input_obj = MLChem.input_processor.InputProcessor("../input.dat")
mol = MLChem.molecule.Molecule(input_obj.zmat_string)
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
    output_obj = MLChem.outputfile.OutputFile(path)
    energy = output_obj.extract_energy_with_regex("Final Energy:\s+(-\d+\.\d+)") 
    data = data.append(df)
    E.append(energy)


os.chdir("../models")
data['Energy'] = E

# we have 100 datapoints of geometry, energy pairs
# take one third for test, validate, train, sample each set evenly along the surface

test_x = data.iloc[1::3, :-1].values
test_y = data.iloc[1::3, -1:].values

valid_x = data.iloc[2::3, :-1].values
valid_y = data.iloc[2::3, -1:].values

train_x = data.iloc[0::3, :-1].values
train_y = data.iloc[0::3, -1:].values

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
percent_error = []
for i in range(50):
    model = Sequential([
    Dense(units=10, input_shape=(1,), activation='softsign'),
    Dense(units=10, activation='softsign'),
    Dense(units=10, activation='softsign'),
    Dense(units=out_dim, activation = 'linear'),
    ])
    
    # fit the model 
    model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    model.fit(x=train_x,y=train_y,epochs=1000,validation_data=valid_set,batch_size=11,verbose=0)
    models.append(model)
    
    # analyze the model performance 
    p = model.predict(np.array(test_x))
    predicted_y = scaler.inverse_transform(p.reshape(-1,1))
    actual_y = scaler.inverse_transform(test_y.reshape(-1,1))
    mae =  np.sum(np.absolute((predicted_y - actual_y))) / len(predicted_y)
    pe = np.mean((predicted_y - actual_y) / actual_y)
    MAE.append(mae)
    percent_error.append(pe)

models = np.asarray(models) 
MAE = np.asarray(MAE) 
percent_error = np.asarray(percent_error) 

models = models[np.argsort(MAE)]
percent_error = percent_error[np.argsort(MAE)]
best_model = models[0]
best_pe = percent_error[0]
best_MAE = MAE.min()
best_model.save("energies_only_1000_epoch.h5")
print(best_MAE)
print(best_pe)

