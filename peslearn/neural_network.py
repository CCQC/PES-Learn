import numpy as np
import sklearn.metrics
from .model import Model
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation
from keras.models import load_model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras import backend as K
import time
import re
import os
import sys
import pandas as pd
import numpy as np
from tensorflow import ConfigProto
from tensorflow import Session

from .constants import hartree2cm, package_directory 
from .printing_helper import hyperopt_complete
from .data_sampler import DataSampler 
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler


class NeuralNetwork(Model):
    """
    Constructs a Neural Network Model using Keras with Tensorflow as a backend
    """
    def __init__(self, dataset_path, input_obj, mol_obj=None):
        super().__init__(dataset_path, input_obj, mol_obj)
        self.set_default_hyperparameters()

    def get_hyperparameters(self):
        """
        Returns hyperparameters of this model
        """
        return self.hyperparameter_space

    def set_hyperparameter(self, key, val):
        """
        Set hyperparameter 'key' to value 'val'.
        Parameters
        ---------
        key : str
            A hyperparameter name
        val : obj
            A HyperOpt object such as hp.choice, hp.uniform, etc.
        """
        self.hyperparameter_space[key] = val

    def set_default_hyperparameters(self):
        self.hyperparameter_space = {
                      'scale_X': hp.choice('scale_X',
                               [
                               # Scale input data to domain of chosen activation functions.
                               #{'scale_X': 'std', 
                               #     'activation_0': hp.choice('activation_a0', ['sigmoid', 'tanh', 'linear']),
                               #     'activation_1': hp.choice('activation_a1', ['sigmoid', 'tanh', 'linear']),
                               #     'activation_2': hp.choice('activation_a2', ['sigmoid', 'tanh', 'linear']),
                               #     'activation_3': hp.choice('activation_a3', ['sigmoid', 'tanh', 'linear'])},
                               #{'scale_X': 'mm01', 
                               #     'activation_0': hp.choice('activation_b0', ['sigmoid', 'linear']),
                               #     'activation_1': hp.choice('activation_b1', ['sigmoid', 'linear']),
                               #     'activation_2': hp.choice('activation_b2', ['sigmoid', 'linear']),
                               #     'activation_3': hp.choice('activation_b3', ['sigmoid', 'linear'])},
                               #{'scale_X': 'mm11',
                               #     'activation_0': hp.choice('activation_c0', ['tanh', 'linear']),
                               #     'activation_1': hp.choice('activation_c1', ['tanh', 'linear']),
                               #     'activation_2': hp.choice('activation_c2', ['tanh', 'linear']),
                               #     'activation_3': hp.choice('activation_c3', ['tanh', 'linear'])},


                               {'scale_X': 'std', 
                                    'activation_0': hp.choice('activation_a0', ['sigmoid', 'tanh', 'linear']),
                                    'activation_1': hp.choice('activation_a1', ['sigmoid', 'tanh', 'linear']),
                                    'activation_2': hp.choice('activation_a2', ['sigmoid', 'tanh', 'linear']),
                                    'activation_3': hp.choice('activation_a3', ['sigmoid', 'tanh', 'linear'])},
                               {'scale_X': 'mm01', 
                                    'activation_0': hp.choice('activation_b0', ['sigmoid']),
                                    'activation_1': hp.choice('activation_b1', ['sigmoid']),
                                    'activation_2': hp.choice('activation_b2', ['sigmoid']),
                                    'activation_3': hp.choice('activation_b3', ['sigmoid'])},
                               {'scale_X': 'mm11',
                                    'activation_0': hp.choice('activation_c0', ['tanh']),
                                    'activation_1': hp.choice('activation_c1', ['tanh']),
                                    'activation_2': hp.choice('activation_c2', ['tanh']),
                                    'activation_3': hp.choice('activation_c3', ['tanh'])},
                              ]),
                      'scale_y': hp.choice('scale_y', ['std', 'mm01', 'mm11']), 
                      'dense_0': hp.choice('dense_0', [16,32,64,128,256]),
                      'dense_1': hp.choice('dense_1', [16,32,64,128,256]),
                      'dense_2': hp.choice('dense_2', [16,32,64,128,256]),
                      'dense_3': hp.choice('dense_3', [16,32,64,128,256]),
                      #'hlayers': hp.choice('hlayers', [1,2,3,4]),
                      'hlayers': hp.choice('hlayers', [1,2]),
                      'lr': hp.choice('lr',[0.001,0.005,0.01,0.05,0.1]),
                      'decay': hp.choice('decay',[1e-5,1e-6,0.0]),
                      'batch': hp.choice('batch', [16,32,64,128,256]),
                      #'optimizer': hp.choice('optimizer',['SGD','RMSprop', 'Adagrad', 'Adadelta','Adam','Adamax']),
                      'optimizer': hp.choice('optimizer',['Adam']),
                     }

        if self.input_obj.keywords['pes_format'] == 'interatomics':
            self.hyperparameter_space['morse_transform'] = hp.choice('morse_transform',[{'morse': True,'morse_alpha': hp.uniform('morse_alpha', 1.0, 2.0)},{'morse': False}])
        else:
            self.hyperparameter_space['morse_transform'] = hp.choice('morse_transform',[{'morse': False}])
        if self.pip:
            val =  hp.choice('pip',[{'pip': True,'degree_reduction': hp.choice('degree_reduction', [True,False])}])
            self.set_hyperparameter('pip', val)
        else:
            self.set_hyperparameter('pip', hp.choice('pip', [False]))

    def preprocess(self, params, raw_X, raw_y):
        """
        Preprocess raw data according to hyperparameters
        """
        # TODO make more flexible. If keys don't exist, ignore them. smth like "if key: if param['key']: do transform"
        # Transform to FIs, degree reduce if called
        if params['pip']:
            # find path to fundamental invariants for an N atom system with molecule type AxByCz...
            path = os.path.join(package_directory, "lib", str(sum(self.mol.atom_count_vector))+"_atom_system", self.mol.molecule_type, "output")
            raw_X, degrees = interatomics_to_fundinvar(raw_X,path)
            if params['pip']['degree_reduction']:
                raw_X = degree_reduce(raw_X, degrees)

        # Transform to morse variables (exp(-r/alpha))
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])


        if params['scale_X']['scale_X']:
            X, Xscaler = general_scaler(params['scale_X']['scale_X'], raw_X)
        else:
            X = raw_X
            Xscaler = None
        if params['scale_y']:
            y, yscaler = general_scaler(params['scale_y'], raw_y)
        else:
            y = raw_y
            yscaler = None
        return X, y, Xscaler, yscaler


    def split_train_test(self, params):
        """
        Take raw dataset and apply hyperparameters/input keywords/preprocessing
        and train/test/validation set splitting.
        Assigns:
        self.X : complete input data, transformed
        self.y : complete output data, transformed
        self.Xscaler : scaling transformer for inputs 
        self.yscaler : scaling transformer for outputs 
        self.Xtr : training input data, transformed
        self.ytr : training output data, transformed
        self.Xtest : test input data, transformed
        self.ytest : test output data, transformed
        self.Xvalid : validation input data, transformed
        self.yvalid : validation output data, transformed
        """
        self.X, self.y, self.Xscaler, self.yscaler = self.preprocess(params, self.raw_X, self.raw_y)
        self.Xtr = self.X[self.train_indices]
        self.ytr = self.y[self.train_indices]
        Xtmp = self.X[self.test_indices]
        ytmp = self.y[self.test_indices]
        self.Xvalid, self.Xtest, self.yvalid, self.ytest = train_test_split(Xtmp, ytmp, train_size = self.input_obj.keywords['validation_points'], random_state=42)

    def build_model(self, params):
        self.split_train_test(params)
        #print("Hyperparameters: ", params)
        config = ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
        session = Session(config=config)
        K.set_session(session)
        in_dim = tuple([self.Xtr.shape[1]])
        out_dim = self.ytr.shape[1]
        valid_set = tuple([self.Xvalid, self.yvalid])

        activ0 = params['scale_X']['activation_0']
        activ1 = params['scale_X']['activation_1']
        activ2 = params['scale_X']['activation_2']
        activ3 = params['scale_X']['activation_3']

        self.model = Sequential()
        self.model.add(Dense(params['dense_0']))
        self.model.add(Activation(activ0))
        if params['hlayers'] > 1:  
            self.model.add(Dense(params['dense_1']))
            self.model.add(Activation(activ1))
            if params['hlayers'] > 2:  
                self.model.add(Dense(params['dense_2']))
                self.model.add(Activation(activ2))
                if params['hlayers'] > 3:  
                    self.model.add(Dense(params['dense_3']))
                    self.model.add(Activation(activ3))
        self.model.add(Dense(out_dim))
        self.model.add(Activation('linear'))
    
        if params['optimizer'] == 'SGD':
            opt = optimizers.SGD(lr=params['lr'],
                                  decay=params['decay'])
        if params['optimizer'] == 'RMSprop':
            opt = optimizers.RMSprop(lr=params['lr'],
                                  decay=params['decay'])
        if params['optimizer'] == 'Adagrad':
            opt = optimizers.Adagrad(lr=params['lr'],
                                  decay=params['decay'])
        if params['optimizer'] == 'Adadelta':
            opt = optimizers.Adadelta(lr=params['lr'],
                                  decay=params['decay'])
        if params['optimizer'] == 'Adam':
            opt = optimizers.Adam(lr=params['lr'],
                                  decay=params['decay'],
                                  amsgrad=True)
        if params['optimizer'] == 'Adamax':
            opt = optimizers.Adamax(lr=params['lr'],
                                  decay=params['decay'])
    
        # if y is scaled, our model performance metric is not in true units of energy
        # we compensate for this here so that we can use early stopping effectively
        # assuming error is the MAE, one can derive the factors which transform the MAE on scaled energies into the MAE on raw energies
        if ((params['scale_y'] == 'mm01') or (params['scale_y'] == 'mm11')):
            descaling_factor = (self.yscaler.data_max_[0] - self.yscaler.data_min_[0]) / (self.yscaler.feature_range[1] - self.yscaler.feature_range[0])
        if params['scale_y'] == 'std':
            descaling_factor = self.yscaler.var_[0]**0.5 
        # 11 cm-1
        # if error does not improve by 1 mH in 500 epochs, kill
        delta = self.early_stopping_error / descaling_factor
        callback = [EarlyStopping(monitor='val_loss', min_delta=delta, patience=self.patience)]
        self.model.compile(loss='mae', optimizer=opt, metrics=['mse'])
        self.model.fit(x=self.Xtr,y=self.ytr,epochs=self.epochs,validation_data=valid_set,batch_size=params['batch'],verbose=0,callbacks=callback)

    def hyperopt_model(self, params):
        # skip building this model if hyperparameter combination already attempted
        is_repeat = None
        for i in self.hyperopt_trials.results:
            if 'memo' in i:
                if params == i['memo']:
                    is_repeat = True
        if is_repeat:
            return {'loss': 0.0, 'status': STATUS_FAIL, 'memo': 'repeat'}
        else:
            self.build_model(params)
            error_test = self.vet_model(self.model)
            return {'loss': error_test, 'status': STATUS_OK, 'memo': params}

    def predict(self, model, data_in):
        prediction = model.predict(data_in)
        return prediction 

    def vet_model(self, model):
        """Convenience method for getting model errors of test and full datasets"""
        pred_train = self.predict(model, self.Xtr)
        pred_test = self.predict(model, self.Xtest)
        pred_valid = self.predict(model, self.Xvalid)
        pred_full = self.predict(model, self.X)
        error_train = self.compute_error(self.Xtr, self.ytr, pred_train, self.yscaler)
        error_test = self.compute_error(self.Xtest, self.ytest, pred_test, self.yscaler)
        error_valid = self.compute_error(self.Xvalid, self.yvalid, pred_valid, self.yscaler)
        error_full, max_errors = self.compute_error(self.X, self.y, pred_full, self.yscaler, 10)
        print("Train {}".format(round(hartree2cm * error_train,2)), end='    ')
        print("Test {}".format(round(hartree2cm * error_test,2)), end='    ')
        print("Validation {}".format(round(hartree2cm * error_valid,2)), end='    ')
        print("Full {}".format(round(hartree2cm * error_full,2)), end='    ')
        print("Max 10 errors: {}".format(np.sort(np.round(max_errors.flatten(),1))))
        return error_test

    def optimize_model(self):
        print("Training with {} points (Full dataset contains {} points).".format(self.ntrain, self.n_datapoints))
        print("Validating with {} points.".format(self.input_obj.keywords['validation_points']))
        print("Using {} training set point sampling.".format(self.sampler))
        print("\nPerforming neural architecture search with early stopping...")
        print("Trying {} combinations of hyperparameters".format(self.hp_max_evals))
        self.hyperopt_trials = Trials()
        if self.input_obj.keywords['rseed']:
            rstate = np.random.RandomState(self.input_obj.keywords['rseed'])
        else:
            rstate = None
        self.epochs = 2000 
        self.early_stopping_error = 5.0e-4
        self.patience = 200
        best = fmin(self.hyperopt_model,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.hp_max_evals,
                    rstate=rstate, 
                    trials=self.hyperopt_trials)
        print("Best performing hyperparameters are:")
        final = space_eval(self.hyperparameter_space, best)
        print(str(sorted(final.items())))
        self.optimal_hyperparameters  = dict(final)
        print("Fine-tuning final model architecture... optimizing learning rate and decay at high epoch limit")
        self.optimal_hyperparameters['decay'] = hp.choice('decay',[1e-5,1e-6,1e-7,0.0]) 
        self.optimal_hyperparameters['lr'] = hp.choice('lr',[0.005,0.01,0.05,0.1]) 

        self.epochs = 30000
        self.early_stopping_error = 5.0e-5
        self.patience = 1000
        best = fmin(self.hyperopt_model,
                    space=self.optimal_hyperparameters,
                    algo=tpe.suggest,
                    max_evals=10,
                    rstate=rstate, 
                    trials=self.hyperopt_trials)
        hyperopt_complete()
        final = space_eval(self.optimal_hyperparameters, best)
        self.optimal_hyperparameters = dict(final)
        self.build_model(self.optimal_hyperparameters)
        print("Final model performance (cm-1):")
        self.vet_model(self.model)
        self.save_model(self.optimal_hyperparameters)

    def save_model(self, params):
        print("Saving ML model data...") 
        model_path = "model1_data"
        while os.path.isdir(model_path):
            new = int(re.findall("\d+", model_path)[0]) + 1
            model_path = re.sub("\d+",str(new), model_path)
        os.mkdir(model_path)
        os.chdir(model_path)
        self.model.save("model.h5")
        with open('hyperparameters', 'w') as f:
            print(params, file=f)
        self.dataset.iloc[self.train_indices].to_csv('train_set',sep=',',index=False,float_format='%12.12f')
        self.dataset.iloc[self.test_indices].to_csv('test_set', sep=',', index=False, float_format='%12.12f')
        # print model performance
        sys.stdout = open('performance', 'w')  
        self.vet_model(self.model)
        sys.stdout = sys.__stdout__
        os.chdir("../")


