import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict

from .model import Model
from ..constants import hartree2cm, package_directory
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler
from sklearn.model_selection import train_test_split   
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from .preprocessing_helper import sort_architectures

class NeuralNetwork(Model):
    """
    Constructs a Neural Network Model using PyTorch
    """
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)
        
        if self.input_obj.keywords['validation_points']:
            self.nvalid = self.input_obj.keywords['validation_points']
        
        if self.pip:
            if molecule_type:
                path = os.path.join(package_directory, "lib", molecule_type, "output")
                self.inp_dim = len(open(path).readlines())
            if molecule:
                path = os.path.join(package_directory, "lib", molecule.molecule_type, "output")
                self.inp_dim = len(open(path).readlines())
        else:
            self.inp_dim = self.raw_X.shape[1]
        
    def neural_architecture_search(self):
        tmp_layers = [(16,), (16,16), (16,16,16), (16,16,16,16),
                      (32,), (32,32), (32,32,32), (32,32,32,32),
                      (64,), (64,64), (64,64,64), (64,64,64,64),
                      (128,), (128,128), (128,128,128),
                      (256,), (256,256)] 
        self.nas_layers = sort_architectures(tmp_layers, self.inp_dim)
        self.nas_size = len(self.nas_layers)
        params = {'morse_transform': {'morse':False},'scale_X':'std', 'scale_y':'std','activation': 'tanh'} 
        if self.pip:
            params['pip'] = {'degree_reduction': False, 'pip': True} 
        else:
            params['pip'] = {'degree_reduction': False, 'pip': False} 
        test = []
        validation = []
        for i in self.nas_layers:
            params['layers'] = i
            testerror, valid = self.build_model(params)
            test.append(testerror)
            validation.append(valid)
        print("Best performing Neural Network: {} (test) {} (validation)".format(min(test), min(validation)))

    def split_train_test(self, params, validation_size=None):
        """
        Take raw dataset and apply hyperparameters/input keywords/preprocessing
        and train/test (tr,test) splitting.
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
        if self.sampler == 'user_supplied':
            self.Xtr = self.transform_new_X(self.raw_Xtr, params, self.Xscaler)
            self.ytr = self.transform_new_y(self.raw_ytr, self.yscaler)
            self.Xtest = self.transform_new_X(self.raw_Xtest, params, self.Xscaler)
            self.ytest = self.transform_new_y(self.raw_ytest, self.yscaler)
            if self.valid_path:
                self.Xvalid = self.transform_new_X(self.raw_Xvalid, params, self.Xscaler)
                self.yvalid = self.transform_new_y(self.raw_yvalid, self.yscaler)
            else:
                raise Exception("Please provide a validation set for Neural Network training.")
        else:
            self.Xtr = self.X[self.train_indices]
            self.ytr = self.y[self.train_indices]
            self.Xtmp = self.X[self.test_indices]
            self.ytmp = self.y[self.test_indices]
            if validation_size:
                self.Xvalid, self.Xtest, self.yvalid, self.ytest =  train_test_split(self.Xtmp,
                                                                                     self.ytmp, 
                                                                   train_size = validation_size, 
                                                                                random_state=42)
        # convert to Torch Tensors
        self.Xtr    = torch.Tensor(data=self.Xtr) 
        self.ytr    = torch.Tensor(data=self.ytr)
        self.Xtest  = torch.Tensor(data=self.Xtest)
        self.ytest  = torch.Tensor(data=self.ytest)
        self.Xvalid = torch.Tensor(data=self.Xvalid)
        self.yvalid = torch.Tensor(data=self.yvalid)        

    def get_optimizer(self, opt_type, mdata, lr=None): 
        if lr:
            rate = lr
        elif opt_type == 'lbfgs':
            rate = 0.5
        else: 
            rate = 0.01
        if opt_type == 'lbfgs':
            optimizer = torch.optim.LBFGS(mdata, tolerance_grad=1e-7, tolerance_change=1e-12, lr=rate)
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(mdata, lr=rate)
        return optimizer

    def build_model(self, params):
        self.split_train_test(params, validation_size=self.nvalid)  # split data, according to scaling hp's
        scale = params['scale_y']                                   # Find descaling factor to convert loss to original energy units
        if scale == 'std':
            factor = self.yscaler.var_[0]
        if scale.startswith('mm'):
            factor = (1/self.yscaler.scale_[0]**2)

        inp_dim = self.inp_dim
        l = params['layers']
        torch.manual_seed(0)
        depth = len(l)
        structure = OrderedDict([('input', nn.Linear(inp_dim, l[0])),
                                 ('activ_in' , nn.Tanh())])
        model = nn.Sequential(structure)
        for i in range(depth-1):
            model.add_module('layer' + str(i), nn.Linear(l[i], l[i+1]))
            model.add_module('activ' + str(i), nn.Tanh())
        model.add_module('output', nn.Linear(l[depth-1], 1))

        metric = torch.nn.MSELoss()
        optimizer = self.get_optimizer('lbfgs', model.parameters(), lr=None)
        prev_loss = 1.0
        # Early stopping tracker
        es_tracker = 0
        for epoch in range(1,1000):
            def closure():
                optimizer.zero_grad()
                y_pred = model(self.Xtr)
                loss = metric(y_pred, self.ytr)
                loss.backward()
                return loss
            optimizer.step(closure)
            # validate
            if epoch % 10 == 0:
                with torch.no_grad():
                    tmp_pred = model(self.Xvalid) 
                    loss = metric(tmp_pred, self.yvalid)
                    val_error_rmse = np.sqrt(loss.item() * factor) * 219474.63
                    print('epoch: ', epoch,'Validation set RMSE (cm-1): ', val_error_rmse)
                    # very simple early stopping implementation
                    if epoch > 1:
                        # does validation error not improve by > 1.0% for 2 sets of 10 epochs in a row?
                        if ((prev_loss - val_error_rmse) / prev_loss) < 1e-2:
                            es_tracker += 1
                            if es_tracker > 2:
                                prev_loss = val_error_rmse * 1.0
                                break
                        else:
                            es_tracker = 0

                    # exploding gradients 
                    if epoch > 10:
                        if (val_error_rmse > prev_loss*10): # detect large increases in loss
                            break
                        if val_error_rmse != val_error_rmse: # detect NaN 
                            break
                        
                    prev_loss = val_error_rmse * 1.0  # save previous loss to track improvement

        test_pred = model(self.Xtest)
        loss = metric(test_pred, self.ytest)
        test_error_rmse = np.sqrt(loss.item()*factor)* 219474.63
        print(l, test_error_rmse)
        return test_error_rmse, val_error_rmse

    def hyperopt_model(self, params):
        # skip building this model if hyperparameter combination already attempted
        for i in self.hyperopt_trials.results:
            if 'memo' in i:
                if params == i['memo']:
                    return {'loss': i['loss'], 'status': STATUS_OK, 'memo': 'repeat'}
        #if self.itercount > self.hp_maxit:
        #    return {'loss': 0.0, 'status': STATUS_FAIL, 'memo': 'max iters reached'}
        self.build_model(params)
        #error_test = self.vet_model(self.model)
        return {'loss': error_test, 'status': STATUS_OK, 'memo': params}

    def preprocess(self, params, raw_X, raw_y):
        """
        Preprocess raw data according to hyperparameters
        """
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])
        if params['pip']['pip']:
            # find path to fundamental invariants form molecule type AxByCz...
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            raw_X, degrees = interatomics_to_fundinvar(raw_X,path)
            if params['pip']['degree_reduction']:
                raw_X = degree_reduce(raw_X, degrees)
        if params['scale_X']:
            X, Xscaler = general_scaler(params['scale_X'], raw_X)
        else:
            X = raw_X
            Xscaler = None
        if params['scale_y']:
            y, yscaler = general_scaler(params['scale_y'], raw_y)
        else:
            y = raw_y
            yscaler = None
        return X, y, Xscaler, yscaler



    def save_model(self):
        pass








            
        




