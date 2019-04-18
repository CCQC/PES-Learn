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
        #self.set_default_hyperparameters()

    def set_nas_hyperparameters(self):
        """
        Set default hyperparameter space for neural architecture search.
        Actual hyperparameter optimization will occur later.
        """
        self.hyperparameter_space = {
                                    'scale_X': hp.choice('scale_X', ['mm01']),
                                    'scale_y': hp.choice('scale_y', ['mm01']),
                                    }
        self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': False}]))
        if self.pip:
            val =  hp.choice('pip',[{'pip': True,'degree_reduction': hp.choice('degree_reduction', [False])}])
            self.set_hyperparameter('pip', val)
        else:
            self.set_hyperparameter('pip', hp.choice('pip', [{'pip': False}]))
        
        self.set_hyperparameter('activation', hp.choice('activation', ['tanh']))
        # Later, when NAS is done,
        #self.set_hyperparameter('hidden_layers': hp.choice('hidden_layers', LAYERLIST),

        #if self.input_obj.keywords['pes_format'] == 'interatomics':
        #    self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': True,'morse_alpha': hp.quniform('morse_alpha', 1, 2, 0.1)},{'morse': False}]))
        #else:
        #    self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': False}]))

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

    def build_model(self, params):
        #if self.input_obj.keywords['validation_points']:
        #    nvalid = self.input_obj.keywords['validation_points']
        self.split_train_test(params, validation_size=20)
        inp_dim = self.Xtr.shape[1]
        all_layers = [(20,), (20,20), (20,20,20), (20,20,20,20),
                  (40,), (40,40), (40,40,40), (40,40,40,40),
                  (60,), (60,60), (60,60,60), (60,60,60,60),
                  (80,), (80,80), (80,80,80), (80,80,80,80)]
        layers = sort_architectures(all_layers, inp_dim)

        #factor = self.yscaler.var_[0]

        for layers in all_layers:
            torch.manual_seed(0)
            depth = len(layers)
            structure = OrderedDict([('input', nn.Linear(inp_dim, layers[0])),
                                     ('activ_in' , nn.Tanh())])
            model = nn.Sequential(structure)
            for i in range(depth-1):
                model.add_module('layer' + str(i), nn.Linear(layers[i], layers[i+1]))
                model.add_module('activ' + str(i), nn.Tanh())
            model.add_module('output', nn.Linear(layers[depth-1], 1))

            metric = torch.nn.MSELoss()
            #optimizer = self.get_optimizer('lbfgs', model.parameters(), lr=None)
            optimizer = torch.optim.LBFGS(model.parameters(), tolerance_grad=1e-7, tolerance_change=1e-12, lr=1.0)
            prev_loss = 1.0
            # Find descaling factor to convert loss to original energy units
            scale = params['scale_y']
            if scale == 'std':
                factor = self.yscaler.var_[0]
            if scale.startswith('mm'):
                factor = (1/self.yscaler.scale_[0]**2)
            # Early stopping tracker
            es_tracker = 0
            for epoch in range(1,100):
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
                        #TODO add checks for nan's, very large numbers
                        if epoch > 1:
                            # does validation error not improve by > 1.0% for 2 sets of 10 epochs in a row?
                            if ((prev_loss - val_error_rmse) / prev_loss) < 1e-2:
                                es_tracker += 1
                                if es_tracker > 2:
                                    prev_loss = val_error_rmse * 1.0
                                    break
                            else:
                                es_tracker = 0
                        prev_loss = val_error_rmse * 1.0  # save previous loss to track improvement

            test_pred = model(self.Xtest)
            loss = metric(test_pred, self.ytest)
            test_error_rmse = np.sqrt(loss.item()*factor)* 219474.63
            print(layers, test_error_rmse)
        return loss

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

    def neural_architecture_search(self):
        """
        Trys several models with varying complexity 
        """
        self.set_nas_hyperparameters()
        #layers = [(20,), (20,20), (20,20,20), (20,20,20,20),
        #          (40,), (40,40), (40,40,40), (40,40,40,40),
        #          (60,), (60,60), (60,60,60), (60,60,60,60),
        #          (80,), (80,80), (80,80,80), (80,80,80,80)]
        #          (100,), (200,), (300,), (100,100), 
        layers = [(16,), (16,16), (16,16,16), (16,16,16,16),
                  (32,), (32,32), (32,32,32), (32,32,32,32),
                  (64,), (64,64), (64,64,64), (64,64,64,64),
                  (128,), (128,128), (128,128,128),
                  (256,), (256,256)] 
        inp_dim = self.raw_X.shape[0]
        self.sorted_layers = sort_architectures(layers, inp_dim)
        self.set_hyperparameter('model_complexity', hp.quniform('model_complexity', 0, len(self.sorted_layers), 1))

        self.hyperopt_trials = Trials()
        if self.input_obj.keywords['rseed']:
            rstate = np.random.RandomState(self.input_obj.keywords['rseed'])
        else:
            rstate = None
        best = fmin(self.hyperopt_model,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=10, # TODO add keyword for NAS search iterations
                    rstate=rstate, 
                    show_progressbar=False,
                    trials=self.hyperopt_trials)

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



    #def get_optimizer(self, opt_type, mdata, lr=None): 
    #    if lr:
    #        rate = lr
    #    elif opt_type == 'lbfgs':
    #        rate = 1.0
    #    else: 
    #        rate = 0.001
    #    if opt_type == 'lbfgs':
    #        optimizer = torch.optim.LBFGS(mdata, tolerance_grad=1e-7, tolerance_change=1e-12, lr=rate)
    #    if opt_type == 'adam':
    #        optimizer = torch.optim.Adam(mdata, lr=rate)
    #    return optimizer





            
        




