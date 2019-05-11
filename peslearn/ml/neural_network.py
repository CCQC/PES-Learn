import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict

from .model import Model
from ..constants import hartree2cm, package_directory
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler
from ..utils.printing_helper import hyperopt_complete
from sklearn.model_selection import train_test_split   
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from .preprocessing_helper import sort_architectures

class NeuralNetwork(Model):
    """
    Constructs a Neural Network Model using PyTorch
    """
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)
        self.set_default_hyperparameters()
        
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

    def set_default_hyperparameters(self):
        """
        Set default hyperparameter space. If none is provided, default is used.
        """
        self.hyperparameter_space = {
                      'scale_X': hp.choice('scale_X',
                               [
                               {'scale_X': 'mm01', 
                                    'activation': hp.choice('activ1', ['sigmoid'])},
                               {'scale_X': 'mm11',
                                    'activation': hp.choice('activ2', ['tanh'])},
                               {'scale_X': 'std',
                                    'activation': hp.choice('activ3', ['tanh'])},
                               ]),
                      'scale_y': hp.choice('scale_y', ['std', 'mm01', 'mm11']),
                                    }

        if self.input_obj.keywords['pes_format'] == 'interatomics':
            self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': True,'morse_alpha': hp.quniform('morse_alpha', 1, 2, 0.1)},{'morse': False}]))
        else:
            self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': False}]))
        if self.pip:
            val =  hp.choice('pip',[{'pip': True,'degree_reduction': hp.choice('degree_reduction', [True,False])}])
            self.set_hyperparameter('pip', val)
        else:
            self.set_hyperparameter('pip', hp.choice('pip', [{'pip': False}]))

    def optimize_model(self):
        best_hlayers = self.neural_architecture_search()
        self.set_hyperparameter('layers', hp.choice('layers', best_hlayers))
        
        self.hyperopt_trials = Trials()
        self.itercount = 1
        if self.input_obj.keywords['rseed']:
            rstate = np.random.RandomState(self.input_obj.keywords['rseed'])
        else:
            rstate = None
        best = fmin(self.hyperopt_model,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.hp_maxit*2,
                    rstate=rstate, 
                    show_progressbar=False,
                    trials=self.hyperopt_trials)
        hyperopt_complete()
        print("Best performing hyperparameters are:")
        final = space_eval(self.hyperparameter_space, best)
        print(str(sorted(final.items())))
        self.optimal_hyperparameters  = dict(final)
        print("Fine-tuning final model architecture...")
        self.build_model(self.optimal_hyperparameters, es_patience=3, decay=True, verbose=True)

        
    def neural_architecture_search(self):
        """
        Finds optimal hidden layer structure
        """
        tmp_layers = [(16,), (16,16), (16,16,16), (16,16,16,16),
                      (32,), (32,32), (32,32,32), (32,32,32,32),
                      (64,), (64,64), (64,64,64), (64,64,64,64),
                      (128,), (128,128), (128,128,128),
                      (256,), (256,256)] 
        #tmp_layers = [(16,), (16,16),
        #              (32,), (32,32)]
        self.nas_layers = sort_architectures(tmp_layers, self.inp_dim)
        self.nas_size = len(self.nas_layers)
        params = {'morse_transform': {'morse':False},'scale_X':{'scale_X':'std', 'activation':'tanh'}, 'scale_y':'std'}#,'activation': 'tanh'} 
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
        #print("Best performing Neural Network: {} (test) {} (validation)".format(min(test), min(validation)))
        # save best architectures
        indices = np.argsort(test)
        best_hlayers = [tmp_layers[i] for i in indices[:3]]
        return best_hlayers

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

    def build_model(self, params, es_patience=2, decay=False, verbose=False):
        print("Hyperparameters: ", params)
        self.split_train_test(params, validation_size=self.nvalid)  # split data, according to scaling hp's
        scale = params['scale_y']                                   # Find descaling factor to convert loss to original energy units
        if scale == 'std':
            factor = self.yscaler.var_[0]
        if scale.startswith('mm'):
            factor = (1/self.yscaler.scale_[0]**2)

        activation = params['scale_X']['activation']
        if activation == 'tanh':
            activ = nn.Tanh()
        if activation == 'sigmoid':
            activ = nn.Sigmoid()
        
        inp_dim = self.inp_dim
        l = params['layers']
        torch.manual_seed(0)
        depth = len(l)
        structure = OrderedDict([('input', nn.Linear(inp_dim, l[0])),
                                 ('activ_in' , activ)])
        model = nn.Sequential(structure)
        for i in range(depth-1):
            model.add_module('layer' + str(i), nn.Linear(l[i], l[i+1]))
            model.add_module('activ' + str(i), activ)
        model.add_module('output', nn.Linear(l[depth-1], 1))

        metric = torch.nn.MSELoss()
        optimizer = self.get_optimizer('lbfgs', model.parameters(), lr=None)
        prev_loss = 1.0
        # Early stopping tracker
        es_tracker = 0
        if decay:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
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
                    val_error_rmse = np.sqrt(loss.item() * factor) * hartree2cm
                    if verbose:
                        print("Epoch {} Validation RMSE (cm-1): {5.2f}".format(val_error_rmse))
                    if decay:
                        scheduler.step(val_error_rmse)
    
                    # very simple early stopping implementation
                    if epoch > 1:
                        # does validation error not improve by > 1.0% for 2 sets of 10 epochs in a row?
                        if ((prev_loss - val_error_rmse) / prev_loss) < 1e-2:
                            es_tracker += 1
                            if es_tracker > es_patience:
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

        with torch.no_grad():
            test_pred = model(self.Xtest)
            loss = metric(test_pred, self.ytest)
            test_error_rmse = np.sqrt(loss.item()*factor)* hartree2cm 
            tmp_pred = model(self.Xvalid) 
            loss = metric(tmp_pred, self.yvalid)
            val_error_rmse = np.sqrt(loss.item() * factor) * hartree2cm
        #print(l, test_error_rmse)
        print("Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f}".format( test_error_rmse, val_error_rmse))
        return test_error_rmse, val_error_rmse

    def hyperopt_model(self, params):
        # skip building this model if hyperparameter combination already attempted
        for i in self.hyperopt_trials.results:
            if 'memo' in i:
                if params == i['memo']:
                    return {'loss': i['loss'], 'status': STATUS_OK, 'memo': 'repeat'}
        if self.itercount > self.hp_maxit:
            return {'loss': 0.0, 'status': STATUS_FAIL, 'memo': 'max iters reached'}
        error_test, error_valid = self.build_model(params)
        #error_test = self.vet_model(self.model)
        self.itercount += 1
        if np.isnan(error_valid):
            return {'loss': error_valid, 'status': STATUS_FAIL, 'memo': params}
        else:
            return {'loss': error_valid, 'status': STATUS_OK, 'memo': params}

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



    def save_model(self):
        pass








            
        




