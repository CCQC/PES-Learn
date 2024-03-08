import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
import re
import copy

from .model import Model
from .data_sampler import DataSampler 
from ..constants import hartree2cm, package_directory, nn_convenience_function
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler
from ..utils.printing_helper import hyperopt_complete
from sklearn.model_selection import train_test_split   
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from .preprocessing_helper import sort_architectures


torch.set_printoptions(precision=15)

class MFModel(Model):
    """
    A class that handles data processing and other convenience functions for multifidelity models
    """
    def __init__(self, dataset_paths, input_objs, molecule_type=None, molecule=None, train_paths=(None, None), test_paths=(None, None), valid_paths=(None, None)): #All input objs are tuples ordered high to low fidelity. Only works with 2 for now
        print("Big BEEBUS")
        self.m_high = Model(dataset_paths[0], input_objs[0], molecule_type, molecule, train_paths[0], test_paths[0], valid_paths[0])
        self.m_low  = Model(dataset_paths[1], input_objs[1], molecule_type, molecule, train_paths[1], test_paths[1], valid_paths[1])
        self.molecule_type = molecule_type
        self.molecule = molecule
        self.set_default_hyperparameters()
        self.initModel(self.m_high)
        self.initModel(self.m_low)

    def initModel(self, m):
        # Because I do not want to write everything in __init__ twice
        m.trial_layers = m.input_obj.keywords['nas_trial_layers']
        if m.input_obj.keywords['validation_points']:
            m.nvalid = m.input_obj.keywords['validation_points']
            if (m.nvalid + m.ntrain + 1) > m.n_datapoints:
                raise Exception("Error: User-specified training set size and validation set size exceeds the size of the dataset.")
        else:
            m.nvalid = round((m.n_datapoints - m.ntrain)  / 2)
        
        if m.pip:
            if self.molecule_type:
                path = os.path.join(package_directory, "lib", self.molecule_type, "output")
                m.inp_dim = len(open(path).readlines())
            if self.molecule:
                path = os.path.join(package_directory, "lib", self.molecule.molecule_type, "output")
                m.inp_dim = len(open(path).readlines())
        else:
            m.inp_dim = m.raw_X.shape[1]

    def set_default_hyperparameters(self, nn_search_space=1):
        """
        Set default hyperparameter space. If none is provided, default is used.

        Parameters
        ----------
        nn_search_space : int
            Which tier of default hyperparameter search spaces to use. Neural networks have too many hyperparameter configurations to search across, 
            so this option reduces the number of variable hyperparameters to search over. Generally, larger integer => more hyperparameters, and more iterations of hp_maxit are recommended.
        """
        if nn_search_space == 1:
            self.hyperparameter_space = {
            'scale_X': hp.choice('scale_X',
                     [
                     {'scale_X': 'mm11',
                          'activation': hp.choice('activ2', ['tanh'])},
                     {'scale_X': 'std',
                          'activation': hp.choice('activ3', ['tanh'])},
                     ]),
            'scale_y': hp.choice('scale_y', ['std', 'mm01', 'mm11']),}
        # TODO make more expansive search spaces, benchmark them, expose them as input options
        #elif nn_search_space == 2:
        #elif nn_search_space == 3:
        else:
            raise Exception("Invalid search space specification")

        # Standard geometry transformations, always use these.
        if self.m_high.input_obj.keywords['pes_format'] == 'interatomics':
            self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': True,'morse_alpha': hp.quniform('morse_alpha', 1, 2, 0.1)},{'morse': False}]))
        else:
            self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': False}]))
        if self.m_high.pip:
            val =  hp.choice('pip',[{'pip': True,'degree_reduction': hp.choice('degree_reduction', [True,False])}])
            self.set_hyperparameter('pip', val)
        else:
            self.set_hyperparameter('pip', hp.choice('pip', [{'pip': False}]))

    def optimize_model(self):
        if not self.m_high.input_obj.keywords['validation_points']:
            print("Number of validation points not specified. Splitting test set in half --> 50% test, 50% validation")
        print("Training with {} points. Validating with {} points. Full dataset contains {} points.".format(self.m_high.ntrain, self.m_high.nvalid, self.m_high.n_datapoints))
        print("Using {} training set point sampling.".format(self.m_high.sampler))
        print("Errors are root-mean-square error in wavenumbers (cm-1)")
        print("Beginning hyperparameter optimization...")
        print("Trying {} combinations of hyperparameters".format(self.m_high.hp_maxit))
        self.hyperopt_trials = Trials()
        self.itercount = 1
        if self.m_high.input_obj.keywords['rseed']:
            rstate = np.random.default_rng(self.m_high.input_obj.keywords['rseed'])
            #rstate = np.random.RandomState(self.m_high.input_obj.keywords['rseed'])
        else:
            rstate = None
        best = fmin(self.hyperopt_model,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.m_high.hp_maxit*2,
                    rstate=rstate, 
                    show_progressbar=False,
                    trials=self.hyperopt_trials)
        hyperopt_complete()
        print("Best performing hyperparameters are:")
        final = space_eval(self.hyperparameter_space, best)
        print(str(sorted(final.items())))
        self.optimal_hyperparameters  = dict(final)
        print("Optimizing learning rate...")
        
        if self.m_high.input_obj.keywords['nn_precision'] == 64:
            precision = 64
        else: 
            precision = 32
        learning_rates = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2]
        val_errors = []
        for i in learning_rates:
            self.optimal_hyperparameters['lr'] = i
            test_error, val_error = self.build_model(self.optimal_hyperparameters, maxit=5000, val_freq=10, es_patience=5, opt='lbfgs', tol=0.5,  decay=False, verbose=False, precision=precision)
            val_errors.append(val_error)
        best_lr = learning_rates[np.argsort(val_errors)[0]]
        self.optimal_hyperparameters['lr'] = best_lr
        print("Fine-tuning final model...")
        model, test_error, val_error, full_error = self.build_model(self.optimal_hyperparameters, maxit=5000, val_freq=1, es_patience=100, opt='lbfgs', tol=0.1,  decay=True, verbose=True,precision=precision,return_model=True)
        performance = [test_error, val_error, full_error]
        print("Model optimization complete. Saving final model...")
        self.save_model(self.optimal_hyperparameters, model, performance)

    def preprocess_protocol(self, params):
        """
        How do you want to handle preprocessing? Choose your own adventure!!!
        Need to set values for:
            self.X_h
            self.y_h
            self.Xscaler_h
            self.yscaler_h
            self.X_l
            self.y_l
            self.Xscaler_l
            self.yscaler_l
        """
        
        # Default
        self.X_h, self.y_h, self.Xscaler_h, self.yscaler_h = self.preprocess(params, self.m_high.raw_X, self.m_high.raw_y)
        self.X_l, self.y_l, self.Xscaler_l, self.yscaler_l = self.preprocess(params, self.m_low.raw_X, self.m_low.raw_y)
        

    def split_train_test(self, params, precision=32):
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

        self.preprocess_protocol(params)
        if self.m_high.sampler == 'user_supplied':
            self.Xtr_h = self.transform_new_X(self.m_high.raw_Xtr, params, self.Xscaler_h)
            self.ytr_h = self.transform_new_y(self.m_high.raw_ytr, self.yscaler_h)
            self.Xtest_h = self.transform_new_X(self.m_high.raw_Xtest, params, self.Xscaler_h)
            self.ytest_h = self.transform_new_y(self.m_high.raw_ytest, self.yscaler_h)
            if self.m_high.valid_path:
                self.Xvalid_h = self.transform_new_X(self.m_high.raw_Xvalid, params, self.Xscaler_h)
                self.yvalid_h = self.transform_new_y(self.m_high.raw_yvalid, self.yscaler_h)
            else:
                raise Exception("Please provide a validation set for Neural Network training.")
        else:
            self.Xtr_h = self.X_h[self.m_high.train_indices]
            self.ytr_h = self.y_h[self.m_high.train_indices]
            #TODO: this is splitting validation data in the same way at every model build, not necessary.
            self.valid_indices_h, self.new_test_indices_h = train_test_split(self.m_high.test_indices, train_size = self.m_high.nvalid, random_state=42)
            if self.m_high.nvalid:
                self.Xvalid_h = self.X_h[self.valid_indices_h]             
                self.yvalid_h = self.y_h[self.valid_indices_h]
                self.Xtest_h = self.X_h[self.new_test_indices_h]
                self.ytest_h = self.y_h[self.new_test_indices_h]

            else:
                raise Exception("Please specify a validation set size for Neural Network training.")

        if self.m_low.sampler == 'user_supplied':
            self.Xtr_l = self.transform_new_X(self.m_low.raw_Xtr, params, self.Xscaler_l)
            self.ytr_l = self.transform_new_y(self.m_low.raw_ytr, self.yscaler_l)
            self.Xtest_l = self.transform_new_X(self.m_low.raw_Xtest, params, self.Xscaler_l)
            self.ytest_l = self.transform_new_y(self.m_low.raw_ytest, self.yscaler_l)
            if self.m_low.valid_path:
                self.Xvalid_l = self.transform_new_X(self.m_low.raw_Xvalid, params, self.Xscaler_l)
                self.yvalid_l = self.transform_new_y(self.m_low.raw_yvalid, self.yscaler_l)
            else:
                raise Exception("Please provide a validation set for Neural Network training.")
        else:
            self.Xtr_l = self.X_l[self.m_low.train_indices]
            self.ytr_l = self.y_l[self.m_low.train_indices]
            #TODO: this is splitting validation data in the same way at every model build, not necessary.
            self.valid_indices_l, self.new_test_indices_l = train_test_split(self.m_low.test_indices, train_size = self.m_low.nvalid, random_state=42)
            if self.m_low.nvalid:
                self.Xvalid_l = self.X_l[self.valid_indices_l]             
                self.yvalid_l = self.y_l[self.valid_indices_l]
                self.Xtest_l = self.X_l[self.new_test_indices_l]
                self.ytest_l = self.y_l[self.new_test_indices_l]

            else:
                raise Exception("Please specify a validation set size for Neural Network training.")
        
        # convert to Torch Tensors
        if precision == 32:
            self.Xtr_h    = torch.tensor(self.Xtr_h,    dtype=torch.float32)
            self.ytr_h    = torch.tensor(self.ytr_h,    dtype=torch.float32)
            self.Xtest_h  = torch.tensor(self.Xtest_h,  dtype=torch.float32)
            self.ytest_h  = torch.tensor(self.ytest_h,  dtype=torch.float32)
            self.Xvalid_h = torch.tensor(self.Xvalid_h, dtype=torch.float32)
            self.yvalid_h = torch.tensor(self.yvalid_h, dtype=torch.float32)
            self.X_h      = torch.tensor(self.X_h,      dtype=torch.float32)
            self.y_h      = torch.tensor(self.y_h,      dtype=torch.float32)

            self.Xtr_l    = torch.tensor(self.Xtr_l,    dtype=torch.float32)
            self.ytr_l    = torch.tensor(self.ytr_l,    dtype=torch.float32)
            self.Xtest_l  = torch.tensor(self.Xtest_l,  dtype=torch.float32)
            self.ytest_l  = torch.tensor(self.ytest_l,  dtype=torch.float32)
            self.Xvalid_l = torch.tensor(self.Xvalid_l, dtype=torch.float32)
            self.yvalid_l = torch.tensor(self.yvalid_l, dtype=torch.float32)
            self.X_l      = torch.tensor(self.X_l,      dtype=torch.float32)
            self.y_l      = torch.tensor(self.y_l,      dtype=torch.float32)

        elif precision == 64:
            self.Xtr_h    = torch.tensor(self.Xtr_h,    dtype=torch.float64)
            self.ytr_h    = torch.tensor(self.ytr_h,    dtype=torch.float64)
            self.Xtest_h  = torch.tensor(self.Xtest_h,  dtype=torch.float64)
            self.ytest_h  = torch.tensor(self.ytest_h,  dtype=torch.float64)
            self.Xvalid_h = torch.tensor(self.Xvalid_h, dtype=torch.float64)
            self.yvalid_h = torch.tensor(self.yvalid_h, dtype=torch.float64)
            self.X_h      = torch.tensor(self.X_h,      dtype=torch.float64)
            self.y_h      = torch.tensor(self.y_h,      dtype=torch.float64)

            self.Xtr_l    = torch.tensor(self.Xtr_l,    dtype=torch.float64)
            self.ytr_l    = torch.tensor(self.ytr_l,    dtype=torch.float64)
            self.Xtest_l  = torch.tensor(self.Xtest_l,  dtype=torch.float64)
            self.ytest_l  = torch.tensor(self.ytest_l,  dtype=torch.float64)
            self.Xvalid_l = torch.tensor(self.Xvalid_l, dtype=torch.float64)
            self.yvalid_l = torch.tensor(self.yvalid_l, dtype=torch.float64)
            self.X_l      = torch.tensor(self.X_l,      dtype=torch.float64)
            self.y_l      = torch.tensor(self.y_l,      dtype=torch.float64)

        else:
            raise Exception("Invalid option for 'precision'")

    def get_optimizer(self, opt_type, mdata, lr=0.1): 
        rate = lr
        if opt_type == 'lbfgs':
            #optimizer = torch.optim.LBFGS(mdata, lr=rate, max_iter=20, max_eval=None, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100) # Defaults
            #optimizer = torch.optim.LBFGS(mdata, lr=rate, max_iter=100, max_eval=None, tolerance_grad=1e-10, tolerance_change=1e-14, history_size=200)
            optimizer = torch.optim.LBFGS(mdata, lr=rate, max_iter=20, max_eval=None, tolerance_grad=1e-8, tolerance_change=1e-12, history_size=100)
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(mdata, lr=rate)
        return optimizer

    def hyperopt_model(self, params):
        """
        A Hyperopt-friendly wrapper for build_model
        """
        # skip building this model if hyperparameter combination already attempted
        for i in self.hyperopt_trials.results:
            if 'memo' in i:
                if params == i['memo']:
                    return {'loss': i['loss'], 'status': STATUS_OK, 'memo': 'repeat'}
        if self.itercount > self.m_high.hp_maxit:
            return {'loss': 0.0, 'status': STATUS_FAIL, 'memo': 'max iters reached'}
        error_test, error_valid = self.build_model(params)
        self.itercount += 1
        if np.isnan(error_valid):
            return {'loss': 1e5, 'status': STATUS_FAIL, 'memo': 'nan'}
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

    def save_model(self, params, model, performance):
        print("Saving ML model data...") 
        model_path = "model1_data"
        while os.path.isdir(model_path):
            new = int(re.findall("\d+", model_path)[0]) + 1
            model_path = re.sub("\d+",str(new), model_path)
        os.mkdir(model_path)
        os.chdir(model_path)
        torch.save(model, 'model.pt')
        
        with open('hyperparameters', 'w') as f:
            print(params, file=f)

        test, valid, full = performance
        with open('performance', 'w') as f:
            print("Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f} Full dataset RMSE (cm-1): {:5.2f}".format(test, valid, full), file=f)
        
        if self.m_high.sampler == 'user_supplied':
            self.m_high.traindata.to_csv('train_set',sep=',',index=False,float_format='%12.12f')
            self.m_high.validdata.to_csv('validation_set',sep=',',index=False,float_format='%12.12f')
            self.m_high.testdata.to_csv('test_set', sep=',', index=False, float_format='%12.12f')
        else:
            self.m_high.dataset.iloc[self.m_high.train_indices].to_csv('train_set',sep=',',index=False,float_format='%12.12f')
            self.m_high.dataset.iloc[self.valid_indices_h].to_csv('validation_set', sep=',', index=False, float_format='%12.12f')
            self.m_high.dataset.iloc[self.new_test_indices_h].to_csv('test_set', sep=',', index=False, float_format='%12.12f')
    
        self.m_high.dataset.to_csv('PES.dat', sep=',',index=False,float_format='%12.12f')
        with open('compute_energy.py', 'w+') as f:
            print(self.write_convenience_function(), file=f)
        os.chdir("../")

    def transform_new_X(self, newX, params, Xscaler=None):
        """
        Transform a new, raw input according to the model's transformation procedure 
        so that prediction can be made.
        """
        # ensure X dimension is n x m (n new points, m input variables)
        if len(newX.shape) == 1:
            newX = np.expand_dims(newX,0)
        elif len(newX.shape) > 2:
            raise Exception("Dimensions of input data is incorrect.")
        if params['morse_transform']['morse']:
            newX = morse(newX, params['morse_transform']['morse_alpha'])
        if params['pip']['pip']:
            # find path to fundamental invariants for an N atom system with molecule type AxByCz...
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            newX, degrees = interatomics_to_fundinvar(newX,path)
            if params['pip']['degree_reduction']:
                newX = degree_reduce(newX, degrees)
        if Xscaler:
            newX = Xscaler.transform(newX)
        return newX

    def transform_new_y(self, newy, yscaler=None):    
        if yscaler:
            newy = yscaler.transform(newy)
        return newy

    def inverse_transform_new_y(self, newy, yscaler=None):    
        if yscaler:
            newy = yscaler.inverse_transform(newy)
        return newy

    def write_convenience_function(self):
        string = "from peslearn.ml import NeuralNetwork\nfrom peslearn import InputProcessor\nimport torch\nimport numpy as np\nfrom itertools import combinations\n\n"
        if self.m_high.pip:
            string += "nn = NeuralNetwork('PES.dat', InputProcessor(''), molecule_type='{}')\n".format(self.molecule_type)
        else:
            string += "nn = NeuralNetwork('PES.dat', InputProcessor(''))\n"
        with open('hyperparameters', 'r') as f:
            hyperparameters = f.read()
        string += "params = {}\n".format(hyperparameters)
        string += "X, y, Xscaler, yscaler =  nn.preprocess(params, nn.raw_X, nn.raw_y)\n"
        string += "model = torch.load('model.pt')\n"
        string += nn_convenience_function
        return string




