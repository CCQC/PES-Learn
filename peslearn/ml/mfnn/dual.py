import torch
import numpy as np
from ..neural_network import NeuralNetwork
import os
from copy import deepcopy
from ...constants import package_directory
from ..preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler
from sklearn.model_selection import train_test_split 

torch.set_printoptions(precision=15)

class DualNN(NeuralNetwork):
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None, valid_path=None):
        #super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        self.trial_layers = self.input_obj.keywords['nas_trial_layers']
        self.set_default_hyperparameters()
        
        if self.input_obj.keywords['validation_points']:
            self.nvalid = self.input_obj.keywords['validation_points']
            if (self.nvalid + self.ntrain + 1) > self.n_datapoints:
                raise Exception("Error: User-specified training set size and validation set size exceeds the size of the dataset.")
        else:
            self.nvalid = round((self.n_datapoints - self.ntrain)  / 2)
        
        if self.pip:
            if molecule_type:
                path = os.path.join(package_directory, "lib", molecule_type, "output")
                self.inp_dim = len(open(path).readlines())+1
            if molecule:
                path = os.path.join(package_directory, "lib", molecule.molecule_type, "output")
                self.inp_dim = len(open(path).readlines())+1
        else:
            self.inp_dim = self.raw_X.shape[1]

    def split_train_test(self, params, validation_size=None, precision=32):
        self.X, self.y, self.Xscaler, self.yscaler, self.lf_E_scaler = self.preprocess(params, self.raw_X, self.raw_y)
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
            #TODO: this is splitting validation data in the same way at every model build, not necessary.
            self.valid_indices, self.new_test_indices = train_test_split(self.test_indices, train_size = validation_size, random_state=42)
            if validation_size:
                self.Xvalid = self.X[self.valid_indices]             
                self.yvalid = self.y[self.valid_indices]
                self.Xtest = self.X[self.new_test_indices]
                self.ytest = self.y[self.new_test_indices]

            else:
                raise Exception("Please specify a validation set size for Neural Network training.")

        # convert to Torch Tensors
        if precision == 32:
            self.Xtr    = torch.tensor(self.Xtr,   dtype=torch.float32)
            self.ytr    = torch.tensor(self.ytr,   dtype=torch.float32)
            self.Xtest  = torch.tensor(self.Xtest, dtype=torch.float32)
            self.ytest  = torch.tensor(self.ytest, dtype=torch.float32)
            self.Xvalid = torch.tensor(self.Xvalid,dtype=torch.float32)
            self.yvalid = torch.tensor(self.yvalid,dtype=torch.float32)
            self.X = torch.tensor(self.X,dtype=torch.float32)
            self.y = torch.tensor(self.y,dtype=torch.float32)
        elif precision == 64:
            self.Xtr    = torch.tensor(self.Xtr,   dtype=torch.float64)
            self.ytr    = torch.tensor(self.ytr,   dtype=torch.float64)
            self.Xtest  = torch.tensor(self.Xtest, dtype=torch.float64)
            self.ytest  = torch.tensor(self.ytest, dtype=torch.float64)
            self.Xvalid = torch.tensor(self.Xvalid,dtype=torch.float64)
            self.yvalid = torch.tensor(self.yvalid,dtype=torch.float64)
            self.X = torch.tensor(self.X,dtype=torch.float64)
            self.y = torch.tensor(self.y,dtype=torch.float64)
        else:
            raise Exception("Invalid option for 'precision'")

    def preprocess(self, params, raw_X_less, raw_y):
        """
        Preprocess raw data according to hyperparameters
        """
        lf_E = deepcopy(raw_X_less[:,-1].reshape(-1,1))
        raw_X = deepcopy(raw_X_less[:,:-1])
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])
        if params['pip']['pip']:
            # find path to fundamental invariants form molecule type AxByCz...
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            #lf_E = raw_X[:,-1]
            raw_X, degrees = interatomics_to_fundinvar(raw_X,path)
            #raw_X = np.hstack((raw_X, lf_E[:,None]))
            if params['pip']['degree_reduction']:
                #raw_X[:,:-1] = degree_reduce(raw_X[:,:-1], degrees)
                raw_X = degree_reduce(raw_X, degrees)
        if params['scale_X']:
            X, Xscaler = general_scaler(params['scale_X']['scale_X'], raw_X)
        else:
            X = raw_X
            Xscaler = None
        if params['scale_y']:
            lf_E, lf_E_scaler = general_scaler(params['scale_y'], lf_E)
            y, yscaler = general_scaler(params['scale_y'], raw_y)
        else:
            lf_E_scaler = None
            y = raw_y
            yscaler = None
        X = np.hstack((X, lf_E))
        #X = np.hstack((X, lf_E[:,None]))
        return X, y, Xscaler, yscaler, lf_E_scaler
    
    def transform_new_X(self, newX, params, Xscaler=None, lf_E_scaler=None):
        """
        Transform a new, raw input according to the model's transformation procedure 
        so that prediction can be made.
        """
        # ensure X dimension is n x m (n new points, m input variables)
        if len(newX.shape) == 1:
            newX = np.expand_dims(newX,0)
        elif len(newX.shape) > 2:
            raise Exception("Dimensions of input data is incorrect.")
        newX_geom = newX[:,:-1]
        lf_E = newX[:,-1].reshape(-1,1)
        if params['morse_transform']['morse']:
            newX_geom = morse(newX_geom, params['morse_transform']['morse_alpha'])
        if params['pip']['pip']:
            # find path to fundamental invariants for an N atom system with molecule type AxByCz...
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            newX_geom, degrees = interatomics_to_fundinvar(newX_geom,path)
            if params['pip']['degree_reduction']:
                newX_geom = degree_reduce(newX_geom, degrees)
        if Xscaler:
            newX_geom = Xscaler.transform(newX_geom)
        if lf_E_scaler:
            lf_E = lf_E_scaler.transform(lf_E)
        #lf_E = lf_E.reshape(-1,1)
        return np.hstack((newX_geom, lf_E))
