import numpy as np
import sklearn.metrics
import json
import os
from .model import Model
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
import GPy
from .data_sampler import DataSampler 
from .constants import hartree2cm, package_directory 
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler

class GaussianProcess(Model):
    """
    Constructs a Gaussian Process Model using GPy
    """
    def __init__(self, dataset_path, input_obj, mol_obj):
        super().__init__(dataset_path, input_obj, mol_obj)

    def optimize_model(self):
        print("Beginning hyperparameter optimization...")
        print("Trying {} combinations of hyperparameters".format(self.hp_max_evals))
        print("Training with {} points".format(self.ntrain))
        print("Using {} training set point sampling.".format(self.sampler))
        self.hyperopt_trials = Trials()
        self.set_hyperparameter_space()
        best = fmin(self.hyperopt_model,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.hp_max_evals,
                    trials=self.hyperopt_trials)
        print("everythings fine")
        self.print_hp_banner()
        print("Best performing hyperparameters are:")
        final = space_eval(self.hyperparameter_space, best)
        print(str(sorted(final.items())))
        self.optimal_hyperparameters  = dict(final)
        # obtain final model from best hyperparameters
        print("Fine-tuning final model architecture...")
        best_model = self.build_model(self.optimal_hyperparameters, nrestarts=10)
        print("Final model performance (cm-1):")
        self.vet_model(best_model)
        # Save model
        # Currently GPy requires saving training data in model for some reason. 
        # Repeated calls can be costly if a lot of training points are used.
        model_dict = best_model.to_dict(save_data=True)
        print("Saving ML model to file 'model.json'...")
        with open('model.json', 'w') as f:
            json.dump(model_dict, f)
    
    def vet_model(self, model):
        """Convenience method for getting model errors of test and full datasets"""
        pred_test = self.predict(model, self.Xtest)
        pred_full = self.predict(model, self.X)
        error_test = self.compute_error(self.Xtest, self.ytest, pred_test, self.yscaler)
        error_full = self.compute_error(self.X, self.y, pred_full, self.yscaler)
        print("Test Dataset {}".format(round(hartree2cm * error_test,2)), end='    ')
        print("Full Dataset {}".format(round(hartree2cm * error_full,2)))
        return error_test

    def split_train_test(self, params):
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
        """
        self.X, self.y, self.Xscaler, self.yscaler = self.preprocess(params, self.raw_X, self.raw_y)
        self.Xtr = self.X[self.train_indices]
        self.ytr = self.y[self.train_indices]
        self.Xtest = self.X[self.test_indices]
        self.ytest = self.y[self.test_indices]

    def build_model(self, params, nrestarts=5):
        # build train test sets
        self.split_train_test(params)
        # make GPy deterministic
        np.random.seed(0)
        dim = self.X.shape[1]
        # TODO add HP control 
        kernel = GPy.kern.RBF(dim, ARD=True) #TODO add more kernels to hyperopt space
        model = GPy.models.GPRegression(self.Xtr, self.ytr, kernel=kernel, normalizer=False)
        model.optimize(max_iters=600)
        model.optimize_restarts(nrestarts, optimizer="bfgs", verbose=False, max_iters=1000)
        return model

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
            model = self.build_model(params)
            print(params) # debugging
            error_test = self.vet_model(model)
            return {'loss': error_test, 'status': STATUS_OK, 'memo': params}

    def predict(self, model, data_in):
        prediction, v1 = model.predict(data_in, full_cov=False)
        return prediction 
     
    def compute_error(self, X, y, prediction, yscaler, max_errors=None):
        """
        Predict the root-mean-square error of model given 
        known X,y, a prediction, and a y scaling object, if it exists.
        
        Parameters
        ----------
        X : array
            Array of model inputs (geometries)
        y : array
            Array of expected model outputs (energies)
        prediction: array
            Array of actual model outputs (energies)
        yscaler: object
            Sci-kit learn scaler object
        max_errors: int
            Returns largest (int) absolute maximum errors 

        Returns
        -------
        """
        if yscaler:
            raw_y = yscaler.inverse_transform(y)
            unscaled_prediction = yscaler.inverse_transform(prediction)
            error = np.sqrt(sklearn.metrics.mean_squared_error(raw_y,  unscaled_prediction))
            if max_errors:
                e = np.abs(raw_y - unscaled_prediction) * hartree2cm
                largest_errors = np.partition(e, -max_errors, axis=0)[-max_errors:]
        else:
            error = np.sqrt(sklearn.metrics.mean_squared_error(y, prediction))
            if max_errors:
                e = np.abs(y, prediction) * hartree2cm
                largest_errors = np.partition(e, -max_errors, axis=0)[-max_errors:]
        if max_errors:
            return error, largest_errors
        else:
            return error

    def set_hyperparameter_space(self, manual_hp_space=None):
        """
        Setter for hyperparameter space. If none is provided, default is used.
        """
        if manual_hp_space:
            self.hyperparameter_space = manual_hp_space
        else:
            self.hyperparameter_space = {
                      'fi_transform': hp.choice('fi_transform',
                                    [
                                    #{'fi': True,
                                    {'fi': True,
                                        'degree_reduction': False},
                                    #{'fi': False,
                                    #    'degree_reduction': False},
                                        #'degree_reduction': hp.choice('degree_reduction', [True,False])},
                                    #{'fi': False}
                                    ]),
                      'morse_transform': hp.choice('morse_transform',
                                    [
                                    {'morse': True,
                                        'morse_alpha': hp.uniform('morse_alpha', 1.0, 2.0)},
                                    {'morse': False}
                                    ]),
                      'scale_X': hp.choice('scale_X', ['std', 'mm01', 'mm11', None]),
                      'scale_y': hp.choice('scale_y', ['std', 'mm01', 'mm11', None]),
                      }
             #TODO add optional space inclusions 
             # something like: if option: self.hyperparameter_space['newoption'] = hp.choice(..)

    def preprocess(self, params, raw_X, raw_y):
        """
        Preprocess raw data according to hyperparameters
        """
        # TODO make more flexible. If keys don't exist, ignore them. smth like "if key: if param['key']: do transform"
        # Transform to morse variables (exp(-r/alpha))
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])
        # Transform to FIs, degree reduce if called
        if params['fi_transform']['fi']:
            # find path to fundamental invariants for an N atom system with molecule type AxByCz...
            path = os.path.join(package_directory, "lib", str(sum(self.mol.atom_count_vector))+"_atom_system", self.mol.molecule_type, "output")
            raw_X, degrees = interatomics_to_fundinvar(raw_X,path)
            if params['fi_transform']['degree_reduction']:
                raw_X = degree_reduce(raw_X, degrees)
        
        # Scaling
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
        # Transform to FIs, degree reduce if called
        if params['fi_transform']['fi']:
            # find path to fundamental invariants for an N atom system with molecule type AxByCz...
            path = os.path.join(package_directory, "lib", str(sum(self.mol.atom_count_vector))+"_atom_system", self.mol.molecule_type, "output")
            newX, degrees = interatomics_to_fundinvar(newX,path)
            if params['fi_transform']['degree_reduction']:
                newX = degree_reduce(newX, degrees)

        if Xscaler:
            newX = Xscaler.transform(newX)
        return newX

    def inverse_transform_new_y(self, newy, yscaler=None):    
        if yscaler:
            newy = yscaler.inverse_transform(newy)
        return newy

    def print_hp_banner(self):
        print("\n###################################################")
        print("#                                                 #")
        print("#     Hyperparameter Optimization Complete!!!     #")
        print("#                                                 #")
        print("###################################################\n")
