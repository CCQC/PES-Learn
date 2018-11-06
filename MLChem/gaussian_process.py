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
    def __init__(self, dataset_path, ntrain, input_obj, mol_obj):
        super().__init__(dataset_path, ntrain, input_obj, mol_obj)
        self.raw_X = self.dataset.values[:, :-1]
        self.raw_y = self.dataset.values[:,-1].reshape(-1,1)
        self.mol = mol_obj

    def optimize_model(self):
        self.hyperopt_trials = Trials()
        self.set_hyperparameter_space()
        best = fmin(self.hyperopt_model,
                    space=self.hyperparameter_space,
                    algo=tpe.suggest,
                    max_evals=self.hp_max_evals,
                    trials=self.hyperopt_trials)
        print("\n###################################################")
        print("#                                                 #")
        print("#     Hyperparameter Optimization Complete!!!     #")
        print("#                                                 #")
        print("###################################################\n")
        print("Best performing hyperparameters are:")
        final = space_eval(self.hyperparameter_space, best)
        print(str(sorted(final.items())))
        print("Best model performance (cm-1):")
        params = dict(final)
        model, X, y, Xtr, ytr, Xtest, ytest, Xscaler, yscaler = self.build_model(params)
        # Save best model
        # Currently GPy requires saving data for some reason. Repeated calls can be costly if a lot of training points are used.
        model_dict = model.to_dict(save_data=True)
        with open('model.json', 'w') as f:
            json.dump(model_dict, f)

        pred_test = self.predict(model, Xtest)
        pred_full = self.predict(model, X)
        error_test = self.compute_error(Xtest, ytest, pred_test, yscaler)
        error_full = self.compute_error(X, y, pred_full, yscaler)
        print("Test Dataset {}".format(round(hartree2cm * error_test,2)))
        print("Full Dataset {}".format(round(hartree2cm * error_full,2)))
        self.optimal_hyperparameters = params

    def build_model(self, params, nrestarts=5):
        X, y, Xscaler, yscaler = self.preprocess(params, self.raw_X, self.raw_y)
        # for debugging 
        #if Xscaler:
        #    print("X Means ",Xscaler.mean_)
        #    print("X Scales ", Xscaler.scale_)
        #if yscaler:
        #    print("y Means ",yscaler.mean_)
        #    print("y Scales ", yscaler.scale_)
    
        if self.input_obj.keywords['n_low_energy_train']:
            n =  self.input_obj.keywords['n_low_energy_train']
            sample = DataSampler(self.dataset, self.ntrain, accept_first_n=n)
        else:
            sample = DataSampler(self.dataset, self.ntrain)

        if self.sampler == 'random':
            sample.smart_random()
        elif self.sampler == 'smart_random':
            sample.smart_random()
        elif self.sampler == 'structure_based':
            sample.structure_based()

        train, test = sample.get_indices()
        Xtr = X[train]
        print(len(train))
        ytr = y[train]
        Xtest = X[test]
        ytest = y[test]
        
        # make GPy deterministic
        np.random.seed(0)
        dim = X.shape[1]
        kernel = GPy.kern.RBF(dim, ARD=True) #TODO add more kernels to hyperopt space
        model = GPy.models.GPRegression(Xtr, ytr, kernel=kernel, normalizer=False)
        model.optimize(max_iters=600)
        model.optimize_restarts(nrestarts, optimizer="bfgs", verbose=False, max_iters=1000)
        return (model, X, y, Xtr, ytr, Xtest, ytest, Xscaler, yscaler)

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
            model, X, y, Xtr, ytr, Xtest, ytest, Xscaler, yscaler = self.build_model(params)
            print(params)
            pred_test = self.predict(model, Xtest)
            pred_full = self.predict(model, X)
            error_test = self.compute_error(Xtest, ytest, pred_test, yscaler)
            error_full = self.compute_error(X, y, pred_full, yscaler)
            print("Test Dataset {}".format(round(hartree2cm * error_test,2)), end='   ')
            print("Full Dataset {}".format(round(hartree2cm * error_full,2)))
            return {'loss': error_test, 'status': STATUS_OK, 'memo': params}



    def predict(self, model, X):
        prediction, v1 = model.predict(X, full_cov=False)
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

    def set_hyperparameter_space(self):
        self.hyperparameter_space = {
                  'fi_transform': hp.choice('fi_transform',
                                [
                                #{'fi': True,
                                {'fi': True,
                                    'degree_reduction': False},
                                    #'degree_reduction': hp.choice('degree_reduction', [True,False])},
                                #{'fi': False}
                                ]),
                  'morse_transform': hp.choice('morse_transform',
                                [
                                {'morse': True,
                                    'morse_alpha': hp.uniform('morse_alpha', 1.0, 2.0)},
                                {'morse': False}
                                ]),
                  'scale_X': hp.choice('scale_X', ['std']),#, 'mm01', 'mm11', None]),
                  'scale_y': hp.choice('scale_y', ['std', 'mm01', 'mm11', None]),
                  }
         #TODO add optional space inclusions 
         # something like: if option: self.hyperparameter_space['newoption'] = hp.choice(..)

    def preprocess(self, params, raw_X, raw_y):
        """
        Preprocess data according to hyperparameters
        """
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

