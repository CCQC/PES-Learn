import numpy as np
import sklearn.metrics
import json
import os
import re
import sys
import gc
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from GPy.models import GPRegression
from GPy.kern import RBF

from .model import Model
from ..constants import hartree2cm, package_directory, gp_convenience_function
from ..utils.printing_helper import hyperopt_complete
from ..lib.path import fi_dir
from .data_sampler import DataSampler 
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler

class GaussianProcess(Model):
    """
    Constructs a Gaussian Process Model using GPy
    """
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)
        self.set_default_hyperparameters()
    
    def set_default_hyperparameters(self):
        """
        Set default hyperparameter space. If none is provided, default is used.
        """
        self.hyperparameter_space = {
                                    #'scale_X': hp.choice('scale_X', ['std', 'mm01', 'mm11', None]),
                                    'scale_y': hp.choice('scale_y', ['std', 'mm01', 'mm11', None]),
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

        if self.input_obj.keywords['gp_ard'] == 'opt': # auto relevancy determination (independant length scales for each feature)
            self.set_hyperparameter('ARD', hp.choice('ARD', [True,False]))
         #TODO add optional space inclusions, something like: if option: self.hyperparameter_space['newoption'] = hp.choice(..)

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
        if self.sampler == 'user_supplied':
            self.Xtr = self.transform_new_X(self.raw_Xtr, params, self.Xscaler)
            self.ytr = self.transform_new_y(self.raw_ytr, self.yscaler)
            self.Xtest = self.transform_new_X(self.raw_Xtest, params, self.Xscaler)
            self.ytest = self.transform_new_y(self.raw_ytest, self.yscaler)
            
        else:
            self.Xtr = self.X[self.train_indices]
            self.ytr = self.y[self.train_indices]
            self.Xtest = self.X[self.test_indices]
            self.ytest = self.y[self.test_indices]

    def build_model(self, params, nrestarts=10, maxit=1000, seed=0):
        params['scale_X'] = 'std'
        print("Hyperparameters: ", params)
        self.split_train_test(params)
        np.random.seed(seed)     # make GPy deterministic for a given hyperparameter config
        dim = self.X.shape[1]
        if self.input_obj.keywords['gp_ard'] == 'opt':
            ard_val = params['ARD']
        elif self.input_obj.keywords['gp_ard'] == 'true':
            ard_val = True
        else:
            ard_val = False
        kernel = RBF(dim, ARD=ard_val)  # TODO add HP control of kernel
        self.model = GPRegression(self.Xtr, self.ytr, kernel=kernel, normalizer=False)
        #self.model.optimize_restarts(nrestarts, optimizer="lbfgsb", robust=True, verbose=False, max_iters=maxit, messages=False)
        self.model.optimize(optimizer="lbfgsb", max_iters=maxit, messages=False)
        #TODO
        err = self.vet_model(self.model)
        #TODO
        gc.collect(2) #fixes some memory leak issues with certain BLAS configs

    def hyperopt_model(self, params):
        # skip building this model if hyperparameter combination already attempted
        for i in self.hyperopt_trials.results:
            if 'memo' in i:
                if params == i['memo']:
                    return {'loss': i['loss'], 'status': STATUS_OK, 'memo': 'repeat'}
        if self.itercount > self.hp_maxit:
            return {'loss': 0.0, 'status': STATUS_FAIL, 'memo': 'max iters reached'}
        self.build_model(params)
        error_test = self.vet_model(self.model)
        self.itercount += 1
        return {'loss': error_test, 'status': STATUS_OK, 'memo': params}

    def predict(self, model, data_in):
        prediction, v1 = model.predict(data_in, full_cov=False)
        return prediction 

    def vet_model(self, model):
        """Convenience method for getting model errors of test and full datasets"""
        pred_test = self.predict(model, self.Xtest)
        pred_full = self.predict(model, self.X)
        error_test = self.compute_error(self.ytest, pred_test, self.yscaler)
        error_full, median_error, max_errors = self.compute_error(self.y, pred_full, yscaler=self.yscaler, max_errors=5)
        print("Test Dataset {}".format(round(hartree2cm * error_test,2)), end='  ')
        print("Full Dataset {}".format(round(hartree2cm * error_full,2)), end='     ')
        print("Median error: {}".format(np.round(median_error[0],2)), end='  ')
        print("Max 5 errors: {}".format(np.sort(np.round(max_errors.flatten(),1))),'\n')
        return error_test
     
    def preprocess(self, params, raw_X, raw_y):
        """
        Preprocess raw data according to hyperparameters
        """
        # TODO make more flexible. If keys don't exist, ignore them. smth like "if key: if param['key']: do transform"
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])  # Transform to morse variables (exp(-r/alpha))
        # Transform to FIs, degree reduce if called 
        if params['pip']['pip']:
            # find path to fundamental invariants form molecule type AxByCz...
            #path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            path = os.path.join(fi_dir, self.molecule_type, "output")
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
    
    def optimize_model(self):
        print("Beginning hyperparameter optimization...")
        print("Trying {} combinations of hyperparameters".format(self.hp_maxit))
        print("Training with {} points (Full dataset contains {} points).".format(self.ntrain, self.n_datapoints))
        print("Using {} training set point sampling.".format(self.sampler))
        print("Errors are root-mean-square error in wavenumbers (cm-1)")
        self.hyperopt_trials = Trials()
        self.itercount = 1  # keep track of hyperopt iterations 
        if self.input_obj.keywords['rseed']:
            rstate = np.random.default_rng(self.input_obj.keywords['rseed'])
            #rstate = np.random.RandomState(self.input_obj.keywords['rseed'])
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
        # obtain final model from best hyperparameters
        print("Fine-tuning final model architecture...")
        self.build_model(self.optimal_hyperparameters, nrestarts=10, maxit=1000)
        print("Final model performance (cm-1):")
        self.test_error = self.vet_model(self.model)
        self.save_model(self.optimal_hyperparameters)

    def save_model(self, params):
        # Save model. Currently GPy requires saving training data in model for some reason. 
        model_dict = self.model.to_dict(save_data=True)
        print("Saving ML model data...") 
        model_path = "model1_data"
        while os.path.isdir(model_path):
            new = int(re.findall("\d+", model_path)[0]) + 1
            model_path = re.sub("\d+",str(new), model_path)
        os.mkdir(model_path)
        os.chdir(model_path)
        with open('model.json', 'w') as f:
            json.dump(model_dict, f)
        with open('hyperparameters', 'w') as f:
            print(params, file=f)
        
        if self.sampler == 'user_supplied':
            self.traindata.to_csv('train_set',sep=',',index=False,float_format='%12.12f')
            self.testdata.to_csv('test_set', sep=',', index=False, float_format='%12.12f')
        else:
            self.dataset.iloc[self.train_indices].to_csv('train_set',sep=',',index=False,float_format='%12.12f')
            self.dataset.iloc[self.test_indices].to_csv('test_set', sep=',', index=False, float_format='%12.12f')
    
        self.dataset.to_csv('PES.dat', sep=',',index=False,float_format='%12.12f')
        # write convenience function
        with open('compute_energy.py', 'w+') as f:
            print(self.write_convenience_function(), file=f)

        # print model performance
        sys.stdout = open('performance', 'w')  
        self.vet_model(self.model)
        sys.stdout = sys.__stdout__
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
        string = "from peslearn.ml import GaussianProcess\nfrom peslearn import InputProcessor\nfrom GPy.core.model import Model\nimport numpy as np\nimport json\nfrom itertools import combinations\n\n"
        if self.pip:
            string += "gp = GaussianProcess('PES.dat', InputProcessor(''), molecule_type='{}')\n".format(self.molecule_type)
        else:
            string += "gp = GaussianProcess('PES.dat', InputProcessor(''))\n"
        with open('hyperparameters', 'r') as f:
            hyperparameters = f.read()
        string += "params = {}\n".format(hyperparameters)
        string += "X, y, Xscaler, yscaler =  gp.preprocess(params, gp.raw_X, gp.raw_y)\n"
        string += "model = Model('mymodel')\n"
        string += "with open('model.json', 'r') as f:\n"
        string += "    model_dict = json.load(f)\n"
        string += "final = model.from_dict(model_dict)\n\n"
        string += gp_convenience_function
        return string


