import numpy as np
import sklearn.metrics
import os
import sys
import re
import json
from sklearn.kernel_ridge import KernelRidge
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval


from .model import Model
from..constants import package_directory, hartree2cm, krr_convenience_funciton
from ..utils.printing_helper import hyperopt_complete
from ..lib.path import fi_dir
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler


class KernelRidgeReg(Model):
    """
    Constructs a Kernel Ridge Regression Model using scikit-learn
    """
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None, valid_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        self.set_default_hyperparameters()

    def set_default_hyperparameters(self):
        """
        Set default hyperparameter space. If none is provided, default is used.
        """
        self.hyperparameter_space = {
                                    'scale_X': hp.choice('scale_X', ['std', 'mm01', 'mm11', None]),
                                    'scale_y': hp.choice('scale_y', ['std', 'mm01', 'mm11', None]),
                                    }
        
        # Standard geometry transformations, always use these.
        if self.input_obj.keywords['pes_format'] == 'interatomics':
            self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': True,'morse_alpha': hp.quniform('morse_alpha', 1, 2, 0.1)},{'morse': False}]))
        else:
            self.set_hyperparameter('morse_transform', hp.choice('morse_transform',[{'morse': False}]))
        if self.pip:
            val =  hp.choice('pip',[{'pip': True,'degree_reduction': hp.choice('degree_reduction', [True,False])}])
            self.set_hyperparameter('pip', val)
        else:
            self.set_hyperparameter('pip', hp.choice('pip', [{'pip': False}]))

        # Kernel hyperparameters
        self.set_hyperparameter('alpha', hp.choice('alpha', [1e-06, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6]))

        # if 'kernel' keyword is 'None' (default) an rbf kernel will be used and only 'alpha' will be hyperparameter
        if self.input_obj.keywords['kernel'] == None:
            self.set_hyperparameter('kernel', hp.choice('kernel',[{'ktype': 'rbf', 'gamma': None, 'degree': None}]))

        # if 'kernel' keyword is 'verbose' choice of kernel will be hyperparameter
        elif self.input_obj.keywords['kernel'] == 'verbose':
            self.set_hyperparameter('kernel', hp.choice('kernel', [
                # {'ktype': 'chi2', 'gamma': hp.quniform('gamma', 0.5, 1.5, 0.1), 'degree': None},
                {'ktype': 'polynomial', 'gamma': None, 'degree': hp.quniform('degree', 1, 5, 1)},
                {'ktype': 'rbf', 'gamma': None, 'degree': None},
                {'ktype': 'laplacian', 'gamma': None, 'degree': None},
                {'ktype': 'sigmoid', 'gamma': None, 'degree': None},
                {'ktype': 'cosine', 'gamma': None, 'degree': None}
                ]))
            
        #TODO add option for coef0 from scikit-learn docs
        # if 'kernel' keyword is 'precomputed' choose hyperparamaters accordingly
        elif self.input_obj.keywords['kernel'] == 'precomputed':
            if self.input_obj.keywords['precomputed_kernel']:
                precomputed_kernel = self.input_obj.keywords['precomputed_kernel']
                if 'kernel' in precomputed_kernel:
                    kernels = list(precomputed_kernel['kernel'])
                    self.set_hyperparameter('kernel', hp.choice('kernel', kernels))
                    if 'polynomial' in kernels or 'poly' in kernels:
                        print("WARNING: Polynomial type kernels are included in this hyperoptimization.")
                        print("\t It is strongly cautioned against optimizing polynomial kernels in a precomputed kernel along with other types of kernels.")
                        print("\t See KRR docs for more info.")
                        # add link to docs?
                    if 'degree' in precomputed_kernel:
                        degrees = np.asarray(precomputed_kernel['degree'])
                        if degrees[0] == 'uniform':
                            self.set_hyperparameter('degree', hp.quniform('degree', int(degrees[1]), int(degrees[2]), int(degrees[3])))
                        else:
                            degrees.astype(np.float64)
                            self.set_hyperparameter('degree', hp.choice('degree', degrees))
                    else:
                        if 'polynomial' in kernels or 'poly' in kernels:
                            self.set_hyperparameter('degree', hp.quniform('degree', 1, 5, 1))
                        else:
                            self.set_hyperparameter('degree', 1)
                else:
                    if 'degree' in precomputed_kernel:
                        degrees = np.asarray(precomputed_kernel['degree'])
                        if degrees[0] == 'uniform':
                            self.set_hyperparameter('kernel', hp.choice('kernel', [
                                {'kernel': 'polynomial', 'degree':  hp.quniform('degree', int(degrees[1]), int(degrees[2]), int(degrees[3]))},
                                {'kernel': 'rbf', 'degree': 1},
                                {'kernel': 'laplacian', 'degree': 1},
                                {'kernel': 'sigmoid', 'degree': 1},
                                {'kernel': 'cosine', 'degree': 1}
                                ]))
                        else:
                            degrees.astype(np.float64)
                            self.set_hyperparameter('kernel', hp.choice('kernel', [
                                {'kernel': 'polynomial', 'degree':  hp.choice('degree', degrees)},
                                {'kernel': 'rbf', 'degree': 1},
                                {'kernel': 'laplacian', 'degree': 1},
                                {'kernel': 'sigmoid', 'degree': 1},
                                {'kernel': 'cosine', 'degree': 1}
                                ]))
                    else:
                        self.set_hyperparameter('kernel', hp.choice('kernel', [
                                {'kernel': 'polynomial', 'degree':  hp.quniform('degree', 1, 5, 1)},
                                {'kernel': 'rbf', 'degree': 1},
                                {'kernel': 'laplacian', 'degree': 1},
                                {'kernel': 'sigmoid', 'degree': 1},
                                {'kernel': 'cosine', 'degree': 1}
                                ]))

                if 'gamma' in precomputed_kernel:
                    gammas = np.asarray(precomputed_kernel['gamma'])
                    if gammas[0] == 'uniform':
                        self.set_hyperparameter('gamma', hp.quniform('gamma', float(gammas[1]), float(gammas[2]), float(gammas[3])))
                    else:
                        gammas.astype(np.float64)
                        self.set_hyperparameter('gamma', hp.choice('gamma', gammas))
                else:
                    self.set_hyperparameter('gamma', None)

                if 'alpha' in precomputed_kernel:
                    alphas = np.asarray(precomputed_kernel['alpha'])
                    if alphas[0] == 'uniform':
                        self.set_hyperparameter('alpha', hp.quniform('alpha', float(alphas[1]), float(alphas[2]), float(alphas[3])))
                    else:
                        alphas = alphas.astype(np.float64)
                        self.set_hyperparameter('alpha', hp.choice('alpha', alphas))


    def split_train_test(self, params):
        """
        Take raw dataset and apply hyperparameters/input keywords/preprocessing 
        and train/test (tr,test) splitting.
        Assigns:
        self.X : complete input data, transformed 
        self.y : complete output data, trsnsformed
        self.Xscaler : scaling transformer for inputs
        self.yscaler : scaling transformer for outputs
        self.Xtr : training input data, transformed
        self.ytr : training output data, transformed
        self.Xtest : test input data, transformed
        self.ytext : test output data, transformed
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

    def optimize_model(self):
        print("Beginning hyperparameter optimization...")
        print("Trying {} combinations of hyperparameters".format(self.hp_maxit))
        print("Training with {} points (Full dataset contains {} points).".format(self.ntrain, self.n_datapoints))
        print("Using {} training set point sampling.".format(self.sampler))
        print("Errors are root-mean-square error in wavenumbers (cm-1)")
        self.hyperopt_trials = Trials()
        self.itercount = 1  # keep track of hyperopt iterations 
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
        print("Fine-tuning final model...")
        self.build_model(self.optimal_hyperparameters)
        print("Final model performance (cm-1):")
        self.test_error = self.vet_model(self.model)
        print("Model optimization complete. Saving final model...")
        self.save_model(self.optimal_hyperparameters)

    def build_model(self, params):
        print("Hyperparameters: ", params)
        self.split_train_test(params)
        if self.input_obj.keywords['kernel'] == 'precomputed':
            gamma = params['gamma']
            if 'kernel' not in self.input_obj.keywords['precomputed_kernel']:
                degree = int(params['kernel']['degree'])
                kernel = params['kernel']['kernel']
            else:
                degree = int(params['degree'])
                kernel = params['kernel']
        else:
            kernel = params['kernel']['ktype']
            if params['kernel']['gamma']:
                gamma = params['kernel']['gamma']
            else:
                gamma = None
            if params['kernel']['degree']:
                degree = int(params['kernel']['degree'])
            else:
                degree = 3
        alpha = params['alpha']
        self.model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree)
        self.model = self.model.fit(self.Xtr, self.ytr)

    def vet_model(self, model):
        """
        Convenience method for getting model errors of test and full datasets
        """
        pred_test, rsq = self.predict(model, self.Xtest, ytest=self.ytest)
        pred_full = self.predict(model, self.X)
        error_test = self.compute_error(self.ytest, pred_test, self.yscaler)
        error_full, median_error, max_errors = self.compute_error(self.y, pred_full, yscaler=self.yscaler, max_errors=5)
        print("R^2 {}".format(rsq))
        print("Test Dataset {}".format(round(hartree2cm * error_test,2)), end='  ')
        print("Full Dataset {}".format(round(hartree2cm * error_full,2)), end='     ')
        print("Median error: {}".format(np.round(median_error[0],2)), end='  ')
        print("Max 5 errors: {}".format(np.sort(np.round(max_errors.flatten(),1))),'\n')
        error_test_invcm = round(hartree2cm * error_test,2)
        return error_test_invcm
    
    def predict(self, model, data_in, ytest=None):
        prediciton = model.predict(data_in)
        # compute R-squared if requested
        if ytest is not None:
            rsq = model.score(data_in, ytest)
            return prediciton, rsq
        else:
            return prediciton

    def hyperopt_model(self, params):
        """
        Hyperopt-friendly wrapper for build_model
        """
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

    def preprocess(self, params, raw_X, raw_y):
        """
        Preprocess raw data according to hyperparameters
        """
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])
        if params['pip']['pip']:
            # find path to fundamental invariants from molecule type AxByCz...
            path = os.path.join(fi_dir, self.molecule_type, "output")
            raw_X, degrees = interatomics_to_fundinvar(raw_X, path)
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

    def save_model(self, params):
        print("Saving ML model data...")
        model_path = "model1_data"
        while os.path.isdir(model_path):
            new = int(re.findall("\d+", model_path)[0]) +  1
            model_path = re.sub("\d+", str(new), model_path)
        os.mkdir(model_path)
        os.chdir(model_path)
        with open('hyperparameters', 'w') as f:
            print(params, file=f)
        from joblib import dump
        dump(self.model, 'model.joblib')
        
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
        Transform a new, raw inpur according to the model's transformation procedure
        so that prediction can be made.
        """
        # ensure x dimension is n x m (n new points, m input variables)
        if len(newX.shape) == 1:
            newX = np.expand_dims(newX, 0)
        elif len(newX) > 2:
            raise Exception("Dimensions of input data is incorrect.")
        if params['morse_transform']['morse']:
            newX = morse(newX, params['morse_transform']['morse_alpha'])
        if params['pip']['pip']:
            # find path to fundamental invariants for an N atom subsystem with molecule type AxByCz...
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            newX, degrees = interatomics_to_fundinvar(newX, degrees)
        if Xscaler:
            newX = Xscaler.transform(newX)
        return newX

    def transform_new_y(self, newy, yscaler=None):
        if yscaler:
            newy = yscaler.transform(newy)
        return newy
    
    def inverse_transform_new_y(self, newy, yscaler=None):
        if yscaler:
            newy = yscaler.transform(newy)
        return newy

    def write_convenience_function(self):
        string = "from peslearn.ml import KernelRidgeReg\nfrom peslearn import InputProcessor\nfrom sklearn.kernel_ridge import KernelRidge\nimport numpy as np\nifrom joblib import load\nfrom itertools import combinationa\n\n"
        if self.pip:
            string += "krr = KernelRidgeReg('PES.dat', InputProcessor(''), molecule_type='{}')\n".format(self.molecule_type)
        else:
            string += "krr = KernelRidgeReg('PES.dat', InputProcessor(''))\n"
        with open('hyperparameters', 'r') as f:
            hyperparameters = f.read()
        string += "params = {}\n".format(hyperparameters)
        string += "X, y, Xscaler, yscaler = krr.preprocess(params, krr.raw_X, krr.raw_y)\n"
        string += "model = load(model.joblib)"
        string += krr_convenience_funciton
        return string
