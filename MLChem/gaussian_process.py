import numpy as np
import sklearn.metrics
from .model import Model
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
import GPy
from .data_sampler import DataSampler 
from .constants import hartree2cm
from .preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler

class GaussianProcess(Model):
    """
    Constructs a Gaussian Process Model using GPy
    """
    def __init__(self, dataset_path, ntrain, input_obj):
        self.hyperopt_trials = Trials()
        super().__init__(dataset_path, ntrain, input_obj)


    def build_model(self, params):
        # skip building this model if already attempted
        is_repeat = None
        for i in self.hyperopt_trials.results:
            if 'memo' in i:
                if params == i['memo']:
                    is_repeat = True
        if is_repeat:
            return {'loss': 0.0, 'status': STATUS_FAIL, 'memo': 'repeat'}
        else:
            raw_X = self.dataset.values[:, :-1]
            raw_y = self.dataset.values[:,-1].reshape(-1,1)
            X, y, Xscaler, yscaler = self.preprocess(params, raw_X, raw_y)
            sample = DataSampler(self.dataset, self.ntrain)
            # TODO if keyword, do sample procedure A,B,C...
            train, test = sample.random()
            Xtr = X[train]
            ytr = y[train]
            Xtest = X[test]
            ytest = y[test]
            
            # build the model
            dim = X.shape[1]
            kernel = GPy.kern.RBF(dim, ARD=True) #TODO add kernels to HPOpt
            self.model = GPy.models.GPRegression(Xtr, ytr, kernel=kernel, normalizer=False)
            self.model.optimize(max_iters=600)
            # TODO handle optimizer restarts
            self.model.optimize_restarts(2, optimizer="bfgs", verbose=False, max_iters=1000)

            # Full Dataset Prediction and Unseen Test Data Prediction
            #pred_full, v1 = model.predict(X, full_cov=False)
            #pred_test, v2 = model.predict(Xtest, full_cov=False)
            pred_test = self.predict(Xtest)
            pred_full = self.predict(X)
            error_test = self.compute_error(Xtest, ytest, pred_test, yscaler)
            error_full = self.compute_error(X, y, pred_full, yscaler)

            # print results of this run
            print("{:<5d} Training Points Avg RMSE (cm-1):".format(self.ntrain))
            print("Test Dataset {}".format(round(hartree2cm * error_test,2)), end='  ')
            print("Full Dataset {}".format(round(hartree2cm * error_full,2)))
            result = {'loss': error_test, 'status': STATUS_OK, 'memo': params}
            return result

    def predict(self, X):
        prediction, v1 = self.model.predict(X, full_cov=False)
        return prediction 
     
    def compute_error(self, X, y, prediction, yscaler):
        """
        Predict the root-mean-square error of model given 
        known X,y, a prediction, and a y scaling object, if it exists.
        """
        if yscaler:
            raw_y = yscaler.inverse_transform(y)
            unscaled_prediction = yscaler.inverse_transform(prediction)
            error = np.sqrt(sklearn.metrics.mean_squared_error(raw_y,  unscaled_prediction))
        else:
            error = np.sqrt(sklearn.metrics.mean_squared_error(y, prediction))
        return error

    def hyperparameter_space(self):
        self.hyperparameter_space = {
                  'fi_transform': hp.choice('fi_transform',
                                [
                                {'fi': True,
                                    'degree_reduction': hp.choice('degree_reduction', [True, False])},
                                {'fi': False}
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
        Preprocess data according to hyperparameters
        """
        # Transform to morse variables (exp(-r/alpha))
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])
        # Transform to FIs, degree reduce if called
        if params['fi_transform']['fi']:
            raw_X, degrees = interatomics_to_fundinvar(raw_X,fi_path)
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

