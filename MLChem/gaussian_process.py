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

    def optimize_model(self):
        self.set_hyperparameter_space()
        print("{:<5d} Training Points Avg RMSE (cm-1):".format(self.ntrain))
        best = fmin(self.build_model,space=self.hyperparameter_space,algo=tpe.suggest,max_evals=self.hp_max_evals,trials=self.hyperopt_trials)
        print("\n###################################################")
        print("##                                               ##")
        print("##    Hyperparameter Optimization Complete!!!    ##")
        print("##                                               ##")
        print("###################################################\n")
        print("Best performing hyperparameters are:")
        final = space_eval(self.hyperparameter_space, best)
        print(str(sorted(final.items())))
        print("Best model performance (cm-1):")
        # temporary:
        # clear trials so that the run isnt rejected as a duplicate:
        self.hyperopt_trials = Trials()
        result = self.build_model(dict(final))
        #print("Test Dataset {}".format(round(hartree2cm * result['loss'],2)))
        #print("Full Dataset {}".format(round(hartree2cm * result['full_error'],2)))

    def build_model(self, params):
        # skip building this model if already attempted
        # (this messes with calling just one run more than once, how to fix elegantly?)
        # split into two functions? check_hyperparameters vs build model?
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
            #train, test = sample.random()
            train, test = sample.smart_random()
            #train, test = sample.energy_ordered()
            #train, test = sample.structure_based()
            Xtr = X[train]
            ytr = y[train]
            Xtest = X[test]
            ytest = y[test]
            
            # build the model
            # make GPy deterministic
            #np.random.seed(11)
            dim = X.shape[1]
            kernel = GPy.kern.RBF(dim, ARD=True) #TODO add kernels to HPOpt
            self.model = GPy.models.GPRegression(Xtr, ytr, kernel=kernel, normalizer=False)
            self.model.optimize(max_iters=600)
            # TODO handle optimizer restarts
            #self.model.optimize_restarts(5, optimizer="bfgs", verbose=True, max_iters=1000)
            self.model.optimize_restarts(5, optimizer="bfgs", verbose=False, max_iters=1000)

            # Full Dataset Prediction and Unseen Test Data Prediction
            pred_test = self.predict(Xtest)
            pred_full = self.predict(X)
            error_test = self.compute_error(Xtest, ytest, pred_test, yscaler)
            error_full = self.compute_error(X, y, pred_full, yscaler)

            # print results of this run
            #print("{:<5d} Training Points Avg RMSE (cm-1):".format(self.ntrain))
            #print("Test Dataset {:<4.2f}".format(round(hartree2cm * error_test,2)), end='  ')
            #print("Full Dataset {:<4.2f}".format(round(hartree2cm * error_full,2)))
            #print("Test Dataset {:<4.2f}".format(hartree2cm * error_test), end='  ')
            #print("Full Dataset {:<4.2f}".format(hartree2cm * error_full))
            print("Test Dataset {:<10}".format(round(hartree2cm * error_test,2)), end='  ')
            print("Full Dataset {:<10}".format(round(hartree2cm * error_full,2)))
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

    def set_hyperparameter_space(self):
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
            #raw_X, degrees = interatomics_to_fundinvar(raw_X,fi_path)
            #TODO generalize
            raw_X, degrees = interatomics_to_fundinvar(raw_X,"/home/adabbott/Git/molssi/MLChem/lib/3_atom_system/A2B/output")
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

