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

class NeuralNetwork(Model):
    """
    Constructs a Neural Network Model using PyTorch
    """
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None, valid_path=None):
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
                self.inp_dim = len(open(path).readlines())
            if molecule:
                path = os.path.join(package_directory, "lib", molecule.molecule_type, "output")
                self.inp_dim = len(open(path).readlines())
        else:
            self.inp_dim = self.raw_X.shape[1]

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
        if not self.input_obj.keywords['validation_points']:
            print("Number of validation points not specified. Splitting test set in half --> 50% test, 50% validation")
        print("Training with {} points. Validating with {} points. Full dataset contains {} points.".format(self.ntrain, self.nvalid, self.n_datapoints))
        print("Using {} training set point sampling.".format(self.sampler))
        print("Errors are root-mean-square error in wavenumbers (cm-1)")
        print("\nPerforming neural architecture search...\n")
        best_hlayers = self.neural_architecture_search(trial_layers = self.trial_layers)
        print("\nNeural architecture search complete. Best hidden layer structures: {}\n".format(best_hlayers))
        print("Beginning hyperparameter optimization...")
        print("Trying {} combinations of hyperparameters".format(self.hp_maxit))
        self.set_hyperparameter('layers', hp.choice('layers', best_hlayers))
        self.hyperopt_trials = Trials()
        self.itercount = 1
        if self.input_obj.keywords['rseed']:
            rstate = np.random.default_rng(self.input_obj.keywords['rseed'])
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
        print("Optimizing learning rate...")
        
        if self.input_obj.keywords['nn_precision'] == 64:
            precision = 64
        elif self.input_obj.keywords['nn_precision'] == 32:
            precision = 32
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
        self.test_error = test_error
        print("Model optimization complete. Saving final model...")
        self.save_model(self.optimal_hyperparameters, model, performance)

    def neural_architecture_search(self, trial_layers=None):
        """
        Finds 'optimal' hidden layer structure. (i.e., tries both wide and deep homogenous hidden layer structures and finds the best 3 for follow-up hyperparameter optimization)
        
        Parameters
        ----------
        trial_layers : list
            A list of tuples describing the number of nodes in each hidden layer. Example: a 3-20-20-1 NN would be a tuple (20,20).
        """
        if trial_layers == None:
            tmp_layers = [(16,), (16,16), (16,16,16), (16,16,16,16),
                          (32,), (32,32), (32,32,32), (32,32,32,32),
                          (64,), (64,64), (64,64,64), (64,64,64,64),
                          (128,), (128,128), (128,128,128),
                          (256,), (256,256)] 
        else:
            tmp_layers = trial_layers
        self.nas_layers = sort_architectures(tmp_layers, self.inp_dim)
        self.nas_size = len(self.nas_layers)
        # force reliable set of hyperparameters
        params = {'morse_transform': {'morse':False},'scale_X':{'scale_X':'std', 'activation':'tanh'}, 'scale_y':'std'}
        if self.pip:
            params['pip'] = {'degree_reduction': False, 'pip': True} 
        else:
            params['pip'] = {'degree_reduction': False, 'pip': False} 
        test = []
        validation = []
        for i in self.nas_layers:
            params['layers'] = i
            print("Hidden layer structure: ", i)
            testerror, valid = self.build_model(params, maxit=300, val_freq=10, es_patience=2, opt='lbfgs', tol=1.0,  decay=False, verbose=False)
            test.append(testerror)
            validation.append(valid)
        # save best architectures for hyperparameter optimization
        indices = np.argsort(test)
        best_hlayers = [self.nas_layers[i] for i in indices[:3]]
        return best_hlayers

    def split_train_test(self, params, validation_size=None, precision=32):
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

    def get_optimizer(self, opt_type, mdata, lr=0.1): 
        rate = lr
        if opt_type == 'lbfgs':
            #optimizer = torch.optim.LBFGS(mdata, lr=rate, max_iter=20, max_eval=None, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100) # Defaults
            optimizer = torch.optim.LBFGS(mdata, lr=rate, max_iter=20, max_eval=None, tolerance_grad=1e-8, tolerance_change=1e-12, history_size=100)
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(mdata, lr=rate)
        return optimizer

    def build_model(self, params, maxit=1000, val_freq=10, es_patience=2, opt='lbfgs', tol=1.0,  decay=False, verbose=False, precision=32, return_model=False):
        """
        Parameters
        ----------
        params : dict
            Hyperparameter dictionary
        maxit : int
            Maximum number of epochs
        val_freq : int
            Validation frequency: Compute error on validation set every 'val_freq' epochs 
        es_patience : int
            Early stopping patience. How many validations to do before giving up training this model according to tolerance 'tol'
        tol : float
            Tolerance for early stopping in wavenumbers cm^-1: if validation set error 
            does not improve by this quantity after waiting for 'es_patience' validation cycles, halt training
        decay : bool
            If True, reduce the learning rate if validation error plateaus
        verbose : bool
            If true, print training progress after every validation  
        """
        print("Hyperparameters: ", params)
        self.split_train_test(params, validation_size=self.nvalid, precision=precision)  # split data, according to scaling hp's
        scale = params['scale_y']                                                        # Find descaling factor to convert loss to original energy units
        if scale == 'std':
            loss_descaler = self.yscaler.var_[0]
        if scale.startswith('mm'):
            loss_descaler = (1/self.yscaler.scale_[0]**2)

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
        if precision == 64: # cast model to proper precision
            model = model.double() 

        metric = torch.nn.MSELoss()
        # Define optimizer
        if 'lr' in params:
            lr = params['lr']
        elif opt == 'lbfgs':
            lr = 0.5
        else:
            lr = 0.1
        optimizer = self.get_optimizer(opt, model.parameters(), lr=lr)
        # Define update variables for early stopping, decay, gradient explosion handling
        prev_loss = 1.0
        es_tracker = 0
        best_val_error = None
        failures = 0
        decay_attempts = 0
        prev_best = None
        decay_start = False
    
        for epoch in range(1,maxit):
            def closure():
                optimizer.zero_grad()
                y_pred = model(self.Xtr)
                loss = torch.sqrt(metric(y_pred, self.ytr)) # passing RMSE instead of MSE improves precision IMMENSELY
                loss.backward()
                return loss
            optimizer.step(closure)
            # validate
            if epoch % val_freq == 0:
                with torch.no_grad():
                    tmp_pred = model(self.Xvalid) 
                    tmp_loss = metric(tmp_pred, self.yvalid)
                    val_error_rmse = np.sqrt(tmp_loss.item() * loss_descaler) * hartree2cm # loss_descaler converts MSE in scaled data domain to MSE in unscaled data domain
                    if best_val_error:
                        if val_error_rmse < best_val_error:
                            prev_best = best_val_error * 1.0
                            best_val_error = val_error_rmse * 1.0 
                    else:
                        record = True
                        best_val_error = val_error_rmse * 1.0 
                        prev_best = best_val_error
                    if verbose:
                        print("Epoch {} Validation RMSE (cm-1): {:5.3f}".format(epoch, val_error_rmse))
                    if decay_start:
                        scheduler.step(val_error_rmse)

                    # Early Stopping 
                    if epoch > 5:
                        # if current validation error is not the best (current - best > 0) and is within tol of previous error, the model is stagnant. 
                        if ((val_error_rmse - prev_loss) < tol) and (val_error_rmse - best_val_error) > 0.0: 
                            es_tracker += 1
                        # else if: current validation error is not the best (current - best > 0) and is greater than the best by tol, the model is overfitting. Bad epoch.
                        elif ((val_error_rmse - best_val_error) > tol) and (val_error_rmse - best_val_error) > 0.0: 
                            es_tracker += 1
                        # else if: if the current validation error is a new record, but not significant, the model is stagnant
                        elif (prev_best - best_val_error) < 0.001:
                            es_tracker += 1
                        # else: model set a new record validation error. Reset early stopping tracker
                        else:
                            es_tracker = 0
                        #TODO this framework does not detect oscillatory behavior about 'tol', though this has not been observed to occur in any case 
                        # Check status of early stopping tracker. First try decaying to see if stagnation can be resolved, if not then terminate training
                        if es_tracker > es_patience:
                            if decay:  # if decay is set to true, if early stopping criteria is triggered, begin LR scheduler and go back to previous model state and attempt LR decay.
                                if decay_attempts < 1:
                                    decay_attempts += 1
                                    es_tracker = 0
                                    if verbose:
                                        print("Performance plateau detected. Reverting model state and decaying learning rate.")
                                    decay_start = True
                                    thresh = (0.1 / np.sqrt(loss_descaler)) / hartree2cm  # threshold is 0.1 wavenumbers
                                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, threshold=thresh, threshold_mode='abs', min_lr=0.05, cooldown=2, patience=10, verbose=verbose)
                                    model.load_state_dict(saved_model_state_dict)
                                    saved_optimizer_state_dict['param_groups'][0]['lr'] = lr*0.9
                                    optimizer.load_state_dict(saved_optimizer_state_dict)
                                    # Since learning rate is decayed, override tolerance, patience, validation frequency for high-precision
                                    #tol = 0.05
                                    #es_patience = 100
                                    #val_freq = 1
                                    continue
                                else:
                                    prev_loss = val_error_rmse * 1.0
                                    if verbose:
                                        print('Early stopping termination')
                                    break
                            else:
                                prev_loss = val_error_rmse * 1.0
                                if verbose:
                                    print('Early stopping termination')
                                break

                    # Handle exploding gradients 
                    if epoch > 10:
                        if (val_error_rmse > prev_loss*10): # detect large increases in loss
                            if epoch > 60: # distinguish between exploding gradients at near converged models and early on exploding grads
                                if verbose:
                                    print("Exploding gradient detected. Resuming previous model state and decaying learning rate")
                                model.load_state_dict(saved_model_state_dict)
                                saved_optimizer_state_dict['param_groups'][0]['lr'] = lr*0.5
                                optimizer.load_state_dict(saved_optimizer_state_dict)
                                failures += 1   # if 
                                if failures > 2: 
                                    break
                                else:
                                    continue
                            else:
                                break
                        if val_error_rmse != val_error_rmse: # detect NaN 
                            break
                        if ((prev_loss < 1.0) and (precision == 32)):  # if 32 bit precision and model is giving very high accuracy, kill so the accuracy does not go beyond 32 bit precision
                            break
                    prev_loss = val_error_rmse * 1.0  # save previous loss to track improvement

            # Periodically save model state so we can reset under instability/overfitting/performance plateau
            if epoch % 50 == 0:
                saved_model_state_dict = copy.deepcopy(model.state_dict())
                saved_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            
        with torch.no_grad():
            test_pred = model(self.Xtest)
            test_loss = metric(test_pred, self.ytest)
            test_error_rmse = np.sqrt(test_loss.item() * loss_descaler) * hartree2cm 
            val_pred = model(self.Xvalid) 
            val_loss = metric(val_pred, self.yvalid)
            val_error_rmse = np.sqrt(val_loss.item() * loss_descaler) * hartree2cm
            full_pred = model(self.X)
            full_loss = metric(full_pred, self.y)
            full_error_rmse = np.sqrt(full_loss.item() * loss_descaler) * hartree2cm
        print("Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f} Full dataset RMSE (cm-1): {:5.2f}".format(test_error_rmse, val_error_rmse, full_error_rmse))
        if return_model:
            return model, test_error_rmse, val_error_rmse, full_error_rmse 
        else:
            return test_error_rmse, val_error_rmse

    def hyperopt_model(self, params):
        """
        A Hyperopt-friendly wrapper for build_model
        """
        # skip building this model if hyperparameter combination already attempted
        for i in self.hyperopt_trials.results:
            if 'memo' in i:
                if params == i['memo']:
                    return {'loss': i['loss'], 'status': STATUS_OK, 'memo': 'repeat'}
        if self.itercount > self.hp_maxit:
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
        
        if self.sampler == 'user_supplied':
            self.traindata.to_csv('train_set',sep=',',index=False,float_format='%12.12f')
            self.validdata.to_csv('validation_set',sep=',',index=False,float_format='%12.12f')
            self.testdata.to_csv('test_set', sep=',', index=False, float_format='%12.12f')
        else:
            self.dataset.iloc[self.train_indices].to_csv('train_set',sep=',',index=False,float_format='%12.12f')
            self.dataset.iloc[self.valid_indices].to_csv('validation_set', sep=',', index=False, float_format='%12.12f')
            self.dataset.iloc[self.new_test_indices].to_csv('test_set', sep=',', index=False, float_format='%12.12f')
    
        self.dataset.to_csv('PES.dat', sep=',',index=False,float_format='%12.12f')
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
        if self.pip:
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







            
        




