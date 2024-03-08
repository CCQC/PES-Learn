from ..neural_network import NeuralNetwork
from ..model import Model
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from ...constants import hartree2cm
import copy
import numpy as np

class WTNN(NeuralNetwork):
    def __init__(self, dataset_path, dataset_path_lf, input_obj, input_obj_lf, molecule_type=None, molecule=None, train_path=None, test_path=None, valid_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        self.lf_model = Model(dataset_path_lf, input_obj_lf, molecule_type, molecule, train_path, test_path, valid_path) # TODO: Paths are for HF model
        if self.lf_model.input_obj.keywords['validation_points']:
            self.nvalid_lf = self.lf_model.input_obj.keywords['validation_points']
            if (self.nvalid_lf + self.lf_model.ntrain + 1) > self.lf_model.n_datapoints:
                raise Exception("Error: User-specified training set size and validation set size exceeds the size of the dataset.")
        else:
            self.nvalid_lf = round((self.lf_model.n_datapoints - self.lf_model.ntrain)  / 2)
 
    def split_train_test(self, params, validation_size=None, validation_size_lf=None, precision=32):
        self.X, self.y, self.Xscaler, self.yscaler = self.preprocess(params, self.raw_X, self.raw_y)
        self.X_lf, self.y_lf, self.Xscaler_lf, self.yscaler_lf = self.preprocess(params, self.lf_model.raw_X, self.lf_model.raw_y)
        if self.sampler == 'user_supplied':
            self.Xtr = self.transform_new_X(self.raw_Xtr, params, self.Xscaler)
            self.ytr = self.transform_new_y(self.raw_ytr, self.yscaler)
            self.Xtest = self.transform_new_X(self.raw_Xtest, params, self.Xscaler)
            self.ytest = self.transform_new_y(self.raw_ytest, self.yscaler)
            
            self.Xtr_lf = self.transform_new_X(self.lf_model.raw_Xtr, params, self.Xscaler_lf)
            self.ytr_lf = self.transform_new_y(self.lf_model.raw_ytr, self.yscaler_lf)
            self.Xtest_lf = self.transform_new_X(self.lf_model.raw_Xtest, params, self.Xscaler_lf)
            self.ytest_lf = self.transform_new_y(self.lf_model.raw_ytest, self.yscaler_lf)
            if self.valid_path:
                self.Xvalid = self.transform_new_X(self.raw_Xvalid, params, self.Xscaler)
                self.yvalid = self.transform_new_y(self.raw_yvalid, self.yscaler)
                
                self.Xvalid_lf = self.transform_new_X(self.lf_model.raw_Xvalid, params, self.Xscaler_lf)
                self.yvalid_lf = self.transform_new_y(self.lf_model.raw_yvalid, self.yscaler_lf)
            else:
                raise Exception("Please provide a validation set for Neural Network training.")
        else:
            self.Xtr = self.X[self.train_indices]
            self.ytr = self.y[self.train_indices]
            
            self.Xtr_lf = self.X_lf[self.lf_model.train_indices]
            self.ytr_lf = self.y_lf[self.lf_model.train_indices]
            #TODO: this is splitting validation data in the same way at every model build, not necessary.
            self.valid_indices, self.new_test_indices = train_test_split(self.test_indices, train_size = validation_size, random_state=42)
            self.valid_indices_lf, self.new_test_indices_lf = train_test_split(self.lf_model.test_indices, train_size = validation_size_lf, random_state=42)
            if validation_size and validation_size_lf:
                self.Xvalid = self.X[self.valid_indices]             
                self.yvalid = self.y[self.valid_indices]
                self.Xtest = self.X[self.new_test_indices]
                self.ytest = self.y[self.new_test_indices]
                
                self.Xvalid_lf = self.X_lf[self.valid_indices_lf]             
                self.yvalid_lf = self.y_lf[self.valid_indices_lf]
                self.Xtest_lf = self.X_lf[self.new_test_indices_lf]
                self.ytest_lf = self.y_lf[self.new_test_indices_lf]

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
            
            self.Xtr_lf    = torch.tensor(self.Xtr_lf,   dtype=torch.float32)
            self.ytr_lf    = torch.tensor(self.ytr_lf,   dtype=torch.float32)
            self.Xtest_lf  = torch.tensor(self.Xtest_lf, dtype=torch.float32)
            self.ytest_lf  = torch.tensor(self.ytest_lf, dtype=torch.float32)
            self.Xvalid_lf = torch.tensor(self.Xvalid_lf,dtype=torch.float32)
            self.yvalid_lf = torch.tensor(self.yvalid_lf,dtype=torch.float32)
            self.X_lf = torch.tensor(self.X_lf,dtype=torch.float32)
            self.y_lf = torch.tensor(self.y_lf,dtype=torch.float32)
        elif precision == 64:
            self.Xtr    = torch.tensor(self.Xtr,   dtype=torch.float64)
            self.ytr    = torch.tensor(self.ytr,   dtype=torch.float64)
            self.Xtest  = torch.tensor(self.Xtest, dtype=torch.float64)
            self.ytest  = torch.tensor(self.ytest, dtype=torch.float64)
            self.Xvalid = torch.tensor(self.Xvalid,dtype=torch.float64)
            self.yvalid = torch.tensor(self.yvalid,dtype=torch.float64)
            self.X = torch.tensor(self.X,dtype=torch.float64)
            self.y = torch.tensor(self.y,dtype=torch.float64)
            
            self.Xtr_lf    = torch.tensor(self.Xtr_lf,   dtype=torch.float64)
            self.ytr_lf    = torch.tensor(self.ytr_lf,   dtype=torch.float64)
            self.Xtest_lf  = torch.tensor(self.Xtest_lf, dtype=torch.float64)
            self.ytest_lf  = torch.tensor(self.ytest_lf, dtype=torch.float64)
            self.Xvalid_lf = torch.tensor(self.Xvalid_lf,dtype=torch.float64)
            self.yvalid_lf = torch.tensor(self.yvalid_lf,dtype=torch.float64)
            self.X_lf = torch.tensor(self.X_lf,dtype=torch.float64)
            self.y_lf = torch.tensor(self.y_lf,dtype=torch.float64)
        else:
            raise Exception("Invalid option for 'precision'")
    
    def build_model(self, params, maxit=1000, val_freq=10, es_patience=2, opt='lbfgs', tol=1, decay=False, verbose=False, precision=32, return_model=False):
        # LF Training
        print("Hyperparameters: ", params)
        self.split_train_test(params, validation_size=self.nvalid, validation_size_lf=self.nvalid_lf, precision=precision)  # split data, according to scaling hp's
        scale = params['scale_y']                    # Find descaling factor to convert loss to original energy units

        if scale == 'std':
            loss_descaler = self.yscaler_lf.var_[0] # Here
        if scale.startswith('mm'):
            loss_descaler = (1/self.yscaler_lf.scale_[0]**2) # Here

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
        
        model = nn.Sequential(structure) # Here
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
                y_pred = model(self.Xtr_lf)
                loss = torch.sqrt(metric(y_pred, self.ytr_lf)) # passing RMSE instead of MSE improves precision IMMENSELY
                loss.backward()
                return loss
            optimizer.step(closure)
            # validate
            if epoch % val_freq == 0:
                with torch.no_grad():
                    tmp_pred = model(self.Xvalid_lf) 
                    tmp_loss = metric(tmp_pred, self.yvalid_lf)
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
            test_pred = model(self.Xtest_lf)
            test_loss = metric(test_pred, self.ytest_lf)
            test_error_rmse = np.sqrt(test_loss.item() * loss_descaler) * hartree2cm 
            val_pred = model(self.Xvalid_lf) 
            val_loss = metric(val_pred, self.yvalid_lf)
            val_error_rmse = np.sqrt(val_loss.item() * loss_descaler) * hartree2cm
            full_pred = model(self.X_lf)
            full_loss = metric(full_pred, self.y_lf)
            full_error_rmse = np.sqrt(full_loss.item() * loss_descaler) * hartree2cm
        print("LF: Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f} Full dataset RMSE (cm-1): {:5.2f}".format(test_error_rmse, val_error_rmse, full_error_rmse))
        
        # HF Training
        
        if scale == 'std':
            loss_descaler = self.yscaler.var_[0]
        if scale.startswith('mm'):
            loss_descaler = (1/self.yscaler.scale_[0]**2)
        
        # Define update variables for early stopping, decay, gradient explosion handling
        prev_loss = 1.0
        es_tracker = 0
        best_val_error = None
        failures = 0
        decay_attempts = 0
        prev_best = None
        decay_start = False
        saved_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        saved_optimizer_state_dict['param_groups'][0]['lr'] = lr * 0.1
        optimizer.load_state_dict(saved_optimizer_state_dict)
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
        print("HF: Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f} Full dataset RMSE (cm-1): {:5.2f}".format(test_error_rmse, val_error_rmse, full_error_rmse))
        
        if return_model:
            return model, test_error_rmse, val_error_rmse, full_error_rmse 
        else:
            return test_error_rmse, val_error_rmse

