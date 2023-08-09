import numpy as np
from .weight_transfer import WTNN
import torch
import torch.nn as nn
from collections import OrderedDict
from ...constants import hartree2cm
import copy

class MKNNModel(nn.Module):
    def __init__(self, inp_dim, layers, activ) -> None:
        super(MKNNModel, self).__init__()
        
        depth = len(layers)
        structure_lf = OrderedDict([('input', nn.Linear(inp_dim, layers[0])),
                                 ('activ_in' , activ)])
        self.model_lf = nn.Sequential(structure_lf)
        for i in range(depth-1):
            self.model_lf.add_module('layer' + str(i), nn.Linear(layers[i], layers[i+1]))
            self.model_lf.add_module('activ' + str(i), activ)
        self.model_lf.add_module('output', nn.Linear(layers[depth-1], 1))
        
        #structure_hf = OrderedDict([('input', nn.Linear(inp_dim+1, layers[0])),
        #                         ('activ_in' , activ)]) # Add one to inp_dim for LF energy
        #self.nonlinear_hf = nn.Sequential(structure_hf) # Nonlinear NN for HF prediction
        #for i in range(depth-1):
        #    self.nonlinear_hf.add_module('layer' + str(i), nn.Linear(layers[i], layers[i+1]))
        #    self.nonlinear_hf.add_module('activ' + str(i), activ)
        #self.nonlinear_hf.add_module('output', nn.Linear(layers[depth-1], 1))
        self.nonlinear_hf = nn.Sequential(
                nn.Linear(inp_dim+1,32),
                nn.Tanh(),
                nn.Linear(32,32),
                nn.Tanh(),
                nn.Linear(32,32),
                nn.Tanh(),
                nn.Linear(32,1),
                nn.Tanh())

        self.linear_hf = nn.Linear(inp_dim+1,1) # Linear NN

    def forward(self, xh, xl):
        yl = self.model_lf(xl)
        yl_xh = self.model_lf(xh)
        #print(xh.shape)
        #print(yl_xh.shape)
        hin = torch.cat((xh,yl_xh), dim=1)
        nliny = self.nonlinear_hf(hin)
        liny = self.linear_hf(hin)
        yh = liny + nliny
        return yh, yl


class MKNN(WTNN):
    def __init__(self, dataset_path, dataset_path_lf, input_obj, input_obj_lf, molecule_type=None, molecule=None, train_path=None, test_path=None, valid_path=None):
        super().__init__(dataset_path, dataset_path_lf, input_obj, input_obj_lf, molecule_type, molecule, train_path, test_path, valid_path)
    
    def build_model(self, params, maxit=1000, val_freq=10, es_patience=2, opt='lbfgs', tol=1, decay=False, verbose=False, precision=32, return_model=False):
        print("Hyperparameters: ", params)
        self.split_train_test(params, validation_size=self.nvalid, validation_size_lf=self.nvalid_lf, precision=precision)  # split data, according to scaling hp's
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
        
        model = MKNNModel(inp_dim, l, activ)
        
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
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr*0.01)
        # Define update variables for early stopping, decay, gradient explosion handling
        prev_loss = 1.0
        es_tracker = 0
        best_val_error = None
        failures = 0
        decay_attempts = 0
        prev_best = None
        decay_start = False
        maxit += 5000
        labda = 1e-6 #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        for epoch in range(1,maxit):
            def closure():
                optimizer.zero_grad()
                y_pred_hf, y_pred_lf = model(self.Xtr, self.Xtr_lf)
                loss = torch.sqrt(metric(y_pred_lf, self.ytr_lf)) + torch.sqrt(metric(y_pred_hf, self.ytr)) + labda*sum(p.pow(2.0).sum() for p in model.parameters()) # L2 regularization
                loss.backward()
                return loss
            optimizer.step(closure)
            # validate
            if epoch % val_freq == 0:
                with torch.no_grad():
                    tmp_pred, trash = model(self.Xvalid, self.Xvalid)
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
            train_pred, trash = model(self.Xtr, self.Xtr)
            train_loss = metric(train_pred, self.ytr)
            train_error_rmse = np.sqrt(train_loss.item() * loss_descaler) * hartree2cm
            test_pred, trash = model(self.Xtest, self.Xtest)
            test_loss = metric(test_pred, self.ytest)
            test_error_rmse = np.sqrt(test_loss.item() * loss_descaler) * hartree2cm 
            val_pred, trash = model(self.Xvalid, self.Xvalid) 
            val_loss = metric(val_pred, self.yvalid)
            val_error_rmse = np.sqrt(val_loss.item() * loss_descaler) * hartree2cm
            full_pred, trash = model(self.X, self.X)
            full_loss = metric(full_pred, self.y)
            full_error_rmse = np.sqrt(full_loss.item() * loss_descaler) * hartree2cm
        print("Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f} Train set RMSE: {:5.2f} Full dataset RMSE (cm-1): {:5.2f}".format(test_error_rmse, val_error_rmse, train_error_rmse, full_error_rmse))
        if return_model:
            return model, test_error_rmse, val_error_rmse, full_error_rmse 
        else:
            return test_error_rmse, val_error_rmse
