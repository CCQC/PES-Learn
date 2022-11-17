import torch
import torch.nn as nn
import numpy as np

from ..mfmodel import MFModel
from ...constants import hartree2cm

class LF(nn.Module):
    def __init__(self, inp_dim, activ):
        super(LF, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(inp_dim,32), 
                activ, 
                nn.Linear(32,32), 
                activ, 
                nn.Linear(32,32), 
                activ, 
                nn.Linear(32,1), 
                activ)
    def forward(self, x):
        y = self.net(x)
        return y

class HF(nn.Module):
    def __init__(self, inp_dim, activ):
        super(HF, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(inp_dim,20), 
                activ, 
                nn.Linear(20,20), 
                activ, 
                nn.Linear(20,20), 
                activ, 
                nn.Linear(20,1), 
                activ)
    def forward(self, x):
        y = self.net(x)
        return y

class Boring(MFModel):
    def __init__(self, dataset_paths, input_objs, molecule_type=None, molecule=None, train_paths=(None, None), test_paths=(None, None), valid_paths=(None, None)):
        super(Boring, self).__init__(dataset_paths, input_objs, molecule_type, molecule, train_paths, test_paths, valid_paths)

    def build_model(self, params, maxit=1000, val_freq=10, es_patience=2, opt='lbfgs', tol=1.0, decay=False, verbose=False, precision=32, return_model=False):

        print("Hyperparameters: ", params)
        self.split_train_test(params, precision=precision)  # split data, according to scaling hp's
        scale = params['scale_y']                                                        # Find descaling factor to convert loss to original energy units
        if scale == 'std':
            loss_descaler_h = self.yscaler_h.var_[0]
            loss_descaler_l = self.yscaler_l.var_[0]
        if scale.startswith('mm'):
            loss_descaler_h = (1/self.yscaler_h.scale_[0]**2)
            loss_descaler_l = (1/self.yscaler_l.scale_[0]**2)

        activation = params['scale_X']['activation']
        if activation == 'tanh':
            activ = nn.Tanh() 
        if activation == 'sigmoid':
            activ = nn.Sigmoid()
        
        inp_dim = self.m_low.inp_dim
        torch.manual_seed(0)
        self.model_low = LF(inp_dim, activ)
        self.model_high = HF(inp_dim, activ)
        if precision == 64: # cast model to proper precision
            self.model = self.model.double() 

        self.metric = torch.nn.MSELoss()
        # Define optimizer
        if 'lr' in params:
            lr = params['lr']
        elif opt == 'lbfgs':
            lr = 0.5
        else:
            lr = 0.1
        opt_l = self.get_optimizer(opt, self.model_low.parameters(), lr=lr)
        
        self.val_freq = val_freq
        self.verbose = verbose
        prev_loss = 1.0
        es_tracker = 0
        best_val_error = None
        failures = 0
        decay_attempts = 0
        prev_best = None
        decay_start = False
    
        for epoch in range(1,maxit):
            def closure():
                opt_l.zero_grad()
                y_pred = self.model_low(self.Xtr_l)
                loss = torch.sqrt(self.metric(y_pred, self.ytr_l))
                loss.backward()
                return loss
            opt_l.step(closure)
            self.validation_step(epoch, self.model_low, self.Xvalid_l, self.yvalid_l, loss_descaler_l)
        test_err, val_err, full_err = self.vet_model(self.model_low, self.X_l, self.y_l, self.Xtest_l, self.ytest_l, self.Xvalid_l,
                                                        self.yvalid_l, loss_descaler_l)
        if return_model:
            return self.model_low, test_err, val_err, full_err
        else:
            return test_err, val_err

    def validation_step(self, epoch, model, Xvalid, yvalid, loss_descaler):
        best_val_error = None
        if epoch % self.val_freq == 0:
            with torch.no_grad():
                tmp_pred = model(Xvalid) 
                tmp_loss = self.metric(tmp_pred, yvalid)
                val_error_rmse = np.sqrt(tmp_loss.item() * loss_descaler) * hartree2cm # loss_descaler converts MSE in scaled data domain to MSE in unscaled data domain
                if best_val_error:
                    if val_error_rmse < best_val_error:
                        prev_best = best_val_error * 1.0
                        best_val_error = val_error_rmse * 1.0 
                else:
                    record = True
                    best_val_error = val_error_rmse * 1.0 
                    prev_best = best_val_error
                if self.verbose:
                    print("Epoch {} Validation RMSE (cm-1): {:5.3f}".format(epoch, val_error_rmse))

    def vet_model(self, model, X, y, Xtest, ytest, Xvalid, yvalid, loss_descaler):
        with torch.no_grad():
            test_pred = model(Xtest)
            test_loss = self.metric(test_pred, ytest)
            test_error_rmse = self.rmse_fxn(test_loss.item(), loss_descaler)
            val_pred = model(Xvalid) 
            val_loss = self.metric(val_pred, yvalid)
            val_error_rmse = self.rmse_fxn(val_loss.item(), loss_descaler)
            full_pred = model(X)
            full_loss = self.metric(full_pred, y)
            full_error_rmse = self.rmse_fxn(full_loss.item(), loss_descaler)
        
        print("Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f} Full dataset RMSE (cm-1): {:5.2f}".format(test_error_rmse, val_error_rmse, full_error_rmse))
        return test_error_rmse, val_error_rmse, full_error_rmse 

    def rmse_fxn(self, x, loss_descaler):
        return np.sqrt(x * loss_descaler) * hartree2cm

