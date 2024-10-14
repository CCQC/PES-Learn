from ..neural_network import NeuralNetwork
from ..model import Model
from .diff_model import DiffModel
import torch
import torch.nn as nn
torch.set_num_threads(8)
print(torch.get_num_threads())
from torch import autograd
import numpy as np
from collections import OrderedDict
import copy
from copy import deepcopy
from sklearn.model_selection import train_test_split
from .utils import Pip_B
from .utils.transform_deriv import degree_B1, degree_B2, morse, morse_B1, morse_B2
from .utils.cart_dist import cart_dist_B_2
import os
from ...constants import package_directory, hartree2cm
import re
from ..preprocessing_helper import morse, interatomics_to_fundinvar, degree_reduce, general_scaler

class DiffNeuralNetwork(NeuralNetwork):
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, der_lvl = 0,
                 train_path=None, test_path=None, valid_path=None, 
                 grad_input_obj=None, hess_input_obj=None, grad_data_path=None, hess_data_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        # Assume raw_X is Cartesian, if not we got problems. Same with grad and Hess data, assume Cartesian basis 
        # Calc. X in interatomic distance basis
        nletters = re.findall(r"[A-Z]", self.molecule_type)
        nnumbers = re.findall(r"\d", self.molecule_type)
        nnumbers2 = [int(i) for i in nnumbers]
        self.natoms = len(nletters) + sum(nnumbers2) - len(nnumbers2)
        self.n_interatomics = int(0.5 * (self.natoms * self.natoms - self.natoms))
        
        if self.pip:
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            self.pip_B = Pip_B(path, self.n_interatomics)
        #self.raw_X_mod = self.raw_X.reshape((self.raw_X.shape[0], self.natoms, 3))
        #self.raw_Xr = np.zeros((self.raw_X.shape[0], self.n_interatomics))

        self.train_grad = der_lvl == 1 or der_lvl == 2
        self.train_hess = der_lvl == 2
        self.grad = False
        self.hess = False

        self.fullraw_X = deepcopy(self.raw_X)
        self.fullraw_y = deepcopy(self.raw_y)
        self.raw_Xr = self.cart_to_interatomic(self.raw_X)
        #self.fullraw_X.reshape((-1,1))
        
        #for atom in range(1, self.natoms):
        #    # Create an array of duplicated cartesian coordinates of this particular atom, for every geometry, which is the same shape as 'cartesians'
        #    tmp1 = np.broadcast_to(self.raw_X_mod[:,atom,:], (self.raw_X.shape[0], 3))
        #    tmp2 = np.tile(tmp1, (self.natoms,1,1)).transpose(1,0,2)
        #    # Take the non-redundant norms of this atom to all atoms after it in cartesian array
        #    diff = tmp2[:, 0:atom,:] - self.raw_X_mod[:, 0:atom,:]
        #    norms = np.sqrt(np.einsum('...ij,...ij->...i', diff , diff))
        #    # Fill in the norms into interatomic distances 2d array , n_interatomic_distances)
        #    if atom == 1:
        #        idx1, idx2 = 0, 1
        #    if atom > 1:
        #        x = int((atom**2 - atom) / 2)
        #        idx1, idx2 = x, x + atom
        #    self.raw_Xr[:, idx1:idx2] = norms 
        if grad_input_obj is not None:
            self.grad = True
            self.grad_model = DiffModel(grad_data_path, grad_input_obj, molecule_type, molecule, der_lvl=1)
            self.grad_offset = self.fullraw_X.shape[0]
            #self.fullraw_X = np.vstack((self.fullraw_X, self.grad_model.raw_X))
            #self.fullraw_y = np.vstack((self.fullraw_y, self.grad_model.raw_y))
            #self.raw_Xr_grad = self.cart_to_interatomic(self.grad_model.raw_X)
            #self.grad_model.raw_grad = self.grad_model.raw_grad[:,-1].reshape((-1,1))
            #print(self.grad_model.raw_grad)
            #self.grad_model.raw_X = self.grad_model.raw_X[:,0:2]#.reshape((-1,1))
            self.fullraw_X = np.vstack((self.fullraw_X, self.grad_model.raw_X))
            self.fullraw_y = np.vstack((self.fullraw_y, self.grad_model.raw_y))
            
            if self.grad_model.input_obj.keywords["validation_points"]:
                self.nvalid_grad = self.grad_model.input_obj.keywords["validation_points"]
                if (self.nvalid_grad + self.grad_model.ntrain + 1) > self.grad_model.n_datapoints:
                    raise Exception("Error: User-specified training set size and validation set size exceeds the size of the dataset.")
            else:
                self.nvalid_grad = round((self.grad_model.n_datapoints - self.grad_model.ntrain)  / 2)
        if hess_input_obj is not None:
            self.hess = True
            self.hess_model = DiffModel(hess_data_path, hess_input_obj, molecule_type, molecule, der_lvl=2)
            self.raw_Xr_hess = self.cart_to_interatomic(self.hess_model.raw_X)
            if self.hess_model.input_obj.keywords["validation_points"]:
                self.nvalid_hess = self.hess_model.input_obj.keywords["validation_points"]
                if (self.nvalid_hess + self.hess_model.ntrain + 1) > self.hess_model.n_datapoints:
                    raise Exception("Error: User-specified training set size and validation set size exceeds the size of the dataset.")
            else:
                self.nvalid_grad = round((self.grad_model.n_datapoints - self.grad_model.ntrain)  / 2)
        if not self.grad and not self.hess:
            raise Exception("Not much point in using this Neural Network without gradients or Hessians")

        self.ndat_full = self.fullraw_X.shape[0]
        self.fullraw_Xr = self.cart_to_interatomic(self.fullraw_X)

    def cart_to_interatomic(self, cartflat):
        ndat = cartflat.shape[0]
        cart_3d = cartflat.reshape((ndat, self.natoms, 3))
        Xr = np.zeros((ndat, self.n_interatomics))
        for atom in range(1, self.natoms):
            # Create an array of duplicated cartesian coordinates of this particular atom, for every geometry, which is the same shape as 'cartesians'
            tmp1 = np.broadcast_to(cart_3d[:,atom,:], (ndat, 3))
            tmp2 = np.tile(tmp1, (self.natoms,1,1)).transpose(1,0,2)
            # Take the non-redundant norms of this atom to all atoms after it in cartesian array
            diff = tmp2[:, 0:atom,:] - cart_3d[:, 0:atom,:]
            norms = np.sqrt(np.einsum('...ij,...ij->...i', diff , diff))
            # Fill in the norms into interatomic distances 2d array , n_interatomic_distances)
            if atom == 1:
                idx1, idx2 = 0, 1
            if atom > 1:
                x = int((atom**2 - atom) / 2)
                idx1, idx2 = x, x + atom
            Xr[:, idx1:idx2] = norms
        return Xr 

    def split_train_test(self, params, validation_size=None, grad_validation_size=None, hess_validation_size=None, precision=32):
        # Do preprocess with interatomic distances
        #self.full_X, self.full_y, self.Xscaler, self.yscaler = self.preprocess(params, self.fullraw_Xr, self.fullraw_y)
        # All geometries and energies preprocessed
        self.full_X, self.full_y, self.Xscaler, self.yscaler = self.preprocess(params, self.fullraw_Xr, self.fullraw_y) # Cartesian inputs

        # Full partitions of energy, gradient, and Hessian datasets
        self.X = self.full_X[0:self.n_datapoints]
        self.y = self.full_y[0:self.n_datapoints]
        if self.grad:
            # Cartesian
            self.X_grad = self.full_X[self.grad_offset:self.grad_offset+self.grad_model.n_datapoints]
            self.y_grad = self.full_y[self.grad_offset:self.grad_offset+self.grad_model.n_datapoints]
            #self.X_grad = self.grad_model.raw_X
            #self.y_grad = self.grad_model.raw_y
            self.grad_grad = self.grad_model.raw_grad
        if self.hess:
            # Cartesian
            self.X_hess = self.hess_model.raw_X
            self.y_hess = self.hess_model.raw_y
            self.grad_hess = self.hess_model.raw_grad
            self.hess_hess = self.hess_model.raw_hess
        
        # Grads and Hess stay raw as transformations are not well defined
        #if self.grad:
        #    self.X_grad, self.y_grad, self.Xscaler_grad, self.yscaler_grad = self.preprocess(params, self.grad_model.raw_X, self.grad_model.raw_y)
        #if self.hess:
        #    self.X_hess, self.y_hess, self.Xscaler_hess, self.yscaler_hess = self.preprocess(params, self.hess_model.raw_X, self.hess_model.raw_y)
        
        if self.sampler == 'user_supplied':
            # TODO: Not implemented
            raise Exception("User supplied sampling not supported for differentiable neural networks")
            self.Xtr = self.transform_new_X(self.raw_Xtr, params, self.Xscaler)
            self.ytr = self.transform_new_y(self.raw_ytr, self.yscaler)
            self.Xtest = self.transform_new_X(self.raw_Xtest, params, self.Xscaler)
            self.ytest = self.transform_new_y(self.raw_ytest, self.yscaler)
            #if self.grad:
            #    self.Xtr_grad   = self.transform_new_X(self.grad_model.raw_Xtr, params, self.Xscaler_grad)
            #    self.ytr_grad   = self.transform_new_y(self.grad_model.raw_ytr, self.yscaler_grad)
            #    self.Xtest_grad = self.transform_new_X(self.grad_model.raw_Xtest, params, self.Xscaler_grad)
            #    self.ytest_grad = self.transform_new_y(self.grad_model.raw_ytest, self.yscaler_grad)
            #if self.hess:
            #    self.Xtr_hess   = self.transform_new_X(self.hess_model.raw_Xtr, params, self.Xscaler_hess)
            #    self.ytr_hess   = self.transform_new_y(self.hess_model.raw_ytr, self.yscaler_hess)
            #    self.Xtest_hess = self.transform_new_X(self.hess_model.raw_Xtest, params, self.Xscaler_hess)
            #    self.ytest_hess = self.transform_new_y(self.hess_model.raw_ytest, self.yscaler_hess)
            
            if self.valid_path:
                self.Xvalid = self.transform_new_X(self.raw_Xvalid, params, self.Xscaler)
                self.yvalid = self.transform_new_y(self.raw_yvalid, self.yscaler)
                #if self.grad:
                #    self.Xvalid_grad = self.transform_new_X(self.grad_model.raw_Xvalid, params, self.Xscaler_grad)
                #    self.yvalid_grad = self.transform_new_y(self.grad_model.raw_yvalid, self.yscaler_grad)
                #if self.hess:
                #    self.Xvalid_hess = self.transform_new_X(self.hess_model.raw_Xvalid, params, self.Xscaler_hess)
                #    self.yvalid_hess = self.transform_new_y(self.hess_model.raw_yvalid, self.yscaler_hess)
            else:
                raise Exception("Please provide a validation set for Neural Network training.")
        
        else:
            self.Xtr = self.X[self.train_indices]
            self.ytr = self.y[self.train_indices]
            if self.grad:
                self.full_indices_grad = np.arange(self.y_grad.shape[0])
                self.Xtr_grad = self.X_grad[self.grad_model.train_indices]
                self.ytr_grad = self.y_grad[self.grad_model.train_indices]
                self.gradtr_grad = self.grad_grad[self.grad_model.train_indices]
            #if self.hess:
            #    self.full_indices_hess = np.arange(self.y_hess.shape[0])
            #    self.Xtr_hess = self.X_hess[self.hess_model.train_indices]
            #    self.ytr_hess = self.y_hess[self.hess_model.train_indices]
            #TODO: this is splitting validation data in the same way at every model build, not necessary.
            self.valid_indices, self.new_test_indices = train_test_split(self.test_indices, train_size = validation_size, random_state=42)
            #if self.grad:
            #    self.valid_indices_grad, self.new_test_indices_grad = train_test_split(self.grad_model.test_indices, train_size = grad_validation_size, random_state=42)
            #if self.hess:
            #    self.valid_indices_hess, self.new_test_indices_hess = train_test_split(self.hess_model.test_indices, train_size = hess_validation_size, random_state=42)
            if validation_size:
                self.Xvalid = self.X[self.valid_indices]             
                self.yvalid = self.y[self.valid_indices]
                self.Xtest = self.X[self.new_test_indices]
                self.ytest = self.y[self.new_test_indices]
                if self.grad:
                    self.valid_indices_grad, self.new_test_indices_grad = train_test_split(self.grad_model.test_indices, train_size = grad_validation_size, random_state=42)
                    self.Xvalid_grad    = self.X_grad[self.valid_indices_grad]             
                    self.yvalid_grad    = self.y_grad[self.valid_indices_grad]
                    self.gradvalid_grad = self.grad_grad[self.valid_indices_grad]
                    self.Xtest_grad     = self.X_grad[self.new_test_indices_grad]
                    self.ytest_grad     = self.y_grad[self.new_test_indices_grad]
                    self.gradtest_grad  = self.grad_grad[self.new_test_indices_grad]
                #if self.hess:
                #    self.Xvalid_hess = self.X_hess[self.valid_indices_hess]             
                #    self.yvalid_hess = self.y_hess[self.valid_indices_hess]
                #    self.Xtest_hess  = self.X_hess[self.new_test_indices_hess]
                #    self.ytest_hess  = self.y_hess[self.new_test_indices_hess]
            else:
                raise Exception("Please specify a validation set size for Neural Network training.")

        # convert to Torch Tensors
        if precision == 32:
            self.Xtr    = torch.tensor(self.Xtr,   dtype=torch.float32, requires_grad=True)
            self.ytr    = torch.tensor(self.ytr,   dtype=torch.float32)
            self.Xtest  = torch.tensor(self.Xtest, dtype=torch.float32, requires_grad=True)
            self.ytest  = torch.tensor(self.ytest, dtype=torch.float32)
            self.Xvalid = torch.tensor(self.Xvalid,dtype=torch.float32, requires_grad=True)
            self.yvalid = torch.tensor(self.yvalid,dtype=torch.float32)
            self.X = torch.tensor(self.X,dtype=torch.float32, requires_grad=True)
            self.y = torch.tensor(self.y,dtype=torch.float32)
            
            if self.grad:
                self.Xtr_grad       = torch.tensor(self.Xtr_grad,       dtype=torch.float32, requires_grad=True)
                self.ytr_grad       = torch.tensor(self.ytr_grad,       dtype=torch.float32)
                self.gradtr_grad    = torch.tensor(self.gradtr_grad,    dtype=torch.float32, requires_grad=True)
                self.Xtest_grad     = torch.tensor(self.Xtest_grad,     dtype=torch.float32, requires_grad=True)
                self.ytest_grad     = torch.tensor(self.ytest_grad,     dtype=torch.float32)
                self.gradtest_grad  = torch.tensor(self.gradtest_grad,  dtype=torch.float32, requires_grad=True)
                self.Xvalid_grad    = torch.tensor(self.Xvalid_grad,    dtype=torch.float32, requires_grad=True)
                self.yvalid_grad    = torch.tensor(self.yvalid_grad,    dtype=torch.float32)
                self.gradvalid_grad = torch.tensor(self.gradvalid_grad, dtype=torch.float32, requires_grad=True)
                # Full gradient data sets
                self.X_grad    = torch.tensor(self.X_grad,    dtype=torch.float32, requires_grad=True)
                self.y_grad    = torch.tensor(self.y_grad,    dtype=torch.float32)
                self.grad_grad = torch.tensor(self.grad_grad, dtype=torch.float32, requires_grad=True)
                
                if False:
                    # WTF is all this? Consider renaming
                    #self.Xtr_grad   = torch.tensor(self.Xtr_grad,   dtype=torch.float32)
                    #self.ytr_grad   = torch.tensor(self.ytr_grad,   dtype=torch.float32)
                    # Geom of gradient training points
                    self.Xtr_t_grad = torch.tensor(self.full_X[self.grad_offset + self.grad_model.train_indices], dtype=torch.float32, requires_grad=True)
                    ## grad gradient training points
                    self.gradtr_grad = torch.tensor(self.grad_grad[self.grad_model.train_indices], dtype=torch.float32, requires_grad=True)
                    #self.gradtr_grad = torch.tensor(self.grad_model.raw_grad[self.grad_model.train_indices], dtype=torch.float32, requires_grad=True)
                
                    #self.Xtest_grad = torch.tensor(self.Xtest_grad, dtype=torch.float32)
                    #self.ytest_grad = torch.tensor(self.ytest_grad, dtype=torch.float32)
                    #self.Xvalid_grad = torch.tensor(self.Xvalid_grad,dtype=torch.float32)
                    #self.yvalid_grad = torch.tensor(self.yvalid_grad,dtype=torch.float32)
                
                    # Test geometries for grad
                    self.Xt_grad = torch.tensor(self.full_X[self.grad_offset:],dtype=torch.float32, requires_grad=True)
                    #self.X_grad = torch.tensor(self.X_grad,dtype=torch.float32, requires_grad=True)
                    # All grad energies
                    self.y_grad = torch.tensor(self.full_y[self.grad_offset:],dtype=torch.float32)
                    # All grad gradients
                    self.grad_grad = torch.tensor(self.grad_grad, dtype=torch.float32)
            if self.hess:
                self.Xtr_hess   = torch.tensor(self.Xtr_hess,   dtype=torch.float32)
                self.ytr_hess   = torch.tensor(self.ytr_hess,   dtype=torch.float32)
                self.Xtest_hess = torch.tensor(self.Xtest_hess, dtype=torch.float32)
                self.ytest_hess = torch.tensor(self.ytest_hess, dtype=torch.float32)
                self.Xvalid_hess = torch.tensor(self.Xvalid_hess,dtype=torch.float32)
                self.yvalid_hess = torch.tensor(self.yvalid_hess,dtype=torch.float32)
                self.X_hess = torch.tensor(self.X_hess,dtype=torch.float32)
                self.y_hess = torch.tensor(self.y_hess,dtype=torch.float32)
        elif precision == 64:
            raise Exception("64 bit float in diff_neural_network not supported currently")
            self.Xtr    = torch.tensor(self.Xtr,   dtype=torch.float64)
            self.ytr    = torch.tensor(self.ytr,   dtype=torch.float64)
            self.Xtest  = torch.tensor(self.Xtest, dtype=torch.float64)
            self.ytest  = torch.tensor(self.ytest, dtype=torch.float64)
            self.Xvalid = torch.tensor(self.Xvalid,dtype=torch.float64)
            self.yvalid = torch.tensor(self.yvalid,dtype=torch.float64)
            self.X = torch.tensor(self.X,dtype=torch.float64)
            self.y = torch.tensor(self.y,dtype=torch.float64)
            if self.grad:
                self.Xtr_grad   = torch.tensor(self.Xtr_grad,   dtype=torch.float64, requires_grad=True)
                self.ytr_grad   = torch.tensor(self.ytr_grad,   dtype=torch.float64)
                self.Xtest_grad = torch.tensor(self.Xtest_grad, dtype=torch.float64, requires_grad=True)
                self.ytest_grad = torch.tensor(self.ytest_grad, dtype=torch.float64)
                self.Xvalid_grad = torch.tensor(self.Xvalid_grad,dtype=torch.float64, requires_grad=True)
                self.yvalid_grad = torch.tensor(self.yvalid_grad,dtype=torch.float64)
                self.X_grad = torch.tensor(self.X_grad,dtype=torch.float64, requires_grad=True)
                self.y_grad = torch.tensor(self.y_grad,dtype=torch.float64)
            if self.hess:
                self.Xtr_hess   = torch.tensor(self.Xtr_hess,   dtype=torch.float64)
                self.ytr_hess   = torch.tensor(self.ytr_hess,   dtype=torch.float64)
                self.Xtest_hess = torch.tensor(self.Xtest_hess, dtype=torch.float64)
                self.ytest_hess = torch.tensor(self.ytest_hess, dtype=torch.float64)
                self.Xvalid_hess = torch.tensor(self.Xvalid_hess,dtype=torch.float64)
                self.yvalid_hess = torch.tensor(self.yvalid_hess,dtype=torch.float64)
                self.X_hess = torch.tensor(self.X_hess,dtype=torch.float64)
                self.y_hess = torch.tensor(self.y_hess,dtype=torch.float64)
        else:
            raise Exception("Invalid option for 'precision'")

    def build_model(self, params, maxit=1000, val_freq=10, es_patience=2, opt='lbfgs', tol=1, decay=False, verbose=False, precision=32, return_model=False):
        #params["morse_transform"]["morse"] = False
        #params["pip"]["pip"] = False
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
        inp_dim = self.X.shape[1]
        #inp_dim = self.inp_dim
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
        #labmda = 1e-4
        for epoch in range(1,maxit):
            #print(f"Begin epoch {epoch}")
            def closure():
                optimizer.zero_grad()
                y_pred = model(self.Xtr)
                #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = torch.sqrt(metric(y_pred, self.ytr))
                if self.train_grad:
                    #y_pred_grad = model(self.Xtr_t_grad)
                    y_pred_grad = model(self.Xtr_grad)
                    #gradspred, = autograd.grad(y_pred_grad, self.Xtr_t_grad, 
                    #       grad_outputs=y_pred_grad.data.new(y_pred_grad.shape).fill_(1),
                    #       create_graph=True)
                    gradspred, = autograd.grad(y_pred_grad, self.Xtr_grad, 
                           grad_outputs=y_pred_grad.data.new(y_pred_grad.shape).fill_(1),
                           create_graph=True)
                    gradpred_cart = self.transform_grad(self.grad_model.train_indices, gradspred, params, self.Xscaler, self.yscaler, precision=precision)
                    #grad_error = torch.sqrt(torch.sum((gradpred_cart - self.gradtr_grad)**2))
                    grad_e_error = torch.sqrt(metric(y_pred_grad, self.ytr_grad))
                    grad_grad_error = torch.sqrt(torch.mean(torch.sum(1.0*(gradpred_cart - self.gradtr_grad) ** 2,dim=1).reshape(-1,1)))
                    floss = loss + grad_e_error + grad_grad_error
                if self.train_hess:
                    floss += 0.0
                if self.train_grad:
                    floss.backward()
                    return floss
                else:
                    loss.backward()
                    return loss
            optimizer.step(closure)
            # validate
            if epoch % val_freq == 0:
                #if self.grad:
                #    valid_grad_pred = model(self.Xvalid_grad)
                #    valid_gradspred, = autograd.grad(valid_grad_pred, self.Xvalid_grad, 
                #                       grad_outputs=valid_grad_pred.data.new(valid_grad_pred.shape).fill_(1),
                #                       create_graph=True)
                with torch.no_grad():
                    tmp_pred = model(self.Xvalid) 
                    tmp_loss = metric(tmp_pred, self.yvalid)
                    val_error_rmse = np.sqrt(tmp_loss.item() * loss_descaler) * hartree2cm # loss_descaler converts MSE in scaled data domain to MSE in unscaled data domain
                    if self.train_grad:
                        valid_grad_pred = model(self.Xvalid_grad)
                        valid_grad_loss = metric(valid_grad_pred, self.yvalid_grad)
                        valid_grad_E_error_rmse = np.sqrt(valid_grad_loss.item() * loss_descaler) * hartree2cm
                        #valid_grad_cart = self.transform_grad(self.valid_indices_grad, valid_gradspred, params, self.Xscaler, self.yscaler, precision=precision)
                        #valid_grad_grad_error_rmse = torch.sqrt(torch.mean(torch.sum(1.0*(valid_grad_cart - self.grad_grad) ** 2,dim=1).reshape(-1,1))) * hartree2cm
                        # Add energy RMSE from gradient data set to valid_error_rmse
                        val_error_rmse = np.sqrt(((val_error_rmse**2) + (valid_grad_E_error_rmse**2))/2.0)
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
        
        # Error w/ grad
        grad_full_pred = model(self.X_grad)
        full_gradspred, = autograd.grad(grad_full_pred, self.X_grad, 
                           grad_outputs=grad_full_pred.data.new(grad_full_pred.shape).fill_(1),
                           create_graph=True)
        with torch.no_grad():
            grad_full_loss = metric(grad_full_pred, self.y_grad)
            grad_test_loss = metric(grad_full_pred, self.y_grad)
            grad_val_loss = metric(grad_full_pred, self.y_grad)
            grad_full_E_error_rmse = np.sqrt(grad_full_loss.item() * loss_descaler) * hartree2cm
            grad_test_E_error_rmse = np.sqrt(grad_test_loss.item() * loss_descaler) * hartree2cm
            grad_val_E_error_rmse = np.sqrt(grad_val_loss.item() * loss_descaler) * hartree2cm
            grad_full_gradcart = self.transform_grad(slice(self.grad_offset, self.ndat_full), full_gradspred, params, self.Xscaler, self.yscaler, precision=precision)
            grad_test_gradcart = grad_full_gradcart[self.new_test_indices_grad]
            grad_val_gradcart = grad_full_gradcart[self.valid_indices_grad]
        
            grad_full_grad_error_rmse = torch.sqrt(torch.mean(torch.sum(1.0*(grad_full_gradcart - self.grad_grad) ** 2,dim=1).reshape(-1,1))) * hartree2cm
            grad_test_grad_error_rmse = torch.sqrt(torch.mean(torch.sum(1.0*(grad_test_gradcart - self.gradtest_grad) ** 2,dim=1).reshape(-1,1))) * hartree2cm
            grad_val_grad_error_rmse = torch.sqrt(torch.mean(torch.sum(1.0*(grad_val_gradcart - self.gradvalid_grad) ** 2,dim=1).reshape(-1,1))) * hartree2cm
        
        #print(f"Grad. Energy Error: {full_grad_E_error_rmse.item()}   Gradient Error: {full_grad_grad_error_rmse.item()}")
        
        output_str = "    {} set RMSE (cm-1):"
        print(f"{output_str.format('Test'):45s} {test_error_rmse:5.2f}")
        print(f"{output_str.format('Validation'):45s} {val_error_rmse:5.2f}") 
        print(f"{output_str.format('Full'):45s} {full_error_rmse:5.2f}")
        #print("Test set RMSE (cm-1): {:5.2f}  Validation set RMSE (cm-1): {:5.2f} Full dataset RMSE (cm-1): {:5.2f}".format(test_error_rmse, val_error_rmse, full_error_rmse))
        if self.grad:
            grad_output_str = "    {} set RMSE {} (cm-1{}):"
            print(f"{grad_output_str.format('Test','Energy',''):45s} {grad_test_E_error_rmse:5.2f}")
            print(f"{grad_output_str.format('Validation','Energy',''):45s} {grad_val_E_error_rmse:5.2f}") 
            print(f"{grad_output_str.format('Full','Energy',''):45s} {grad_full_E_error_rmse:5.2f}")
            print(f"{grad_output_str.format('Test','Gradient','/bohr'):45s} {grad_test_grad_error_rmse.item():5.2f}")
            print(f"{grad_output_str.format('Validation','Gradient','/bohr'):45s} {grad_val_grad_error_rmse.item():5.2f}") 
            print(f"{grad_output_str.format('Full','Gradient','/bohr'):45s} {grad_full_grad_error_rmse.item():5.2f}")
            #print("Grad. Test set RMSE Energy (cm-1): {:5.2f}  Grad. Validation set RMSE Energy (cm-1): {:5.2f} Grad. Full dataset RMSE Energy (cm-1): {:5.2f}".format(grad_test_E_error_rmse, grad_val_E_error_rmse, grad_full_E_error_rmse))
            #print("Grad. Test set RMSE Gradient (cm-1/bohr): {:5.2f}  Grad. Validation set RMSE Gradient (cm-1/bohr): {:5.2f} Grad. Full dataset RMSE Gradient (cm-1/bohr): {:5.2f}".format(grad_test_grad_error_rmse.item(), grad_val_grad_error_rmse.item(), grad_full_grad_error_rmse.item()))
            #print(f"Grad. Energy Test set RMSE (cm-1): {full_grad_E_error_rmse.item():5.2f}   ")
            #print(f"Grad. Energy Test set RMSE (cm-1): {full_grad_E_error_rmse.item():5.2f}   Gradient Error: {full_grad_grad_error_rmse.item()}")
        #assert False
        if return_model:
            return model, test_error_rmse, val_error_rmse, full_error_rmse 
        else:
            return test_error_rmse, val_error_rmse
    
    def holland(self, X, grad_vec):
        # Assume PIP true, scale_X/y std, and noting else (default params of NN architecture search)
        # Transfrom known gradient from Cartesian to PIP, NOT GENERALIZABLE!!!
        print(grad_vec)
        # Cart to dist
        ndat = grad_vec.shape[0]
        ncart = self.natoms*3
        nr = self.n_interatomics
        X_r, B1_r, B2_r = cart_dist_B_2(X) # dr/dx
        print(B1_r)
        print("SVD")
        print(np.linalg.svd(B1_r)[1][:,-1])
        A_r = np.zeros((ndat, ncart, nr))
        for i in range(ndat):
            A_r[i,:,:] = np.linalg.pinv(B1_r[i,:,:])
        print(A_r)
        print(np.dot(B1_r[0,:,:], A_r[0,:,:]))
        print(np.dot(A_r[0,:,:], B1_r[0,:,:]))
        # Calc. C
        #C_r = np.einsum("naij,nir,njs->nars", B2_r, A_r, A_r)

        # Calc. Grad. and Hess. in interatomic dist. from Cart.
        G_r = np.einsum("ni,nir->nr", grad_vec, A_r)
        print(G_r)
        # Dist to PIP
        path = os.path.join(package_directory, "lib", self.molecule_type, "output")
        X_p, degrees, B1_p, B2_p = self.pip_B.transform(path, X_r)
        npip = X_p.shape[1]
        A_p = np.zeros((ndat, nr, npip))
        for i in range(ndat):
            A_p[i,:,:] = np.linalg.pinv(B1_p[i,:,:])
        G_p = np.einsum("ni,nir->nr", G_r, A_p)
        print(G_p)
        # PIP to std PIP
        X_scale = self.Xscaler.transform(X_p)
        G_scale = G_p / self.yscaler.scale_ 
        G_scale *= self.Xscaler.scale_[None,:]
        print(G_scale)
        return G_scale

    def preprocess(self, params, raw_X, raw_y):
        """
        Preprocess raw data according to hyperparameters
        """
        #raw_X = deepcopy(raw_X_in)
        if params['morse_transform']['morse']:
            raw_X = morse(raw_X, params['morse_transform']['morse_alpha'])
            self.raw_Xm = deepcopy(raw_X)
        if params['pip']['pip']:
            # find path to fundamental invariants form molecule type AxByCz...
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            raw_X, self.degrees = interatomics_to_fundinvar(raw_X,path)
            self.raw_Xp = deepcopy(raw_X)
            if params['pip']['degree_reduction']:
                raw_X = degree_reduce(raw_X, self.degrees)
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

    def transform_grad(self, Xindices, grad, params, Xscaler=None, yscaler=None, precision=32):
        if precision == 32:
            dtype = torch.float32
        elif precision == 64:
            dtype = torch.float64
        # Transform gradient from NN to Cartesian
        scaler = torch.ones(grad.shape[1], dtype=dtype)
        if yscaler:
            # Multiply by stdev of E, dE/dX = dEmean/dX * sigma
            if params['scale_y'] == "std":    
                #grad *= torch.from_numpy(yscaler.scale_)
                scaler *= torch.tensor(yscaler.scale_, dtype=dtype)
            else:
                #grad /= torch.from_numpy(yscaler.scale_)
                scaler /= torch.tensor(yscaler.scale_, dtype=dtype)
            #grad *= np.sqrt(yscaler)
        if Xscaler:
            # Divide by stdev of X, dXmean/dX = 1/sigma
            #X_std = torch.from_numpy(Xscaler.scale_[None,:])
            X_sc = torch.tensor(Xscaler.scale_, dtype=dtype)
            #X = Xscaler.inverse_transform(X)
            if params['scale_X']['scale_X'] == "std":
                #grad *= X_std**-1
                scaler *= X_sc**-1
            else:
                #grad *= X_std
                scaler *= X_sc
        # Rename lower grads TODO
        scaled_grad = grad * scaler
        if params['pip']['pip']:
            path = os.path.join(package_directory, "lib", self.molecule_type, "output")
            #X, degrees, B1_p, B2_p = pip_B(path, X)
            if params['pip']['degree_reduction']:
                B1_dr = degree_B1(self.raw_Xp[Xindices,:], self.degrees)
                #B2_dr = degree_B2(X, degrees)
                scaled_grad *= torch.from_numpy(B1_dr)
            if params['morse_transform']['morse']:
                #X, degrees, B1_p, B2_p = pip_B(path, self.raw_Xm[Xindices,:])
                X, degrees, B1_p, B2_p = self.pip_B.transform(self.raw_Xm[Xindices,:])
            else:
                #X, degrees, B1_p, B2_p = pip_B(path, self.fullraw_Xr[Xindices,:])
                X, degrees, B1_p, B2_p = self.pip_B.transform(self.fullraw_Xr[Xindices,:])
            scaled_grad = torch.einsum("np,npi->ni", scaled_grad, torch.tensor(B1_p, dtype=dtype))
        if params['morse_transform']['morse']:
            B1_m = morse_B1(self.fullraw_Xr[Xindices,:], alpha=params['morse_transform']['morse_alpha'])
            scaled_grad = scaled_grad * torch.tensor(B1_m, dtype=dtype)
        #return scaled_grad
        # r to Cart.
        X_r, B1_r, B2_r = cart_dist_B_2(self.fullraw_X[Xindices]) # dr/dx
        #print(B1_r[5,:])
        grad_cart = torch.einsum("np,npi->ni", scaled_grad, torch.tensor(B1_r, dtype=dtype))
        return grad_cart

    def transform_hess(self, hess, params):
        # Transform Hessian from NN to Cartesian
        pass
