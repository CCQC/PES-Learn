import numpy as np
from sklearn.utils import shuffle
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from .gpytorch_gpr import GaussianProcess
import itertools
import gc
from ..constants import hartree2cm

class SVI(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, inducing_points):
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        #variational_strategy = gpytorch.variational.CiqVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVI, self).__init__(variational_strategy)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = ScaleKernel(RBFKernel(ard_num_dims = train_x.size(1)))
    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar(x)
        #np.savetxt('/home/smg13363/GPR_PES/gpytorch_test_space/spgp/benchmarks/array.dat', covar_x.detach().numpy())
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = ScaleKernel(RBFKernel(ard_num_dims=1, active_dims=(1))) * ScaleKernel(RBFKernel(ard_num_dims=train_x.size()[1])) + ScaleKernel(RBFKernel(ard_num_dims=train_x.size()[1]))

    def forward(self, x):
        mean_x = self.mean(x)
        kernel_x = self.covar(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, kernel_x)


class MFGP(GaussianProcess):
    def __init__(self, dataset_path, lf_dataset_path, input_obj, input_obj_l, molecule_type=None, molecule=None, train_path=None, test_path=None, train_path_low=None, test_path_low=None, epochs=(100,100), num_inducing=50, batchsize=100):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)
        self.m_low = GaussianProcess(lf_dataset_path, input_obj_l, molecule_type, molecule, train_path_low, test_path_low)
        torch.set_default_tensor_type(torch.DoubleTensor)
        #gpytorch.settings.tridiagonal_jitter(1e-5)
        torch.set_default_dtype(torch.float64)
        #gpytorch.settings.lazily_evaluate_kernels(False)
        self.epochs_h = epochs[1]
        self.epochs_l = epochs[0]
        self.num_inducing = num_inducing
        self.batchsize = batchsize

    """
    Process LF and HF data
    Build models simultaneously, LF the HF
    Vet based on HF model
    Win?
    """

    def split_train_test(self, params, precision=64):
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
        self.X_l, self.y_l, self.Xscaler_l, self.yscaler_l = self.preprocess(params, self.m_low.raw_X, self.m_low.raw_y)
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

        if self.m_low.sampler == 'user_supplied':
            self.Xtr_l = self.transform_new_X(self.m_low.raw_Xtr, params, self.Xscaler_l)
            self.ytr_l = self.transform_new_y(self.m_low.raw_ytr, self.yscaler_l)
            self.Xtest_l = self.transform_new_X(self.m_low.raw_Xtest, params, self.Xscaler_l)
            self.ytest_l = self.transform_new_y(self.m_low.raw_ytest, self.yscaler_l)    
        else:
            self.Xtr_l = self.X_l[self.m_low.train_indices]
            self.ytr_l = self.y_l[self.m_low.train_indices]
            self.Xtest_l = self.X_l[self.m_low.test_indices]
            self.ytest_l = self.y_l[self.m_low.test_indices]
            
        # convert to Torch Tensors
        if precision == 32:
            self.Xtr    = torch.tensor(self.Xtr,   dtype=torch.float32)
            self.ytr    = torch.tensor(self.ytr,   dtype=torch.float32)
            self.Xtest  = torch.tensor(self.Xtest, dtype=torch.float32)
            self.ytest  = torch.tensor(self.ytest, dtype=torch.float32)
            self.X      = torch.tensor(self.X,     dtype=torch.float32)
            self.y      = torch.tensor(self.y,     dtype=torch.float32)
            
            self.Xtr_l    = torch.tensor(self.Xtr_l,   dtype=torch.float32)
            self.ytr_l    = torch.tensor(self.ytr_l,   dtype=torch.float32)
            self.Xtest_l  = torch.tensor(self.Xtest_l, dtype=torch.float32)
            self.ytest_l  = torch.tensor(self.ytest_l, dtype=torch.float32)
            self.X_l      = torch.tensor(self.X_l,     dtype=torch.float32)
            self.y_l      = torch.tensor(self.y_l,     dtype=torch.float32)
        
        elif precision == 64:
            self.Xtr    = torch.tensor(self.Xtr,   dtype=torch.float64)
            self.ytr    = torch.tensor(self.ytr,   dtype=torch.float64)
            self.Xtest  = torch.tensor(self.Xtest, dtype=torch.float64)
            self.ytest  = torch.tensor(self.ytest, dtype=torch.float64)
            self.X      = torch.tensor(self.X,     dtype=torch.float64)
            self.y      = torch.tensor(self.y,     dtype=torch.float64)
            
            self.Xtr_l    = torch.tensor(self.Xtr_l,   dtype=torch.float64)
            self.ytr_l    = torch.tensor(self.ytr_l,   dtype=torch.float64)
            self.Xtest_l  = torch.tensor(self.Xtest_l, dtype=torch.float64)
            self.ytest_l  = torch.tensor(self.ytest_l, dtype=torch.float64)
            self.X_l      = torch.tensor(self.X_l,     dtype=torch.float64)
            self.y_l      = torch.tensor(self.y_l,     dtype=torch.float64)
        
        else:
            raise Exception("Invalid option for 'precision'")



    def build_model(self, params, nrestarts=10, maxiter=1000, seed=0):
        self.split_train_test(params)
        np.random.seed(seed)     # make GPy deterministic for a given hyperparameter config
        print("\n")
        print("-"*128)
        print(f"\nParams: \n{params}")
        # LF Training
        self.Z = self.X_l[:self.num_inducing,:]
        train_l_ds = torch.utils.data.TensorDataset(self.Xtr_l, self.ytr_l)
        train_l_loader = torch.utils.data.DataLoader(train_l_ds, batch_size = self.batchsize, shuffle=True)
        self.likelihood_l = gpytorch.likelihoods.GaussianLikelihood()
        self.model_l = SVI(self.Xtr_l, self.ytr_l, inducing_points = self.Z)
        self.model_l.train()
        self.likelihood_l.train()
        opt_ngd = gpytorch.optim.NGD(self.model_l.variational_parameters(), num_data=self.ytr_l.size(0), lr=0.01)
        opt_hyp = torch.optim.Adam([{'params':self.model_l.parameters()}, {'params':self.likelihood_l.parameters()}], lr=0.01)
        mll_l = gpytorch.mlls.VariationalELBO(self.likelihood_l, self.model_l, num_data=self.ytr_l.size(0))
        for i in range(self.epochs_l):
            for x_batch, y_batch in train_l_loader:
                opt_ngd.zero_grad()
                opt_hyp.zero_grad()
                out = self.model_l(x_batch)
                loss = -mll_l(out, y_batch.squeeze())
                loss.backward()
                opt_ngd.step()
                opt_hyp.step()
        print('\nLF Training Done')
        self.model_l.eval()
        self.likelihood_l.eval()

        # HF Training

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean_low = self.model_l(self.Xtr).mean.unsqueeze(-1)

        xx = torch.hstack((self.Xtr.squeeze(0), mean_low.squeeze(0)))
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GP(xx, self.ytr.squeeze(), self.likelihood)
        self.likelihood.train()
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(self.epochs_h):
            opt.zero_grad()
            out = self.model(xx)
            loss = -mll(out, torch.squeeze(self.ytr))
            loss.backward()
            opt.step()
        print('HF Training Done\n')
        self.model.eval()
        self.likelihood.eval()
        gc.collect(2) #fixes some memory leak issues with certain BLAS configs
    

    def predict(self, hf_model, lf_model, x_in):
        #xpred_dataset = torch.utils.data.TensorDataset(x_in)
        xpred_dataloader = torch.utils.data.DataLoader(x_in, batch_size = 1024, shuffle = False)
        prediction = torch.tensor([0.])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for x_batch in xpred_dataloader:
                lf_pred = lf_model(x_batch).mean.unsqueeze(-1)
                xx = torch.hstack((x_batch, lf_pred.squeeze(0)))
                hfpred = hf_model(xx).mean.unsqueeze(1)
                prediction = torch.cat([prediction, hfpred.squeeze(-1)])
        return prediction[1:].unsqueeze(1)

    def vet_model(self, model):
        """Convenience method for getting model errors of test and full datasets"""
        #pred_test = self.predict(model, self.model_l, self.Xtest)
        pred_full = self.predict(model, self.model_l, self.X)
        #error_test = self.compute_error(self.ytest.squeeze(), pred_test, self.yscaler)
        error_full, median_error, max_errors = self.compute_error(self.y.squeeze(0), pred_full, yscaler=self.yscaler, max_errors=5)
        #print("Test Dataset {}".format(round(hartree2cm * error_test,2)), end='  ')
        print("Full Dataset {}".format(round(hartree2cm * error_full,2)), end='     ')
        print("Median error: {}".format(np.round(median_error,2)), end='  ')
        print("Max 5 errors: {}".format(np.sort(np.round(max_errors.flatten(),1))),'\n')
        print("-"*128)
        return error_full # was test
     
