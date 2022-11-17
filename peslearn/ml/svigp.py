import numpy as np
import torch
import gpytorch
from .gpytorch_gpr import GaussianProcess
import itertools
import gc

class SVI(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, inducing_points):
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        #variational_strategy = gpytorch.variational.CiqVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVI, self).__init__(variational_strategy)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = train_x.size(1)))
    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar(x)
        #np.savetxt('/home/smg13363/GPR_PES/gpytorch_test_space/spgp/benchmarks/array.dat', covar_x.detach().numpy())
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SVIGP(GaussianProcess):
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None, epochs=100, num_inducing=50, batchsize=100):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)
        torch.set_default_tensor_type(torch.DoubleTensor)
        gpytorch.settings.verbose_linalg(True)
        gpytorch.settings.tridiagonal_jitter(1e-5)
        torch.set_default_dtype(torch.float64)
        gpytorch.settings.lazily_evaluate_kernels(False)
        self.epochs = epochs
        self.num_inducing = num_inducing
        self.batchsize = batchsize

    def build_model(self, params, nrestarts=10, maxiter=10000, seed=0):
        print("Hyperparameters: ", params)
        self.split_train_test(params)
        np.random.seed(seed)     # make GPy deterministic for a given hyperparameter config
        #TODO: ARD
        
        #epochs = 1000
        #self.num_inducing = 100
        #self.batchsize = 300
        self.Z = self.X[:self.num_inducing,:]
        #self.Z = torch.rand(self.Xtr.size(0), self.num_inducing)
        #self.Z = self.Xtr[np.random.choice(len(self.Xtr), self.num_inducing, replace=False),:]
        #scale_rand = 1e-4
        #ytritty = self.ytr + scale_rand * torch.Tensor(np.random.rand(self.ytr.size(0))-0.5).unsqueeze(dim=1)
        
        #train_ds = torch.utils.data.TensorDataset(self.Xtr, ytritty)
        train_ds = torch.utils.data.TensorDataset(self.Xtr, self.ytr)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size = self.batchsize, shuffle=True)
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SVI(self.Xtr, self.ytr, inducing_points = self.Z)
        
        #cuda = 'cuda'
        #self.Xtr = self.Xtr.cuda()
        #self.ytr = self.ytr.cuda()
        #self.model = self.model.to(cuda)
        #self.likelihood = self.likelihood.to(cuda)
        
        self.model.train()
        self.likelihood.train()
        opt_ngd = gpytorch.optim.NGD(self.model.variational_parameters(), num_data=self.ytr.size(0), lr=0.01)
        opt_hyp = torch.optim.Adam([{'params':self.model.parameters()}, {'params':self.likelihood.parameters()}], lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.ytr.size(0))
        
        for i in range(self.epochs):
            print(f'\nEpoch {i}/{self.epochs}\n')
            for x_batch, y_batch in train_loader:
                #x_batch = x_batch.to(cuda)
                #y_batch = y_batch.to(cuda)
                opt_ngd.zero_grad()
                opt_hyp.zero_grad()
                out = self.model(x_batch)
                loss = -mll(out, y_batch.squeeze())
                loss.backward()
                opt_ngd.step()
                opt_hyp.step()
            #if i % 5 == 0:
            #    self.model.eval()
            #    self.likelihood.eval()
            #    print(f'\nEpoch {i}/{self.epochs}\n')
            #    self.vet_model(self.model)
            #    self.model.train()
            #    self.likelihood.train()
        print('\nTraining Done\n')
        self.model.eval()
        self.likelihood.eval()
        gc.collect(2) #fixes some memory leak issues with certain BLAS configs


