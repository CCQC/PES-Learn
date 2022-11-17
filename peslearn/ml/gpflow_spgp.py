import numpy as np
import tensorflow as tf
import gpflow
from .gaussian_process import GaussianProcess
import itertools
import gc

class SVIGP(GaussianProcess):
    

    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path)

    def build_model(self, params, nrestarts=10, maxiter=10000, seed=0, dont_do_it_jeffrey=False):
        print("Hyperparameters: ", params)
        # Jeffrey won't do it if he's using MFGP's
        if not dont_do_it_jeffrey:
            self.split_train_test(params)
        np.random.seed(seed)     # make GPy deterministic for a given hyperparameter config
        #TODO: ARD
        
        self.num_inducing = 100
        self.batchsize = 200
        self.Z = self.Xtr[np.random.choice(len(self.Xtr), self.num_inducing, replace=False),:].copy()
        kernel = gpflow.kernels.RBF() + gpflow.kernels.White()
        self.model = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), self.Z, num_data=len(self.Xtr))
        self.elbo = tf.function(self.model.elbo)
        tensor_data = tuple(map(tf.convert_to_tensor, (self.Xtr, self.ytr)))
        self.elbo(tensor_data)  # run it once to trace & compile
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.Xtr, self.ytr)).repeat().shuffle(len(self.Xtr))
        train_iter = iter(self.train_dataset.batch(self.batchsize))
        ground_truth = self.elbo(tensor_data).numpy()
        evals = [self.elbo(minibatch).numpy() for minibatch in itertools.islice(train_iter, 100)]
        #gpflow.set_trainable(self.model.inducing_variable, False)
        self.logf = self.run_adam(self.model, maxiter)
        print(self.logf[-10:])
        gc.collect(2) #fixes some memory leak issues with certain BLAS configs

    def run_adam(self, model, iterations):
        """
        Utility function running the Adam optimizer
    
        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(self.train_dataset.batch(self.batchsize))
        training_loss = model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam()
    
        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, self.model.trainable_variables)
    
        for step in range(iterations):
            optimization_step()
            if step % 10 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)
        return logf

    def predict(self, model, data_in):
        prediction, v1 = model.predict_y(data_in)
        return prediction

