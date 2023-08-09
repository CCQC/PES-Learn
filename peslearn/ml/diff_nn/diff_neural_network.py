from ..neural_network import NeuralNetwork

class DiffNeuralNetwork(NeuralNetwork):
    def __init__(self, dataset_path, input_obj, zmat_idxs, perm_vec, molecule_type=None, molecule=None, 
                 train_path=None, test_path=None, valid_path=None, 
                 grad_train_path=None, grad_test_path=None, grad_valid_path=None,
                 hess_train_path=None, hess_test_path=None, hess_valid_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        self.zmat_idxs = zmat_idxs
        self.perm_vec = perm_vec
        self.natoms = len(self.perm_vec)

    def split_train_test(self, params, validation_size=None, precision=32):
        return super().split_train_test(params, validation_size, precision)

    def preprocess(self, params, raw_X, raw_y):
        # Rewrite to take raw_X in internal coordinates
        # Rescale gradients and Hessians as per raw_y
        pass
    
    def build_model(self, params, maxit=1000, val_freq=10, es_patience=2, opt='lbfgs', tol=1, decay=False, verbose=False, precision=32, return_model=False):
        return super().build_model(params, maxit, val_freq, es_patience, opt, tol, decay, verbose, precision, return_model)