from copy import deepcopy
from ..neural_network import NeuralNetwork

class DeltaNN(NeuralNetwork):
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None, valid_path=None):
        super().__init__(dataset_path, input_obj, molecule_type, molecule, train_path, test_path, valid_path)
        lf_E = self.raw_X[:,-1].reshape(-1,1)
        self.raw_X = self.raw_X[:,:-1]
        self.raw_y = deepcopy(self.raw_y) - lf_E # If modified in place (i.e. self.raw_y -= lf_E) then PES.dat will be modified to delta rather than HF_E
    