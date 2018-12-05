from abc import ABC, abstractmethod
from .data_sampler import DataSampler 
import pandas as pd

class Model(ABC):
    """
    Abstract class for Machine Learning Models

    Subclasses which inherit from Model: 
    - GaussianProcess

    Parameters
    ----------
    dataset_path : str 
        A path to a potential energy surface file, which is readable as a
        pandas DataFrame by pandas.read_csv()

    input_obj : MLChem object 
        InputProcessor object from MLChem. Used for keywords related to machine learning.

    mol_obj : MLChem object 
        Molecule object from MLChem. Used for molecule information if permutation invariant geometry basis is used.
    """
    def __init__(self, dataset_path, input_obj, mol_obj=None):
        # get PES data. #TODO relax formatting requirements, make more general
        try:
            data = pd.read_csv(dataset_path)
        except:   
            raise Exception("Could not read dataset. Check to be sure the path is correct,and it is a csv with the first line being column labels.")

        # data
        self.dataset = data.sort_values("E")
        self.n_datapoints = self.dataset.shape[0]
        self.raw_X = self.dataset.values[:, :-1]
        self.raw_y = self.dataset.values[:,-1].reshape(-1,1)
        # settings, molecule information
        self.input_obj = input_obj
        self.mol = mol_obj
        self.pip = False
        if self.input_obj.keywords['use_pips'] == 'true':
            if self.mol:
                self.pip = True
            else:
                raise Exception("The use of permutation invariant polynomials ('use_pips' = true) requires Model objects are instantiated with a Molecule object: model = Model(dataset_path, input_obj, mol_obj)")
        else:
            print("Warning: Molecular geometry will not be transformed to permutation-invariant representation ('use_pips' = false). The model will therefore not generalize to symmetry-equivalent points. Ensure that the dataset is properly built to compensate for this.")

        # keyword control
        self.ntrain = self.input_obj.keywords['training_points']
        if self.ntrain >= self.dataset.shape[0]:
            raise Exception("Requested number of training points is greater than size of the dataset.")
        self.hp_max_evals = self.input_obj.keywords['hp_max_evals']
        self.sampler = self.input_obj.keywords['sampling']
        # for input, output style
        self.do_hp_opt = self.input_obj.keywords['hp_opt']
        # more keywords...

        # train test split
        if self.input_obj.keywords['n_low_energy_train']:
            n =  self.input_obj.keywords['n_low_energy_train']
            sample = DataSampler(self.dataset, self.ntrain, accept_first_n=n)
        else:
            sample = DataSampler(self.dataset, self.ntrain)
        if self.sampler == 'random':
            sample.random()
        elif self.sampler == 'smart_random':
            sample.smart_random()
        elif self.sampler == 'structure_based':
            sample.structure_based()
        elif self.sampler == 'sobol':
            sample.sobol()
        self.train_indices, self.test_indices = sample.get_indices()

        super().__init__()

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def compute_error(self):
        pass
