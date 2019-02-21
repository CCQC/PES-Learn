from .data_sampler import DataSampler 
from . import geometry_transform_helper as gth
from .constants import hartree2cm, package_directory 
from .regex import xyz_block_regex
from abc import ABC, abstractmethod
import re
import pandas as pd
import warnings
import numpy as np
import sklearn.metrics
# GPy and sklearn output a bunch of annoying warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class Model(ABC):
    """
    Abstract class for Machine Learning Models

    Subclasses which inherit from Model: 
    - GaussianProcess
    - NeuralNetwork

    Parameters
    ----------
    dataset_path : str 
        A path to a potential energy surface file, which is readable as a
        pandas DataFrame by pandas.read_csv()

    input_obj : peslearn object 
        InputProcessor object from peslearn. Used for keywords related to machine learning.

    mol_obj : peslearn object 
        Molecule object from peslearn. Used for molecule information if permutation-invariant geometry representation is used.
    """

    def __init__(self, dataset_path, input_obj, mol_obj=None, train_path=None, test_path=None):
        data = self.interpret_dataset(dataset_path)
        if train_path:
            self.traindata = self.interpret_dataset(train_path)
            self.raw_Xtr = self.traindata.values[:, :-1]
            self.raw_ytr = self.traindata.values[:,-1].reshape(-1,1)
            if test_path:
                self.testdata = self.interpret_dataset(test_path)
                self.raw_Xtest = self.testdata.values[:, :-1]
                self.raw_ytest = self.testdata.values[:,-1].reshape(-1,1)

        self.dataset = data.sort_values("E")
        self.n_datapoints = self.dataset.shape[0]
        self.raw_X = self.dataset.values[:, :-1]
        self.raw_y = self.dataset.values[:,-1].reshape(-1,1)
        self.input_obj = input_obj
        self.mol = mol_obj
        self.pip = False
        if (self.input_obj.keywords['pes_format'] == 'interatomics') and (self.input_obj.keywords['use_pips'] == 'true'):
            if self.mol:
                self.pip = True
                print("Using permutation invariant polynomial transformation for molecule type ", self.mol.molecule_type)
            else:
                raise Exception(
                "The use of permutation invariant polynomials ('use_pips' = true) requires Model objects are",
                "instantiated with a Molecule object: model = Model(dataset_path, input_obj, mol_obj)")
        else:
            print(
            "Warning: Molecular geometry will not be transformed to permutation-invariant representation", 
            "(either pes_format is not 'interatomics' or 'use_pips' = false). The model will therefore not generalize", 
            "to symmetry-equivalent points. Ensure that the dataset is properly built to compensate for this.")

        # keyword control
        self.ntrain = self.input_obj.keywords['training_points']
        if train_path:
            self.ntrain = self.traindata.shape[0]
        if self.ntrain > self.dataset.shape[0]:
            raise Exception("Requested number of training points is greater than size of the dataset.")
        self.hp_max_evals = self.input_obj.keywords['hp_max_evals']

        if (train_path==None and test_path==None):
            self.sampler = self.input_obj.keywords['sampling']
        else:
            self.sampler = 'user_supplied'

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
        elif self.sampler == 'energy_ordered':
            sample.energy_ordered()
        elif self.sampler == 'user_supplied':
            pass
        else:
            raise Exception("Specified sampling method '{}' is not a valid option.".format(input_obj.keywords['sampling']))
        self.train_indices, self.test_indices = sample.get_indices()
        super().__init__()


    def interpret_dataset(self, path):
        with open(path) as f:
            read = f.read()
        if re.findall(xyz_block_regex, read):
            data = gth.load_cartesian_dataset(path)
        else:
            try:
                data = pd.read_csv(path)
            except:   
                raise Exception("Could not read dataset. Check to be sure the path is correct, and it is properly",
                                "formatted. Can either be 1. A csv-style file with the first line being a list of",
                                "arbitrary geometry labels with last column labeled 'E', e.g.  r1,r2,r3,...,E or 2.",
                                "A single energy value on its own line followed by a standard cartesian coordinate block.")
        return data


    @abstractmethod
    def build_model(self):
        pass
    @abstractmethod
    def save_model(self):
        pass

    def compute_error(self, X, y, prediction, yscaler, max_errors=None):
        """
        Predict the root-mean-square error (in wavenumbers) of model given 
        known X,y, a prediction, and a y scaling object, if it exists.
        
        Parameters
        ----------
        X : array
            Array of model inputs (geometries)
        y : array
            Array of expected model outputs (energies)
        prediction: array
            Array of actual model outputs (energies)
        yscaler: object
            Sci-kit learn scaler object
        max_errors: int
            Returns largest (int) absolute maximum errors 

        Returns
        -------
        error : float
            Root mean square error in wavenumbers (cm-1)
        """
        # TODO prevent incorrectly shaped y from being inputted otherwise huge memory/cost requirements
        if yscaler:
            raw_y = yscaler.inverse_transform(y)
            unscaled_prediction = yscaler.inverse_transform(prediction)
            error = np.sqrt(sklearn.metrics.mean_squared_error(raw_y,  unscaled_prediction))
            if max_errors:
                e = np.abs(raw_y - unscaled_prediction) * hartree2cm
                largest_errors = np.partition(e, -max_errors, axis=0)[-max_errors:]
        else:
            error = np.sqrt(sklearn.metrics.mean_squared_error(y, prediction))
            if max_errors:
                e = np.abs(y - prediction) * hartree2cm
                largest_errors = np.partition(e, -max_errors, axis=0)[-max_errors:]
        if max_errors:
            return error, largest_errors
        else:
            return error

