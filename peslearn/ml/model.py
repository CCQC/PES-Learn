from .data_sampler import DataSampler 
from ..constants import hartree2cm, package_directory 
from ..utils.regex import xyz_block_regex
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

    molecule_type : str
        Molecule type defining number of each atom in decreasing order. 
        AxByCz... where x,y,z are integers. E.g., H2O --> A2B,  C2H4 --> A4B2

    molecule : peslearn object 
        Molecule object from peslearn. Used to automatically define molecule_type
    """
    def __init__(self, dataset_path, input_obj, molecule_type=None, molecule=None, train_path=None, test_path=None, valid_path=None):
        self.hyperparameter_space = {}
        data = self.interpret_dataset(dataset_path)
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        if train_path:
            self.traindata = self.interpret_dataset(train_path)
            self.raw_Xtr = self.traindata.values[:, :-1]
            self.raw_ytr = self.traindata.values[:,-1].reshape(-1,1)
            if test_path:
                self.testdata = self.interpret_dataset(test_path)
                self.raw_Xtest = self.testdata.values[:, :-1]
                self.raw_ytest = self.testdata.values[:,-1].reshape(-1,1)
                if valid_path:
                    self.validdata = self.interpret_dataset(valid_path)
                    self.raw_Xvalid = self.validdata.values[:, :-1]
                    self.raw_yvalid = self.validdata.values[:,-1].reshape(-1,1)

        self.dataset = data.sort_values("E")
        self.n_datapoints = self.dataset.shape[0]
        self.raw_X = self.dataset.values[:, :-1]
        self.raw_y = self.dataset.values[:,-1].reshape(-1,1)
        self.input_obj = input_obj

        self.pip = False
        if molecule:
            self.molecule_type = molecule.molecule_type
            if self.input_obj.keywords['use_pips'] == 'true':
                self.pip = True
                print("Using permutation invariant polynomial transformation for molecule type ", self.molecule_type)
        if molecule_type:
            self.molecule_type = molecule_type
            if self.input_obj.keywords['use_pips'] == 'true':
                self.pip = True
                print("Using permutation invariant polynomial transformation for molecule type ", self.molecule_type)
            
        # keyword control
        self.ntrain = self.input_obj.keywords['training_points']
        if train_path:
            self.ntrain = self.traindata.shape[0]
        if self.ntrain > self.dataset.shape[0]:
            raise Exception("Requested number of training points is greater than size of the dataset.")
        self.hp_maxit = self.input_obj.keywords['hp_maxit']

        if (train_path==None and test_path==None):
            self.sampler = self.input_obj.keywords['sampling']
        else:
            self.sampler = 'user_supplied'

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
    @abstractmethod
    def preprocess(self):
        pass
    @abstractmethod
    def split_train_test(self):
        pass

    def get_hyperparameters(self):
        """
        Returns hyperparameters of this model
        """
        return self.hyperparameter_space

    def set_hyperparameter(self, key, val):
        """
        Set hyperparameter 'key' to value 'val'.
        Parameters
        ---------
        key : str
            A hyperparameter name
        val : obj
            A HyperOpt object such as hp.choice, hp.uniform, etc.
        """
        self.hyperparameter_space[key] = val

    def compute_error(self, known_y, prediction, yscaler=None, max_errors=None):
        """
        Predict the root-mean-square error (in wavenumbers) of model given 
        known X,y, a prediction, and a y scaling object, if it exists.
        
        Parameters
        ----------
        known_y : ndarray
            Array of expected model outputs (energies)
        prediction: ndarray
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
        if known_y.shape != prediction.shape:
            raise Exception("Shape of known_y and prediction must be the same")
        if yscaler:
            raw_y = yscaler.inverse_transform(known_y)
            unscaled_prediction = yscaler.inverse_transform(prediction)
            error = np.sqrt(sklearn.metrics.mean_squared_error(raw_y,  unscaled_prediction))
            if max_errors:
                e = np.abs(raw_y - unscaled_prediction) * hartree2cm
                median_error = np.median(e, axis=0)
                largest_errors = np.partition(e, -max_errors, axis=0)[-max_errors:]
        else:
            error = np.sqrt(sklearn.metrics.mean_squared_error(known_y, prediction))
            if max_errors:
                e = np.abs(known_y - prediction) * hartree2cm
                median_error = np.median(e, axis=0)
                largest_errors = np.partition(e, -max_errors, axis=0)[-max_errors:]
        if max_errors:
            return error, median_error, largest_errors
        else:
            return error

