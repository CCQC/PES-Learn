from abc import ABC, abstractmethod
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

    ntrain : int
        The number of datapoints used to train.

    input_obj : object 
        InputProcessor object from MLChem. Used for keywords related to machine learning.
    """
    def __init__(self, dataset_path, ntrain, input_obj):
        # get PES data. #TODO relax formatting requirements, make more general
        try:
            data = pd.read_csv(dataset_path)
        except:   
            raise Exception("Could not read dataset. Check to be sure the path is correct,and it is a csv with the first line being column labels.")

        self.dataset = data.sort_values("E")
        self.ntrain = ntrain
        self.input_obj = input_obj
        self.hyperparameter_optimization = self.input_obj.keywords['hp_opt']
        self.hp_max_evals = self.input_obj.keywords['hp_max_evals']
        # more keywords...

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
