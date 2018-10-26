


class Model(object):
    """
    Abstract class for Machine Learning Models

    Subclasses which inherit from model: GaussianProcess

    Parameters
    ----------
    dataset : str 
        A path to a potential energy surface file, which is readable by
        numpy.loadtxt() or pandas.read_csv()

    ntrain : int
        The number of datapoints used to train.

    input_obj : object 
        InputProcessor object from MLChem. Used for keywords related to machine learning.
    """
    def __init__(self, dataset_path, ntrain, input_obj):
        try:
            data = np.loadtxt(dataset_path, delimiter=',', skiprows=1
        except:   
            raise Exception("Could not read dataset. Check to be sure the path is correct, 
                                and it is a csv with the first line being column labels")
        
        self.dataset = data 
        self.ntrain = ntrain
        self.input_obj = input_obj
        self.hyperparameter_optimization = input_obj.keywords['hp_opt']
        self.hp_max_evals = input_obj.keywords['hp_max_evals']


    def evaluate(self):
        pass
