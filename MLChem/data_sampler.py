"""
A class for sampling training and testing sets from datasets 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSampler(object):
    """
    docstring
    """
    def __init__(self, dataset, ntrain):
        # needs to be pandas dataframe 
        self.full_dataset = dataset
        self.ntrain = ntrain
    
    def random(self, rseed=42):
        """
        """
        X = data[:, :-1]
        y = data[:,-1].reshape(-1,1)
        X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size=self.ntrain, random_state=rseed)

    def energy_ordered_uniform_random(self, rseed=42):
        """
        """
        ordered_dataset = self.full_dataset.sort_values("E")
        data = ordered_dataset.values
        np.random.seed(0)
        indices = np.random.choice(data.shape[0], size=self.ntrain, replace=False)
        train = data[indices,:]
