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
    def __init__(self, dataset, ntrain, rseed=42):
        # needs to be pandas dataframe 
        self.full_dataset = dataset
        self.ntrain = ntrain
        self.rseed = rseed
    
    def random(self):
        """
        docstring
        """
        X = data[:, :-1]
        y = data[:,-1].reshape(-1,1)
        X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size=self.ntrain, random_state=self.rseed)
        return X_train, X_test, y_train, y_test

    #def energy_ordered_uniform_random(self):
    #    """
    #    docstring
    #    """
    #    ordered_dataset = self.full_dataset.sort_values("E")
    #    data = ordered_dataset.values
    #    np.random.seed(0)
    #    indices = np.random.choice(data.shape[0], size=self.ntrain, replace=False)
    #    train = data[indices,:]


    def sobol(self, delta=0.002278):
        """
        docstring
        """
        # sort and subtract lowest energy from every energy 
        # TODO causes one datapoint to be E = 0.00... Problematic?
        data = self.full_dataset.sort_values("E")
        data['E'] = data['E'] - data['E'].min()

        max_e = data['E'].max()
        denom = (1 / (max_e + delta))
        train = []
        test = self.full_dataset.copy()
        
        while len(train) < self.ntrain:
            # randomly draw a PES datapoint 
            rand_point = data.sample(n=1)
            rand_E = rand_point['E'].values
            condition = (max_e - rand_E + delta) * denom
            rand = np.random.uniform(0.0,1.0)
            # if this datapoint is already accepted into training set, skip it
            if any((rand_point.values == x).all() for x in train):
                continue
            # add to training set if sobol condition is satisfied. Remove it from the test dataset as well.
            if condition > rand:
                tmp.append(rand_point.values)

        # convert back to pandas dataframe
        train = np.asarray(train).reshape(self.ntrain,len(self.full_dataset.columns))
        train = pd.DataFrame(train, columns=self.full_dataset.columns)
        train = train.sort_values("E")


    def structure_based(self):
        pass
