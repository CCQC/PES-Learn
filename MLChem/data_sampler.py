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
        self.dataset_size = dataset.shape[0]
        self.ntrain = ntrain
        self.rseed = rseed
    
    def random(self):
        """
        docstring
        """
        data = self.full_dataset.values
        X = data[:, :-1]
        y = data[:,-1].reshape(-1,1)
        X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size=self.ntrain, random_state=self.rseed)
        return X_train, X_test, y_train, y_test

    def energy_ordered(self):
        """
        A naive sampling algorithm, where we order the PES dataset
        in terms of increasing energy, and take every nth datapoint such that we 
        get approximately the right amount of training points.
        """
        # TODO
        # Problem: Does not return desired number of training points. Is there a general way to do this in a reproducible manner?
        ordered_dataset = self.full_dataset.sort_values("E")
        s = round(self.dataset_size / self.ntrain)
        train = ordered_dataset[0::s]
        # to create test set, set training set elements equal to None and remove
        ordered_dataset[0::s] = None
        test = ordered_dataset.dropna()

        train_data = train.values
        X_train = train_data[:, :-1]
        y_train = train_data[:,-1].reshape(-1,1)
        test_data = test.values
        X_test = test_data[:, :-1]
        y_test = test_data[:,-1].reshape(-1,1)
        return X_train, X_test, y_train, y_test
        


    def sobol(self, delta=0.002278):
        """
        A quasi-random sampling of the PES based on the relative energies.
        First, the PES data is ordered in terms of increasing energy, 
        and each energy is shifted by the lowest energy in the dataset so the energy range becomes [0.00, max_E - min_E].
        In each iteration, we draw a random number between 0 and 1 and a random datapoint from the PES.
        We then compare the magnitude of the random number to the expression of the energy:  
        (V_max - V + delta) / (V_max + delta) > random_number
        where V is the energy of the random datapoint, V_max is the maximum energy of the dataset, 
        and delta is a shift factor  (default is 0.002278 Hartrees, 500 cm-1).
        We accept the random datapoint to be a training point if the above condition is satisfied.
        The result is a quasi random series of training points whose distribution DOES NOT follow
        the distribution of the full dataset. Instead, it is biased towards low to mid range energies. 
        This is appropriate for accurately modeling a minimum for vibrational applications, for example.

        The Sobol expression is as implemented in Manzhos, Carrington J Chem Phys 145, 2016, and papers they cite.
        """
        # TODO 
        # Problems:
        # 1. causes one datapoint to be E = 0.00
        # 2. not reproducible with a random seed.
        # 3. could in principle improve scaling by doing minibatches in while loop... e.g. test.sample(n=minibatch)
        data = self.full_dataset.sort_values("E")
        data['E'] = data['E'] - data['E'].min()

        max_e = data['E'].max()
        denom = (1 / (max_e + delta))
        train = []
        test = self.full_dataset.copy()
        
        while len(train) < self.ntrain:
            # randomly draw a PES datapoint 
            rand_point = test.sample(n=1)
            rand_E = rand_point['E'].values
            condition = (max_e - rand_E + delta) * denom
            rand = np.random.uniform(0.0,1.0)
            # if this datapoint is already accepted into training set, skip it
            # (Not needed, as long as there are not equivalent geometries)
            #if any((rand_point.values == x).all() for x in train):
            #    continue
            # add to training set if sobol condition is satisfied. Remove it from the test dataset as well.
            if condition > rand:
                train.append(rand_point.values)
                test = test.drop(rand_point.index[0])

        # convert back to pandas dataframe
        train = np.asarray(train).reshape(self.ntrain,len(self.full_dataset.columns))
        train = pd.DataFrame(train, columns=self.full_dataset.columns)
        train = train.sort_values("E")


    def structure_based(self):
        pass

    def energy_gaussian(self):
        pass
