"""
A class for sampling train and test sets from PES datasets 
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

class DataSampler(object):
    """
    docstring
    """
    def __init__(self, dataset, ntrain, accept_first_n=None, rseed=42):
        self.full_dataset = dataset.sort_values("E")
        if accept_first_n:
            if accept_first_n > ntrain:
                raise Exception("Number of forced low-energy training points exceeds the indicated total training set size")
            # remove first n points 
            self.dataset = self.full_dataset[accept_first_n:]
            self.ntrain = ntrain - accept_first_n
        else:
            self.ntrain = ntrain
            self.dataset = self.full_dataset
            
        self.dataset_size = self.dataset.shape[0]
        # currently needs to be pandas dataframe 
        #if "E" in dataset.columns:
        #    self.full_dataset = dataset.sort_values("E")
        #else:
        #    self.full_dataset = dataset
        self.rseed = rseed
        self.first_n = accept_first_n
        self.train_indices = None
        self.test_indices = None

    def set_indices(self, train_indices, test_indices):
        if self.first_n:
            # train/test indices were obtained relative to the dataset that had removed first n datapoints, adjust accordingly 
            train_indices += self.first_n 
            test_indices += self.first_n 
            self.train_indices, self.test_indices = self.include_first_n(train_indices, test_indices)
        else:
            self.train_indices = train_indices
            self.test_indices = test_indices

    def get_indices(self):
        return self.train_indices, self.test_indices
    
    def include_first_n(self, train_indices, test_indices):
        """
        Force first n lowest energy points to be in training set 
        Useful for global-minimum-biased fits for applications such as vibrational computations.
        """ 
        # force first n indices to be in training set
        a = np.arange(self.first_n) 
        tmp =  np.concatenate((train_indices, a) ,axis=0)
        train_indices = np.unique(tmp)  #  avoids double counting
        # adjust test set accordingly
        condition = test_indices > self.first_n
        test_indices = np.extract(condition, test_indices)
        return train_indices, test_indices
                
    def random(self):
        """
        Randomly sample the dataset to obtain a training set of proper size.
        """
        data = self.dataset.values
        X = data[:, :-1]
        y = data[:,-1].reshape(-1,1)
        indices = np.arange(self.dataset_size)
        train_indices, test_indices  = train_test_split(indices, train_size=self.ntrain, random_state=self.rseed)
        #if self.first_n:
        #    train_indices = self.include_first_n(train_indices)

        self.set_indices(train_indices, test_indices)

    def smart_random(self):
        """
        Choose a random training set that has an energy distribution most resembling that of the full dataset.
        Uses the Chi-Squared method to estimate the similarity of the energy distrubtions.
        """
        data = self.dataset.values
        X = data[:, :-1]
        y = data[:,-1].reshape(-1,1)
        full_dataset_dist, binedges = np.histogram(y, bins=10, density=True)
        pvalues = []
        chi = []
        for seed in range(500):
            X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size=self.ntrain, random_state=seed)
            train_dist, tmpbin = np.histogram(y_train, bins=binedges, density=True)
            chisq, p = stats.chisquare(train_dist, f_exp=full_dataset_dist)
            chi.append(chisq)
            pvalues.append(p)
        best_seed = np.argmin(chi)
        #best_seed = np.argmax(chi)
        X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size=self.ntrain, random_state=best_seed)
        train_dist, tmpbin = np.histogram(y_train, bins=binedges, density=True)

        indices = np.arange(self.dataset_size)
        train_indices, test_indices  = train_test_split(indices, train_size=self.ntrain, random_state=best_seed)
        self.set_indices(train_indices, test_indices)


    def energy_ordered(self):
        """
        A naive sampling algorithm, where we order the PES dataset
        in terms of increasing energy, and take every nth datapoint such that we 
        get approximately the right amount of training points.

        A dataset first needs to be sorted by energy before calling.
        Warning: Does not return exact number of desired training points. 
        """
        interval = round(self.dataset_size / self.ntrain)
        indices = np.arange(self.dataset_size)
        train_indices = indices[0::interval]
        test_indices = np.delete(indices, indices[0::interval])
        self.set_indices(train_indices, test_indices)

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
        # Problems:
        # 1. not easily reproducible with a random seed.
        # 2. Scaling. could in principle improve scaling by doing minibatches in while loop... e.g. test.sample(n=minibatch)
        data = self.dataset.sort_values("E")
        data['E'] = data['E'] - data['E'].min()
        
        max_e = data['E'].max()
        denom = (1 / (max_e + delta))
        train_indices = []
        indices = np.arange(data.shape[0])
        while len(train_indices) < self.ntrain:
            # randomly draw a PES datapoint 
            rand_point = data.sample(n=1)
            rand_E = rand_point['E'].values
            condition = (max_e - rand_E + delta) * denom
            rand = np.random.uniform(0.0,1.0)
            # if this datapoint is already accepted into training set, skip it
            # (Not needed, as long as there are not equivalent geometries)    
            #if any((rand_point.values == x).all() for x in train):           
            #    continue                                                     
            if condition > rand:
                train_indices.append(rand_point.index[0])
                data = data.drop(rand_point.index[0])
        test_indices = np.delete(indices, indices[train_indices])

        self.set_indices(train_indices, test_indices)


    def structure_based(self):
        """
        Sample the geometries according to their L2 norms from one another.
        Based on the algorithm described in Dral et al, J Chem Phys, 146, 244108, 2017
        and references therein. Please cite appropriately if used. 
        First the point closest to the global minimum is taken as the first training point.
        The second training point is that which is 'furthest' from the first.
        Each additional training point is added by  
        1. For each new training point candidate, compute shortest distance 
                to every point in training set.
        2. Find the training point candidate with the largest shortest distance to the training set
        3. Add this candidate to the training set, remove from the test set.
        4. Repeat 1-3 until desired number of points obtained.
        """
        data = self.dataset
        train = []
        train.append(data.values[0])

        def norm(train_point, data=data):
            """ Computes norm between training point geometry and every point in dataset"""
            tmp1 = np.tile(train_point[:-1], (data.shape[0],1))
            diff = tmp1 - data.values[:,:-1]
            norm_vector = np.sqrt(np.einsum('ij,ij->i', diff, diff))
            return norm_vector

        # accept farthest point from 1st training point as the 2nd training point
        norm_vector_1 = norm(train[0])
        idx = np.argmax(norm_vector_1)
        newtrain = data.values[idx]
        train.append(newtrain)

        # create norm matrix, whose rows are all the norms to 1st and 2nd training points 
        norm_vector_2 = norm(train[1])
        norm_matrix = np.vstack((norm_vector_1, norm_vector_2))

        # find the minimum value along the columns of this 2xN array of norms
        min_array = np.amin(norm_matrix, axis=0)
        train_indices = []
        train_indices.append(0)
        train_indices.append(idx)

        while len(train) < self.ntrain:
            # min_array contains the smallest norms into the training set, by datapoint.
            # We take the largest one.
            idx = np.argmax(min_array)
            train_indices.append(idx)
            new_geom = data.values[idx]
            train.append(new_geom)
            # update norm matrix with the norms of newly added training point
            norm_vec = norm(train[-1])
            stack = np.vstack((min_array, norm_vec))
            min_array = np.amin(stack, axis=0)

        indices = np.arange(self.dataset_size)
        test_indices = np.delete(indices, indices[train_indices])
        train_indices = np.asarray(train_indices)
        # do not sort. This ruins the building-up method of the PES
        #train_indices = np.sort(train_indices)

        self.set_indices(train_indices, test_indices)

    def energy_gaussian(self):
        """
        Heavily biases towards low energy region
        """
        pass
