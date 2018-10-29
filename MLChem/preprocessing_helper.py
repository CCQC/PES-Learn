import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def general_scaler(scale_type, data):
    """
    Scales each column of an array according to scale_type
    """
    if scale_type == 'std':
        scaler = StandardScaler()
        new_data = scaler.fit_transform(data)
    if scale_type == 'mm01':
        scaler = MinMaxScaler(feature_range=(0,1))
        new_data = scaler.fit_transform(data)
    if scale_type == 'mm11':
        scaler = MinMaxScaler(feature_range=(-1,1))
        new_data = scaler.fit_transform(data)
    return new_data, scaler

def morse(raw_X, alpha=1.00):
    """
    Element-wise morse variable transformation on an array of interatom distances
    r_morse = exp(-r/alpha)
    Assumes units of Angstroms 
    """
    return np.exp(-raw_X / alpha)

def interatomics_to_fundinvar(raw_X, fi_path):
    """
    Transfrom interatom distances to fundamental invariants 
    Parameters
    ---------
    raw_X : array 
        Array of interatomic distances in Standard Order: 
        r1
        r2 r3
        r4 r5 r6...
        where the order of atoms along columns/rows of interatomic distance matrix  
        is determined by highest frequency atoms first, alphabetical tiebreaker) 
        e.g. HCOOH --> HHOOC
    fi_path : str
        Path to Singular outputfile containing Fundamental Invariants
    """
    #TODO
    pass

def degree_reduce(raw_X, degrees):
    """
    Take every fundamental invariant f and raise to f^(1/m) where m is degree of f
    """
    for i, degree in enumerate(degrees):
        raw_X[:,i] = np.power(raw_X[:,i], 1/degree)
    return raw_X


