import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import re


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
        e.g. HCOOH would be ordered as HHOOC
    fi_path : str
        Path to Singular outputfile containing Fundamental Invariants
    """
    nbonds = raw_X.shape[1]
    with open(fi_path, 'r') as f:
        data = f.read()
        data = re.sub('\^', '**', data)
        #  convert subscripts of bonds to 0 indexing
        for i in range(1, nbonds+1):
            data = re.sub('x{}(\D)'.format(str(i)), 'x{}\\1'.format(i-1), data)

        polys = re.findall("\]=(.+)",data)

    # create a new_X matrix that is the shape of number geoms, number of Fundamental Invariants
    new_X = np.zeros((raw_X.shape[0],len(polys)))
    for i, p in enumerate(polys):    # evaluate each FI 
        # convert the FI to a python expression of raw_X, e.g. x1 + x2 becomes raw_X[:,1] + raw_X[:,2]
        eval_string = re.sub(r"(x)(\d+)", r"raw_X[:,\2]", p)
        # evaluate that column's FI from columns of raw_X
        new_X[:,i] = eval(eval_string)

    # find degree of each FI
    degrees = []
    for p in polys:
        # just checking first, assumes every term in each FI polynomial has the same degree (seems to always be true)
        tmp = p.split('+')[0]
        # count number of exponents and number of occurances of character 'x'
        exps = [int(i) - 1 for i in re.findall("\*\*(\d+)", tmp)]
        ndegrees = len(re.findall("x", tmp)) + sum(exps)
        degrees.append(ndegrees)

    return new_X, degrees

def degree_reduce(raw_X, degrees):
    """
    Take every fundamental invariant f and raise to f^(1/m) where m is degree of f
    """
    for i, degree in enumerate(degrees):
        raw_X[:,i] = np.power(raw_X[:,i], 1/degree)
    return raw_X

def sort_architectures(layers, inp_dim):
    """
    Takes a list of hidden layer tuples (n,n,n...) and input dimension size and
    sorts it by the number of expected weights in the neural network
    """
    out_dim = 1
    sizes = []
    for struct in layers:
        size = 0
        idx = 0
        size += inp_dim * struct[idx]
        idx += 1
        n = len(struct)
        while idx < n:
            size += struct[idx - 1] * struct[idx]
            idx += 1
        size += out_dim * struct[-1]
        sizes.append(size)
    sorted_indices = np.argsort(sizes).tolist()
    layers = np.asarray(layers)
    layers = layers[sorted_indices].tolist()
    return layers





