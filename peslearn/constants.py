import numpy as np
import os

# some constants used in the code
rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0

bohr2angstroms = 0.52917720859
hartree2ev =  27.21138505
hartree2cm =  219474.63 

#PES-Learn package absolute path (used for accessing FI library)
package_directory = os.path.dirname(os.path.abspath(__file__)) 

# Gaussian process convenience function writer 
gp_convenience_function = """
# How to use 'pes()' function
# --------------------------------------
# E = pes(geom_vectors, cartesian=bool)
# 'geom_vectors' is either: 
#  1. A list or tuple of coordinates for a single geometry. 
#  2. A column vector of one or more sets of 1d coordinate vectors as a list of lists or 2D NumPy array:
# [[ coord1, coord2, ..., coordn],
#  [ coord1, coord2, ..., coordn],
#      :       :             :  ], 
#  [ coord1, coord2, ..., coordn]]
# In all cases, coordinates should be supplied in the exact same format and exact same order the model was trained on.
# If the coordinates format used to train the model was interatomic distances, each set of coordinates should be a 1d array of either interatom distances or cartesian coordinates. 
# If cartesian coordinates are supplied, cartesian=True should be passed and it will convert them to interatomic distances. 
# The order of coordinates matters. If PES-Learn datasets were used they should be in standard order;
# i.e. cartesians should be supplied in the order x,y,z of most common atoms first, with alphabetical tiebreaker. 
# e.g., C2H3O2 --> H1x H1y H1z H2x H2y H2z H3x H3y H3z C1x C1y C1z C2x C2y C2z O1x O1y O1z O2x O2y O2z
# and interatom distances should be the row-wise order of the lower triangle of the interatom distance matrix, with standard order atom axes:
#    H  H  H  C  C  O  O 
# H 
# H  1
# H  2  3
# C  4  5  6 
# C  7  8  9  10 
# O  11 12 13 14 15
# O  16 17 18 19 20 21

# The returned energy array is a column vector of corresponding energies. Elements can be accessed with E[0,0], E[0,1], E[0,2]
# NOTE: Sending multiple geometries through at once is much faster than a loop of sending single geometries through.

def pes(geom_vectors, cartesian=True):
    g = np.asarray(geom_vectors)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = gp.transform_new_X(g, params, Xscaler)
    E, cov = final.predict(newX, full_cov=False)
    e = gp.inverse_transform_new_y(E,yscaler)
    #e = e - (insert min energy here)
    #e *= 219474.63  ( convert units )
    return e

def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1,3)
    n = len(vec)
    distance_matrix = np.zeros((n,n))
    for i,j in combinations(range(len(vec)),2):
        R = np.linalg.norm(vec[i]-vec[j])
        distance_matrix[j,i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix),-1)]
    return distance_vector
"""    


nn_convenience_function = """
# How to use 'pes()' function
# --------------------------------------
# E = pes(geom_vectors, cartesian=bool)
# 'geom_vectors' is either: 
#  1. A list or tuple of coordinates for a single geometry. 
#  2. A column vector of one or more sets of 1d coordinate vectors as a list of lists or 2D NumPy array:
# [[ coord1, coord2, ..., coordn],
#  [ coord1, coord2, ..., coordn],
#      :       :             :  ], 
#  [ coord1, coord2, ..., coordn]]
# In all cases, coordinates should be supplied in the exact same format and exact same order the model was trained on.
# If the coordinates format used to train the model was interatomic distances, each set of coordinates should be a 1d array of either interatom distances or cartesian coordinates. 
# If cartesian coordinates are supplied, cartesian=True should be passed and it will convert them to interatomic distances. 
# The order of coordinates matters. If PES-Learn datasets were used they should be in standard order;
# i.e. cartesians should be supplied in the order x,y,z of most common atoms first, with alphabetical tiebreaker. 
# e.g., C2H3O2 --> H1x H1y H1z H2x H2y H2z H3x H3y H3z C1x C1y C1z C2x C2y C2z O1x O1y O1z O2x O2y O2z
# and interatom distances should be the row-wise order of the lower triangle of the interatom distance matrix, with standard order atom axes:
#    H  H  H  C  C  O  O 
# H 
# H  1
# H  2  3
# C  4  5  6 
# C  7  8  9  10 
# O  11 12 13 14 15
# O  16 17 18 19 20 21

# The returned energy array is a column vector of corresponding energies. Elements can be accessed with E[0,0], E[0,1], E[0,2]
# NOTE: Sending multiple geometries through at once is much faster than a loop of sending single geometries through.

def pes(geom_vectors, cartesian=True):
    g = np.asarray(geom_vectors)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = nn.transform_new_X(g, params, Xscaler)
    x = torch.tensor(data=newX)
    with torch.no_grad():
        E = model(x)
    e = nn.inverse_transform_new_y(E, yscaler)
    #e = e - (insert min energy here)
    #e *= 219474.63  ( convert units )
    return e

def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1,3)
    n = len(vec)
    distance_matrix = np.zeros((n,n))
    for i,j in combinations(range(len(vec)),2):
        R = np.linalg.norm(vec[i]-vec[j])
        distance_matrix[j,i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix),-1)]
    return distance_vector
"""    

krr_convenience_funciton = """
# How to use 'pes()' function
# --------------------------------------
# E = pes(geom_vectors, cartesian=bool)
# 'geom_vectors' is either: 
#  1. A list or tuple of coordinates for a single geometry. 
#  2. A column vector of one or more sets of 1d coordinate vectors as a list of lists or 2D NumPy array:
# [[ coord1, coord2, ..., coordn],
#  [ coord1, coord2, ..., coordn],
#      :       :             :  ], 
#  [ coord1, coord2, ..., coordn]]
# In all cases, coordinates should be supplied in the exact same format and exact same order the model was trained on.
# If the coordinates format used to train the model was interatomic distances, each set of coordinates should be a 1d array of either interatom distances or cartesian coordinates. 
# If cartesian coordinates are supplied, cartesian=True should be passed and it will convert them to interatomic distances. 
# The order of coordinates matters. If PES-Learn datasets were used they should be in standard order;
# i.e. cartesians should be supplied in the order x,y,z of most common atoms first, with alphabetical tiebreaker. 
# e.g., C2H3O2 --> H1x H1y H1z H2x H2y H2z H3x H3y H3z C1x C1y C1z C2x C2y C2z O1x O1y O1z O2x O2y O2z
# and interatom distances should be the row-wise order of the lower triangle of the interatom distance matrix, with standard order atom axes:
#    H  H  H  C  C  O  O 
# H 
# H  1
# H  2  3
# C  4  5  6 
# C  7  8  9  10 
# O  11 12 13 14 15
# O  16 17 18 19 20 21

# The returned energy array is a column vector of corresponding energies. Elements can be accessed with E[0,0], E[0,1], E[0,2]
# NOTE: Sending multiple geometries through at once is much faster than a loop of sending single geometries through.

def pes(geom_vectors, cartesian=True):
    g = np.asarray(geom_vectors)
    if cartesian:
        axis = 1
        if len(g.shape) < 2:
            axis = 0
        g = np.apply_along_axis(cart1d_to_distances1d, axis, g)
    newX = krr.transform_new_X(g, params, Xscaler)
    E = model.predict(newX)
    e = krr.inverse_transform_new_y(E, yscaler)
    #e = e - (insert min energy here)
    #e *= 219474.63  ( convert units )
    return e

def cart1d_to_distances1d(vec):
    vec = vec.reshape(-1,3)
    n = len(vec)
    distance_matrix = np.zeros((n,n))
    for i,j in combinations(range(len(vec)),2):
        R = np.linalg.norm(vec[i]-vec[j])
        distance_matrix[j,i] = R
    distance_vector = distance_matrix[np.tril_indices(len(distance_matrix),-1)]
    return distance_vector
"""

gradient_nn_convenience_function = """
# how to use 'gradient_compute()' function
# --------------------------------------
# grads = gradient_compute(cartesian_dataset_path)
# 'cartesian_dataset_path' is a path to a dataset with cartesian coordinates that will be used with model to predict gradients.
# Cartesian geometries should contain atomic symbol (e.g. H, C, Br, etc.) followed by XYZ cartesian coordinates each separated by spaces.
# The dataset may contain one or multiple geometries, if given multiple geometries it will return a list of multiple gradients
# The output of the 'gradient_compute()' function will be the negative derivative of the predicted energy from the model with respect to
# the cartesian coordinate of the provided geometry. 
# i.e. ouput = -dE/dq, where output is the returned value from the 'gradient_compute()' function, E is the model predicted energy, and q are the cartesian coordinates
# Input coordinates need not be in standard order, they will however be transformed into standard order for the output. 
# Standard order in PES-Learn lists most common atoms first with alphabetical tiebreakers. 
# e.g. If the input provided for water lists the XYZ coordinates in order of O H1 H2, then the output gradient will be in the order H1 H2 O.
# The returned gradients will be in units of Hartree/distance where distance is either Bohr or Angstrom. 
# The distance unit used to construct the model should be the same distance unit used to predict gradients.
# Outputs are of the form of a list of torch tensors, where each tensor is a predicted gradient in the order provided in the catesian dataset.
# Using this function for the first time creates a new file containing a function tailored to the given input molecule.

def gradient_compute(cart_dataset_path):
    grad = []
    # transform 
    sorted_atoms, geoms = geometry_transform_helper.load_cartesian_dataset(cart_dataset_path, no_energy=True)
    for i in range(len(geoms)):
        geoms[i] = geometry_transform_helper.remove_atom_labels(geoms[i])
    if not os.path.exists('grad_func.py'):
        geometry_transform_helper.write_grad_input(sorted_atoms)
    from grad_func import gradient_prediction
    for g in range(len(geoms)):
        grad.append(gradient_prediction(nn, params, model, Xscaler, yscaler, geoms[g]))
    return grad
"""






