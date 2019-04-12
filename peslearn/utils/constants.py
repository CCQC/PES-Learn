import numpy as np
import os

# some constants used in the code
rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0

bohr2angstroms = 0.52917720859
hartree2ev =  27.21138505
hartree2cm =  219474.63 

#MLChem package absolute path
package_directory = os.path.dirname(os.path.abspath(__file__)) + "/../"

# Gaussian process convenience function writer 
gp_convenience_function = """
# How to use 'compute_energy()' function
# --------------------------------------
# E = compute_energy(geom_vectors, cartesian=bool)
# 'geom_vectors' is a column vector of one or more sets of 1d coordinate vectors as a list of lists or 2D NumPy array.
# [[ coord1, coord2, ..., coordn],
#  [ coord1, coord2, ..., coordn],
#                             ...,
#  [ coord1, coord2, ..., coordn]]
# In cases, coordinates should be supplied in the exact same format and exact same order the model was trained on.
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

# The returned energy array is a column vector of corresponding energies.

def compute_energy(geom_vectors, cartesian=True):
    g = np.asarray(geom_vectors)
    if cartesian:
        g = np.apply_along_axis(cart1d_to_distances1d, 1, g)
    newX = gp.transform_new_X(g, params, Xscaler)
    E, cov = final.predict(newX, full_cov=False)
    E = gp.inverse_transform_new_y(E,yscaler)
    return E

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
