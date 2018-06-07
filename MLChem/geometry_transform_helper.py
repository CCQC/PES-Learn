"""
Various functions for molecular geometry transformations
"""
import math
import numpy as np
from itertools import combinations

def unit_vector(coords1, coords2):
    """
    Calculate the unit vector between two cartesian coordinates
    """
    distance = np.linalg.norm(coords2 - coords1)
    unit_vec = [0.0 for p in range(3)]
    for p in range(3):
        unit_vec[p] = (coords2[p] - coords1[p]) / distance 
    return unit_vec

def unit_cross_product(uvec1, uvec2):
    """
    Returns unit cross product between two unit vectors
    Ensures the result is itself a unit vector
    """
    cos = np.dot(uvec1, uvec2)
    sin = math.sqrt(1 - cos**2)
    # if the number of atoms is > 3 and there are 3 colinear atoms this will fail
    csc = sin**-1
    return np.cross(uvec1, uvec2) * csc


def get_local_axes(coords1, coords2, coords3):
    u12 = unit_vector(coords1, coords2)
    u23 = unit_vector(coords2, coords3)
    if (abs(np.dot(u12, u23)) >= 1.0):
      print('\nError: Co-linear atoms in an internal coordinate definition')
    u23_x_u12 = unit_cross_product(u23, u12)
    u12_x_u23_x_u12 = unit_cross_product(u12, u23_x_u12)
    z = u12
    y = u12_x_u23_x_u12
    x = unit_cross_product(y, z)
    local_axes = np.array([x, y, z])
    return local_axes

# calculate vector of bond in local axes of internal coordinates
def get_bond_vector(r, a, d):
    x = r * math.sin(a) * math.sin(d)
    y = r * math.sin(a) * math.cos(d)
    z = r * math.cos(a)
    bond_vector = np.array([x, y, z])
    return bond_vector

def get_interatom_distances(cart):
    matrix = np.zeros_like(cart)
    for i,j in combinations(range(len(cart)),2):
        R = np.linalg.norm(cart[i]-cart[j])
        #create lower triangle matrix
        matrix[j,i] = R
    return matrix
