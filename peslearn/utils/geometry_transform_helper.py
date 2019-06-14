"""
Various functions for molecular geometry transformations
"""
import math
import numpy as np
import pandas as pd
import re
import os
from itertools import combinations
from .regex import xyz_block_regex,maybe
import collections

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
    #if (abs(np.dot(u12, u23)) >= 1.0):
      #print('\nError: Co-linear atoms in an internal coordinate definition')
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
    n = len(cart)
    matrix = np.zeros((n,n))
    for i,j in combinations(range(len(cart)),2):
        R = np.linalg.norm(cart[i]-cart[j])
        #create lower triangle matrix
        matrix[j,i] = R
    return matrix

def load_cartesian_dataset(xyz_path):
    """
    Loads a cartesian dataset with energies on their own line and with standard cartesian coordinates.
    Reorganizes atoms into standard order (most common elements first, alphabetical tiebreaker)
    """
    print("Loading Cartesian dataset: {}".format(xyz_path))
    xyz_re = xyz_block_regex
    with open(xyz_path) as f:
        data = ''
        # remove trailing whitespace
        for line in f:
            line = line.rstrip()
            data += line + '\n'
    # extract energy,geometry pairs
    #data_regex = "\s*-?\d+\.\d+\s*\n" + xyz_re
    #data_regex = maybe("\d\d?\n") + "\s*-?\d+\.\d+\s*\n" + xyz_re
    data_regex = maybe("\d+\n") + "\s*-?\d+\.\d+\s*\n" + xyz_re
    datablock = re.findall(data_regex, data)
    for i in range(len(datablock)):
        datablock[i] = list(filter(None, datablock[i].split('\n')))
    energies = [] 
    for datapoint in datablock:
        # check if atom numbers are used, energy line
        if datapoint[0].isdigit():
            a = datapoint.pop(0)
            e = datapoint.pop(0)
        else:
            e = datapoint.pop(0)
        energies.append(e)
    geoms = datablock
    # find atom labels
    sample = geoms[0]
    atom_labels = [re.findall('\w+', s)[0] for s in sample]
    natoms = len(atom_labels)
    # convert atom labels to standard order (most common element first, alphabetical tiebreaker)
    sorted_atom_counts = collections.Counter(atom_labels).most_common()
    sorted_atom_counts = sorted(sorted_atom_counts, key = lambda x: (-x[1], x[0]))
    sorted_atom_labels = []
    for tup in sorted_atom_counts:
        for i in range(tup[1]):
            sorted_atom_labels.append(tup[0])
    # find the permutation vector which maps unsorted atom labels to standard order atom labels
    p = []
    for i,j in enumerate(atom_labels):
        for k,l in enumerate(sorted_atom_labels):
            if j == l:
                p.append(k)
                sorted_atom_labels[k] = 'done'
                continue
    # permute all xyz geometries to standard order 
    for g in range(len(geoms)):
        geoms[g] = [geoms[g][i] for i in p]

    # write new xyz file with standard order
    #with open('std_' + xyz_path, 'w+') as f:
    #    for i in range(len(energies)):
    #        f.write(energies[i] +'\n')
    #        for j in range(natoms):
    #            f.write(geoms[i][j] +'\n')

    # remove everything from XYZs except floats and convert to numpy arrays
    for i,geom in enumerate(geoms):
        for j,string in enumerate(geom):
            string = string.split()
            del string[0] # remove atom label
            geom[j] = np.asarray(string, dtype=np.float64)
    
    # convert to interatomic distances
    final_geoms = []
    for i in geoms:
        idm = get_interatom_distances(i)
        idm = idm[np.tril_indices(idm.shape[0],-1)]
        final_geoms.append(idm)
    
    final_geoms = np.asarray(final_geoms)
    energies = np.asarray(energies, dtype=np.float64)
    n_interatomics =  int(0.5 * (natoms * natoms - natoms))
    bond_columns = []
    for i in range(n_interatomics):
        bond_columns.append("r%d" % (i))
    DF = pd.DataFrame(data=final_geoms, columns=bond_columns)
    DF['E'] = energies

    # remove suffix of xyz path if it exists
    finalpath = xyz_path.rsplit(".",1)[0]
    finalpath = os.path.splitext(xyz_path)[0]
    DF.to_csv(finalpath + '_interatomics.dat',index=False, float_format='%12.10f')
    return DF




