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
from ..constants import deg2rad, rad2deg
import collections

def get_interatom_distances(cart):
    n = len(cart)
    matrix = np.zeros((n,n))
    for i,j in combinations(range(len(cart)),2):
        R = np.linalg.norm(cart[i]-cart[j])
        #create lower triangle matrix
        matrix[j,i] = R
    return matrix

def vectorized_unit_vector(coord_pairs):
    """
    Finds all unit vectors between a series of coordinate pairs
    """
    # First split arrays along pairs of atom coordinates between which unit vectors will be computed
    split = np.split(coord_pairs, 2, 1)
    a, b = np.squeeze(split[0]), np.squeeze(split[1])
    # compute unit vectors between points 
    #einsum may be faster than linalg?? https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
    tmp = b - a
    #norms = np.linalg.norm(tmp, axis=1).reshape(-1,1)
    norms = np.sqrt(np.einsum('ij,ij->i', tmp, tmp)).reshape(-1,1)
    unit_vecs = tmp[:] / norms
    return unit_vecs

def vectorized_unit_cross_product(uvec1, uvec2):
    """
    Returns all cross products between every pair of unit vectors in uvec1 and uvec2
    """
    #products = np.cross(uvec1, uvec2)
    products = np.cross(np.round(uvec1,12), np.round(uvec2,12))
    # If cross product is zero, it is due to co-linear atoms
    #print(np.all(np.isclose(products, np.zeros_like(products)), axis=1))
    #colinear_atoms_bool = np.all(np.isclose(products, np.zeros_like(products)), axis=1)
    #if np.any(np.all(np.isclose(products, np.zeros_like(products)), axis=1)):
    #    print('co-linear atoms detected')
    #norms = np.linalg.norm(products, axis=1).reshape(-1,1)
    norms = np.sqrt(np.einsum('ij,ij->i', products, products)).reshape(-1,1)
    unit_vecs = products[:] / norms
    return unit_vecs

def vectorized_bond_vector(coords):
    """
    Compute vector of bond in local axes of internal coordinates for all internal coordinates.
    coords is an array of bond, angle, dihedral values for a particular atom
    """
    x = coords[:,0] * np.sin(coords[:,1]) * np.sin(coords[:,2])
    y = coords[:,0] * np.sin(coords[:,1]) * np.cos(coords[:,2])
    z = coords[:,0] * np.cos(coords[:,1])
    return np.array([x,y,z]).T

def vectorized_local_axes(three_atoms_coords):
    """
    Takes as an argument a Nx3x3 block of reference atom coordinates to construct N local axes systems (Nx3x3)
    """
    u12 = vectorized_unit_vector(three_atoms_coords[:, [0,1], :])
    u23 = vectorized_unit_vector(three_atoms_coords[:, [1,2], :])
    if np.any(np.einsum('ij,ij->i', u12,u23)) > 1.0:
        print("co-linear atoms detected")
    u23_x_u12 = vectorized_unit_cross_product(u23, u12)
    u12_x_u23_x_u12 = vectorized_unit_cross_product(u12, u23_x_u12)
    z = u12
    y = u12_x_u23_x_u12
    x = vectorized_unit_cross_product(y, z)
    local_axes = np.transpose(np.array([x, y, z]), (1,0,2))
    return local_axes

def vectorized_zmat2xyz(intcos, zmat_indices, permutation_vector, natoms):
    """
    Takes array of internal coordinates, creates 3d cartesian coordinate block of all Cartesian coordinates

    Parameters
    ---------
    intcos : arr
        A NumPy array of shape (n_geoms, n_internal_coords) containing a series of internal coordinate definitions
    zmat_indices : arr
        A NumPy array of shape (n_internal_coords) containing a series of ZMAT connectivity indices (NOT zero-indexed.)
    permutation_vector : arr
        A NumPy array of shape (n_atoms) describing how to permute atom order to standard order 
    natoms : int
        The number of atoms (including dummy atoms)

    Returns 
    ---------
    cart : arr
        A NumPy array of all Cartesian coordinates corresponding to internal coordinates. 
        Has shape (n_geoms, n_atoms, 3), i.e., it is a list of 2d Cartesian coordinate blocks.
        Cartesian coordinates of atoms are then permuted according to the permutation_vector.
        In PES-Learn, this is done such that the element order is most common atom to least common, 
        with an alphabetical tiebreaker. 
        Example: C C H H H O O would be transformed to --> H H H C C O O
    """
    zmat_indices = zmat_indices - 1
    # Convert all angular coordinates (which are in degrees) into radians 
    angular_coord_indices = [i for i in range(2,intcos.shape[1], 3)] + [i for i in range(4,intcos.shape[1] ,3)]
    intcos[:,angular_coord_indices] *= deg2rad
    # Create Cartesians zero array
    cart = np.zeros((intcos.shape[0],natoms,3))
    # Assign Cartesian coordinates of first three atoms: Atom0: origin. Atom1:x=0,y=0,z=r1. Atom2:x=0, y=r2*sin(a1), z=complicated
    cart[:,1,2] = intcos[:,0]
    cart[:,2,1] = intcos[:,1]*np.sin(intcos[:,2])
    cart[:,2,2] = cart[:,zmat_indices[1],2] + (1 - 2 * float(zmat_indices[1]==1))*intcos[:,1]*np.cos(intcos[:,2])

    # Assign Cartesian coordinates of all additional atoms
    j = 3
    for i in range(3, natoms):
        # Pass the Cartesian coordinates of 3 reference atoms for all displacements at once
        local_axes = vectorized_local_axes(cart[:, [zmat_indices[j], zmat_indices[j+1], zmat_indices[j+2]], :])
        bond_vectors = vectorized_bond_vector(intcos[:, [j, j+1, j+2]] )
        disp_vectors = np.einsum('...j, ...jk->...k', bond_vectors, local_axes)
        newcart = cart[:, zmat_indices[j], :] + disp_vectors
        cart[:,i,:] = newcart
        j += 3
    intcos[:,angular_coord_indices] *= rad2deg
    # Permute to standard order (most common elements first, alphabetical tiebreaker)
    p = permutation_vector
    return cart[:,p,:]



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
    data_regex = maybe(r"\d+\n") + r"\s*-?\d+\.\d+\s*\n" + xyz_re
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
    atom_labels = [re.findall(r'\w+', s)[0] for s in sample]
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
    for i,j in enumerate(sorted_atom_labels):
        for k,l in enumerate(atom_labels):
            if j == l:
                p.append(k)
                atom_labels[k] = 'done'
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




