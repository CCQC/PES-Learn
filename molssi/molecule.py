import constants
import re
import regex
import math
import numpy as np
from geometry_transform_helper import get_local_axes, get_bond_vector

"""
Contains Atom and Molecule classes for reading, saving and editing the geometry of a molecule 
Built around reading internal coordinates 
"""

class Atom(object):
    """
    The Atom class holds information about the geometry of an atom
        Parameters
        ----------
        label : str
            The atomic symbol
        r_idx  : str
            The bond connectivity index, as represented in a Z-matrix
        a_idx  : str
            The angle connectivity index, as represented in a Z-matrix
        d_idx  : str
            The dihedral connectivity index, as represented in a Z-matrix
        intcoords  : dict
            A dictionary of geometry parameter labels (e.g. "R1") and the value for this atom
    """
    def __init__(self, label, r_idx=None, a_idx=None, d_idx=None, intcoords={}):
        self.label = label
        self.r_idx = r_idx
        self.a_idx = a_idx
        self.d_idx = d_idx
        self.intcoords = intcoords
        self.update_intcoords(intcoords)
    
    def update_intcoords(self, new_intcoords):
        self.intcoords = new_intcoords
        self.geom_vals = list(self.intcoords.values())
        while len(self.geom_vals) < 3:
            self.geom_vals.append(None)
        self.rval, self.aval, self.dval = self.geom_vals[0], self.geom_vals[1], self.geom_vals[2]
    

    
# This framework is likely temporary. 
# It is probably better to let github/mcocdawc/chemcoord handle internals and cartesians in the future.
class Molecule(object):
    """
    The Molecule class holds geometry information about all the atoms in the molecule
    Requires initialization with a file string containing internal coordinates
    """
    def __init__(self, zmat_string):
        self.zmat_string = zmat_string
        self.extract_zmat(self.zmat_string)
         
    
    def extract_zmat(self, zmat_string):
        """
        This should maybe just be in the init method.
        Take the string which contains an isolated Z matrix definition block,
        and extract information and save the following attributes:
        self.n_atoms         - the number of atoms in the molecule
        self.atom_labels     - a list of element labels 'H', 'O', etc.
        self.geom_parameters - a list of geometry labels 'R3', 'A2', etc.
        self.atoms           - a list of Atom objects containing complete Z matrix information for each Atom
        """
        # grab array like representation of zmatrix and count the number of atoms 
        zmat_array = [line.split() for line in zmat_string.splitlines()]
        self.n_atoms = len(zmat_array)

        # find geometry parameter labels 
        # atom labels will always be at index 0, 1, 3, 6, 6++4... 
        # and geometry parameters are all other matches
        tmp = re.findall(regex.coord_label, zmat_string)
        self.atom_labels = []
        for i, s in enumerate(tmp):
                if (i == 0) or (i == 1) or (i == 3):
                    self.atom_labels.append(tmp[i])
                if ((i >= 6) and ((i-6) % 4 == 0)):
                    self.atom_labels.append(tmp[i])
        self.geom_parameters = [x for x in tmp if x not in self.atom_labels]
        
        self.atoms = []
        for i in range(self.n_atoms):
            label = zmat_array[i][0]
            intcoords = {}
            r_idx, a_idx, d_idx = None, None, None
            if (i >= 1):
                r_idx = int(zmat_array[i][1]) - 1
                intcoords[zmat_array[i][2]] = None
            if (i >= 2):
                a_idx = int(zmat_array[i][3]) - 1
                intcoords[zmat_array[i][4]] = None
            if (i >= 3):
                d_idx = int(zmat_array[i][5]) - 1
                intcoords[zmat_array[i][6]] = None
            self.atoms.append(Atom(label, r_idx, a_idx, d_idx, intcoords))
    
    def update_intcoords(self, disp):
        """
        Disp is a list, by atom, of dictionaries of geometry parameters
        [{}, {'R1': 1.01}, {'R2':2.01, 'A1':104.5} ...]
        NOTE: allows geometry parameter labels to be changed by the disp
        """
        for i, atom in enumerate(mol.atoms):
            atom.update_intcoords(disp[i])

    def zmat2xyz(self):
        """
        Convert Z-matrix representation to cartesian coordinates
        Perserves the element ordering of the Z-matrix
        """
        if (self.n_atoms >= 1):
            self.atoms[0].coords = np.array([0.0, 0.0, 0.0])
        if (self.n_atoms >= 2):
            self.atoms[1].coords = np.array([0.0, 0.0, self.atoms[1].rval])
        if (self.n_atoms >= 3):
            r1,  r2  = self.atoms[1].rval, self.atoms[2].rval
            rn1, rn2 = self.atoms[1].r_idx, self.atoms[2].r_idx
            a1 = self.atoms[2].aval
            y = r2*math.sin(a1)
            z = self.atoms[rn2].coords[2] + (1-2*float(rn2==1))*r2*math.cos(a1)
            self.atoms[2].coords = np.array([0.0, y, z])
        for i in range(3, self.n_atoms):
            atom = self.atoms[i]
            coords1 = self.atoms[atom.r_idx].coords
            coords2 = self.atoms[atom.a_idx].coords
            coords3 = self.atoms[atom.d_idx].coords
            self.atoms[i].local_axes = get_local_axes(coords1, coords2, coords3)
            bond_vector = get_bond_vector(atom.rval, atom.aval, atom.dval)
            disp_vector = np.array(np.dot(bond_vector, self.atoms[i].local_axes))
            for p in range(3):
                atom.coords[p] = self.atoms[atom.r_idx].coords[p] + disp_vector[p]

        cartesian_coordinates = []
        for atom in self.atoms:
            cartesian_coordinates.append(atom.coords)
        return np.array(cartesian_coordinates)

 

##zmatstring = 'O\nH 1 R1\nH 1 R2 2 A1\nH 1 R3 2 A2 3 D1\n'
#zmatstring = 'O\nH 1 R1\nH 1 R2 2 A1\nH 3 R3 1 A2 2 D1\n'
#zmatstring = 'O\nH 1 R1\nH 1 R2 2 A1\n'
#
#mol = Molecule(zmatstring)
#for atom in mol.atoms:
#    print(atom.intcoords)
#
#
#disp = [{}, {'R1': 1.1}, {'R2': 1.1, 'A2': 150.0 * constants.deg2rad}]
#mol.update_intcoords(disp)
#
#for atom in mol.atoms:
#    print(atom.intcoords)

#a = mol.zmat2xyz()
#print(a)
#print(mol.n_atoms)

#print(zmatstring)
#mol = Molecule(zmatstring)
#print(mol.geom_parameters)
##mol.atoms[1].r["R1"] = 1.2
#mol.atoms[1].rval = 1.1
##mol.atoms[2].r["R2"] = 1.1
#mol.atoms[2].rval = 1.1
##mol.atoms[2].a["A1"] = -109.1
#mol.atoms[2].aval = 150.0 * constants.deg2rad
###mol.atoms[3].r["R3"] = 1.1
#mol.atoms[3].rval = 1.1
###mol.atoms[3].a["A2"] = 90.0
#mol.atoms[3].aval = 180.0* constants.deg2rad
###mol.atoms[3].d["D1"] = 180.1
#mol.atoms[3].dval = 1.0 * constants.deg2rad
#a = mol.zmat2xyz()
#print(a)
#for atom in mol.atoms:
#    print(atom.coords)
#    print(atom.rval)
#    print(atom.aval)
#    print(atom.dval)
#print(mol.geom_parameters)

