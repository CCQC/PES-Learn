import constants
import re
import regex
import math
import numpy as np

"""
Contains Atom and Molecule classes for reading, saving and editing the geometry of a molecule 
Much of the code was adapted from tmpchem/computational_chemistry/scripts/geometry_analysis/zmat2xyz.py
"""

# calculate distance between two 3-d cartesian coordinates
def get_r12(coords1, coords2):
    r2 = 0.0
    for p in range(3):
        r2 += (coords2[p] - coords1[p])**2
    r = math.sqrt(r2)
    return r

# calculate unit vector between to 3-d cartesian coordinates
def get_u12(coords1, coords2):
    r12 = get_r12(coords1, coords2)
    u12 = [0.0 for p in range(3)]
    for p in range(3):
        u12[p] = (coords2[p] - coords1[p]) / r12
    return u12

# calculate dot product between two unit vectors
def get_udp(uvec1, uvec2):
    udp = 0.0
    for p in range(3):
        udp += uvec1[p] * uvec2[p]
    udp = max(min(udp, 1.0), -1.0)
    return udp

# calculate unit cross product between two unit vectors
def get_ucp(uvec1, uvec2):
    ucp = [0.0 for p in range(3)]
    cos_12 = get_udp(uvec1, uvec2)
    sin_12 = math.sqrt(1 - cos_12**2)
    ucp[0] = (uvec1[1]*uvec2[2] - uvec1[2]*uvec2[1]) / sin_12
    ucp[1] = (uvec1[2]*uvec2[0] - uvec1[0]*uvec2[2]) / sin_12
    ucp[2] = (uvec1[0]*uvec2[1] - uvec1[1]*uvec2[0]) / sin_12
    return ucp

def get_local_axes(coords1, coords2, coords3):
    u21 = get_u12(coords1, coords2)
    u23 = get_u12(coords2, coords3)
    if (abs(get_udp(u21, u23)) >= 1.0):
      print('\nError: Co-linear atoms in an internal coordinate definition')
      sys.exit()
    u23c21 = get_ucp(u23, u21)
    u21c23c21 = get_ucp(u21, u23c21)
    z = u21
    y = u21c23c21
    x = get_ucp(y, z)
    local_axes = [x, y, z]
    return local_axes

# calculate vector of bond in local axes of internal coordinates
def get_bond_vector(r, a, d):
    x = r * math.sin(a) * math.sin(d)
    y = r * math.sin(a) * math.cos(d)
    z = r * math.cos(a)
    bond_vector = [x, y, z]
    return bond_vector


class Atom(object):
    """
    The Atom class holds information about the geometry of an atom
        Parameters
        ----------
        label : str
            The atomic symbol
        rnum  : str
            The bond connectivity indice, as represented in a Z-matrix
        anum  : str
            The angle connectivity indice, as represented in a Z-matrix
        dnum  : str
            The dihedral connectivity indice, as represented in a Z-matrix
        r  : dict
            A dictionary of the bond label e.g. "R1" and the value 
        a  : dict
            A dictionary of the angle label e.g. "A1" and the value 
        d  : dict
            A dictionary of the dihedral label e.g. "D1" and the value 
    """
    def __init__(self, label, rnum, anum, dnum, r, a, d):
        self.label = label
        self.rnum = rnum
        self.anum = anum
        self.dnum = dnum
        self.r    = r
        self.a    = a
        self.d    = d
        rlist     = list(self.r.values())
        alist     = list(self.a.values())
        dlist     = list(self.d.values())
        self.rval = rlist[0]
        self.aval = alist[0] 
        self.dval = dlist[0]
        
        self.coords = [None for j in range(3)]

    

class Molecule(object):
    """
    The Molecule class holds geometry information about all the atoms in the molecule
    Requires initialization with a file string containing internal coordinates
    """
    def __init__(self, zmat_string):
        self.zmat_string = zmat_string
        self.extract_zmat(self.zmat_string)
         
    
    def extract_zmat(self, zmat_string):
        # grab array like representation of zmatrix and count the number of atoms 
        zmat_array = [line.split() for line in zmat_string.splitlines()]
        self.n_atoms = len(zmat_array)

        # find geometry parameter labels 
        # atom labels will always be at index 0, 1, 3, 6, 6++4... 
        # and geometry parameters are all other matches
        tmp = re.findall(regex.coord_label, zmat_string)
        atom_labels = []
        for i, s in enumerate(tmp):
                if (i == 0) or (i == 1) or (i == 3):
                    atom_labels.append(tmp[i])
                if ((i >= 6) and ((i-6) % 4 == 0)):
                    atom_labels.append(tmp[i])
        self.geom_parameters = [x for x in tmp if x not in atom_labels]
        
        self.atoms = []
        for i in range(self.n_atoms):
            label = zmat_array[i][0]
            if (i >= 1):
                rnum = int(zmat_array[i][1]) - 1
                r = {zmat_array[i][2]: None} #change later to work for both compact and std internals
            else:
                rnum = None
                r = {None: None}
            if (i >= 2):
                anum = int(zmat_array[i][3]) - 1
                a = {zmat_array[i][4]: None}
            else:
                anum = None
                a = {None: None}
            if (i >= 3):
                dnum = int(zmat_array[i][5]) - 1
                d = {zmat_array[i][6]: None}
            else:
                dnum = None
                d = {None: None}
            self.atoms.append(Atom(label, rnum, anum, dnum, r, a, d))

    def zmat2xyz(self):
        """
        Convert Z-matrix representation to cartesian coordinates
        """
        if (self.n_atoms >= 1):
            self.atoms[0].coords = [0.0, 0.0, 0.0]
        if (self.n_atoms >= 2):
            self.atoms[1].coords = [0.0, 0.0, self.atoms[1].rval]
        if (self.n_atoms >= 3):
            r1,  r2  = self.atoms[1].rval, self.atoms[2].rval
            rn1, rn2 = self.atoms[1].rnum, self.atoms[2].rnum
            a1 = self.atoms[2].aval
            y = r2*math.sin(a1)
            z = self.atoms[rn2].coords[2] + (1-2*float(rn2==1))*r2*math.cos(a1)
            self.atoms[2].coords = [0.0, y, z]
        for i in range(3, self.n_atoms):
            atom = self.atoms[i]
            coords1 = self.atoms[atom.rnum].coords
            coords2 = self.atoms[atom.anum].coords
            coords3 = self.atoms[atom.dnum].coords
            # get_local_axes, get_bond_vector
            self.atoms[i].local_axes = get_local_axes(coords1, coords2, coords3)
            bond_vector = get_bond_vector(atom.rval, atom.aval, atom.dval)
            disp_vector = np.array(np.dot(bond_vector, self.atoms[i].local_axes))
            for p in range(3):
                atom.coords[p] = self.atoms[atom.rnum].coords[p] + disp_vector[p]

 

zmatstring = 'O\nH 1 R1\nH 1 R2 2 A1\nH 1 R3 2 A2 3 D1\n'
print(zmatstring)
mol = Molecule(zmatstring)
mol.atoms[1].r["R1"] = 1.2
mol.atoms[1].rval = 1.2
mol.atoms[2].r["R2"] = 1.1
mol.atoms[2].rval = 1.1
mol.atoms[2].a["A1"] = -109.1
mol.atoms[2].aval = -109.1
mol.atoms[3].r["R3"] = 1.1
mol.atoms[3].rval = 1.1
mol.atoms[3].a["A2"] = 109.
mol.atoms[3].aval = 109.
mol.atoms[3].d["D1"] = 180.1
mol.atoms[3].dval = 180.1
mol.zmat2xyz()
for atom in mol.atoms:
    print(atom.coords)
#    print(atom.rval)
#    print(atom.aval)
#    print(atom.dval)
#print(mol.geom_parameters)

