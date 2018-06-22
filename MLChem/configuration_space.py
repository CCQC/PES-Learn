"""
A class for building PES geometries 
"""

from geometry_transform_helper import get_interatom_distances
from permutation_helper import permute_bond_indices, induced_permutations

class ConfigurationSpace(object):
    """
    Generates PES geometries and removes redundancies, including like-atom permutation redundancies
    Parameters
    ----------
    molecule_obj : Instance of Molecule class
    input_obj    : Instance of InputProcessor class
    """
    def __init__(self, molecule_obj, input_obj):
        self.mol = molecule_obj
        self.input_obj = input_obj
        self.disps = self.input_obj.generate_displacements() 
        self.n_init_disps = len(self.disps)


    def generate_geometry_dataframes(self):
        print("Number of displacements without redundancy removal: {}".format(self.n_init_disps))
        n_interatomics =  int(0.5 * (self.mol.n_atoms * self.mol.n_atoms - self.mol.n_atoms))
        print("Number of interatomic distances: {}".format(n_interatomics))
        bond_columns = []
        for i in range(n_interatomics):
            bond_columns.append("r%d" % (i))
        # preallocate df space, much faster
        df = pd.DataFrame(index=np.arange(0, len(disps)), columns=bond_columns)
        # grab cartesians and internals 
        cartesians = []
        internals = []
        for i, disp in enumerate(disps):
            mol.update_intcoords(disp)
            cart = mol.zmat2xyz()
            cartesians.append(cart)
            internals.append(disp)
            idm = get_interatom_distances(cart)
            idm = idm[np.tril_indices(len(idm),-1)]
            # remove float noise for duplicate detection
            df.iloc[i] = np.round(idm.astype(np.double),10) 
        df['cartesians'] = cartesians
        df['internals'] = internals 
        # remove straightforward duplicates (e.g., angular, dihedral equivalencies)
        df.drop_duplicates(subset=bond_columns, inplace=True)
        return df
        
