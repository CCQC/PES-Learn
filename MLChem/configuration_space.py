"""
A class for building PES geometries 
"""
from . import geometry_transform_helper as gth
from . import permutation_helper as ph
import os
import json
import pandas as pd
import numpy as np

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

    def generate_geometries(self):
        print("Number of displacements: {}".format(self.n_init_disps))
        n_atoms = self.mol.n_atoms - self.mol.n_dummy
        n_interatomics =  int(0.5 * (n_atoms * n_atoms - n_atoms))
        print("Number of interatomic distances: {}".format(n_interatomics))
        self.bond_columns = []
        for i in range(n_interatomics):
            self.bond_columns.append("r%d" % (i))
        # preallocate df space, much faster
        df = pd.DataFrame(index=np.arange(0, len(self.disps)), columns=self.bond_columns)
        # grab cartesians and internals 
        cartesians = []
        internals = []
        for i, disp in enumerate(self.disps):
            self.mol.update_intcoords(disp)
            cart = self.mol.zmat2xyz()
            cartesians.append(cart)
            internals.append(disp)
            idm = gth.get_interatom_distances(cart)
            idm = idm[np.tril_indices(len(idm),-1)]
            # remove float noise for duplicate detection
            df.iloc[i] = np.round(idm.astype(np.double),10) 
        df['cartesians'] = cartesians
        df['internals'] = internals 
        self.all_geometries = df

    def remove_redundancies(self):
        """
        Automated Redundant Geometry Removal
        Handles the removal of redundant geometries arising from 
        angular scans and like-atom position permutations
        """
        nrows_before = len(self.all_geometries.index)
        # first remove straightforward duplicates using interatomic distances
        # (e.g., angular, dihedral equivalencies)
        self.unique_geometries = self.all_geometries.drop_duplicates(subset=self.bond_columns)
        # remove like-atom permutation duplicates
        bond_indice_permutations = ph.permute_bond_indices(self.mol.atom_count_vector)
        bond_permutation_vectors = ph.induced_permutations(self.mol.atom_count_vector, bond_indice_permutations) 
        print("Interatomic distances equivalent permutations: ", bond_permutation_vectors)
        for perm in bond_permutation_vectors:
            new_df = []
            permuted_rows = []
            for row in self.unique_geometries.itertuples(index=False):
                # apply induced bond permutation derived from like-atom permutations
                # the first n rows are the interatomic distances which we want to permute, the last two rows are the cartesian and internal coordinates
                new = [row[0:-2][i] for i in perm]  
                permuted_rows.append(new)
                # if its unaffected by the permutation, we want to keep one copy
                if new == list(row[0:-2]):
                    new_df.append(row)
                # uniqueness check
                if list(row[0:-2]) not in permuted_rows:
                    new_df.append(row)
            # update dataframe with removed rows for this particular permutation vector
            self.unique_geometries = pd.DataFrame(new_df)
        nrows_after = len(self.unique_geometries.index)
        print("Removed {} redundant geometries from a set of {} geometries".format(nrows_before-nrows_after, nrows_before))


    def generate_PES(self, template_obj):
        # generate the full geometry set or the removed redundancy geometry set?
        self.generate_geometries()
        if self.input_obj.keywords['remove_redundancy'].lower() == 'true':
            self.remove_redundancies()
            df = self.unique_geometries 
        elif self.input_obj.keywords['remove_redundancy'].lower() == 'false':
            df = self.all_geometries
          
        if not os.path.exists("./PES_data"):
            os.mkdir("./PES_data")
        os.chdir("./PES_data")

        for i, cart_array in enumerate(df['cartesians'], start=1):
            # build xyz input file
            xyz = ''
            xyz += template_obj.header_xyz()
            for j in range(len(self.mol.std_order_atoms)):
                xyz += "%s %10.10f %10.10f %10.10f\n" % (self.mol.std_order_atom_labels[j], cart_array[j][0], cart_array[j][1], cart_array[j][2])
            xyz += template_obj.footer_xyz()

            if not os.path.exists(str(i)):
                os.mkdir(str(i))
            # grab original internal coordinates 
            with open("{}/geom".format(str(i)), 'w') as f:
                # preserve OrderedDict order by dumping as an iterable (list)
                f.write(json.dumps([df.iloc[i-1,-1]])) 
            # grab interatomic distances 
            #print(df.iloc[i-1,0:-2])
            #print(df.iloc[i-1])
            df.iloc[i-1,0:-2].to_json("{}/interatomics".format(str(i)))
            # write input file 
            with open("{}/{}".format(str(i), self.input_obj.keywords['input_name']), 'w') as f:
                f.write(xyz)
        print("Your PES inputs are now generated. Run the jobs in the PES_data directory and then parse.")
        
        
