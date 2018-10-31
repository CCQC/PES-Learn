"""
A class for building PES geometries 
"""
from . import geometry_transform_helper as gth
from . import permutation_helper as ph
from collections import OrderedDict
import os
import json
import pandas as pd
import numpy as np
pd.set_option('display.width',200)
pd.set_option('display.max_colwidth',200)

class ConfigurationSpace(object):
    """
    Generates PES geometries and removes redundancies, including like-atom permutation redundancies
    Parameters
    ----------
    molecule_obj : Instance of Molecule class. Required for basic information about the molecule; internal coordinates, xyz coordinates, number of atoms
    input_obj    : Instance of InputProcessor class. Required for user keyword considerations, and the generation of displacements.
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
        # preallocate dataframe space, much faster
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
        print("Removing symmetry-redundant geometries... this may take a while")
        nrows_before = len(self.all_geometries.index)
        # first remove straightforward duplicates using interatomic distances
        # (e.g., angular, dihedral equivalencies)
        self.unique_geometries = self.all_geometries.drop_duplicates(subset=self.bond_columns)
        print("Removed {} angular-redundant geometries. Now removing permutation-redundant geometries.".format(len(self.all_geometries) - len(self.unique_geometries)))
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
                # add new geometry to checklist
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


    def add_redundancies_back(self):
        """
        Takes self.unique_geometries (which contains [bond_columns], cartesians, internals)
        and adds a last column, called duplicates, which contains internal coordinate dictionaries of duplicate geometries
        """
        # WARNING since you do not drop straightforward dupes from self.all_geometries, there may be multiple 'new's in tmp_geoms
        # this is a fix, is it problematic?
        self.all_geometries = self.all_geometries.drop_duplicates(subset=self.bond_columns)
        # add column of duplicates, each row has its own empty list
        self.unique_geometries['duplicate_internals'] = np.empty((len(self.unique_geometries), 0)).tolist()
        self.unique_geometries['duplicate_interatomics'] = np.empty((len(self.unique_geometries), 0)).tolist()
        # current column structure of self.unique_geometries:
        # [interatomics], cartesians, internals, duplicate_internals, duplicate_interatomics

        # grab interatomic distance equivalent permutation operations
        bond_indice_permutations = ph.permute_bond_indices(self.mol.atom_count_vector)
        bond_permutation_vectors = ph.induced_permutations(self.mol.atom_count_vector, bond_indice_permutations) 
        # list of lists of bond interatomics from self.all_geometries
        tmp_geoms = self.all_geometries[self.bond_columns].values.tolist() 
        # for every permutation on every unique geometry, apply the permutation and see if it exists in the original dataset
        # if it does, add the internal and interatomic distance coordinates of duplicate from original geom dataset to duplicates column in self.unique_geometries
        for perm in bond_permutation_vectors:
            permuted_rows = []
            for row in self.unique_geometries.itertuples(index=False):
                # apply permutation to interatomic distances (index 0 --> -3, check if it changed, if it did, check if it is in original geom dataset 
                # if it is in original dataset, and not already in the duplicates column of self.unique_geometries, add it 
                new = [row[0:-4][i] for i in perm]  
                if new != list(row[0:-4]):
                    if new in tmp_geoms:
                        intcoord = self.all_geometries.iloc[tmp_geoms.index(new)]['internals']  #grab internal coords
                        # add duplicate to duplicate_internals column if it has not been found
                        if intcoord not in row[-2]:
                            row[-2].append(intcoord)
                        # save as OrderedDict since internal coordinates are also OrderedDict
                        idm = OrderedDict(self.all_geometries.iloc[tmp_geoms.index(new)][self.bond_columns])  #grab interatomic distance coords
                        # add duplicate to duplicate_interatomics column if it has not been found
                        if idm not in row[-1]:
                            row[-1].append(idm) 


    def generate_PES(self, template_obj):
        # generate the full geometry set or the removed redundancy geometry set?
        self.generate_geometries()
        if self.input_obj.keywords['remove_redundancy'].lower().strip() == 'true':
            self.remove_redundancies()
            # keep track of redundant geometries for later?
            if self.input_obj.keywords['remember_redundancy'].lower().strip() == 'true':
                self.add_redundancies_back()
                df = self.unique_geometries 
        elif self.input_obj.keywords['remove_redundancy'].lower().strip() == 'false':
            df = self.all_geometries
          
        if not os.path.exists("./PES_data"):
            os.mkdir("./PES_data")
        os.chdir("./PES_data")

        for i, cart_array in enumerate(df['cartesians'], start=1):
            # build xyz input file and put in directory
            xyz = ''
            xyz += template_obj.header_xyz()
            for j in range(len(self.mol.std_order_atoms)):
                xyz += "%s %10.10f %10.10f %10.10f\n" % (self.mol.std_order_atom_labels[j], cart_array[j][0], cart_array[j][1], cart_array[j][2])
            xyz += template_obj.footer_xyz()
            if not os.path.exists(str(i)):
                os.mkdir(str(i))

            # tag with internal coordinates, include duplicates if requested
            with open("{}/geom".format(str(i)), 'w') as f:
                f.write(json.dumps([df.iloc[i-1]['internals']])) 
                if 'duplicate_internals' in df:
                    for j in range(len(df.iloc[i-1]['duplicate_internals'])):
                        f.write("\n")
                        f.write(json.dumps([df.iloc[i-1]['duplicate_internals'][j]])) 
            # tag with interatomic distance coordinates, include duplicates if requested
            with open("{}/interatomics".format(str(i)), 'w') as f:
                f.write(json.dumps([OrderedDict(df.iloc[i-1][self.bond_columns])]))
                if 'duplicate_interatomics' in df:
                    for j in range(len(df.iloc[i-1]['duplicate_interatomics'])):
                        f.write("\n") 
                        f.write(json.dumps([df.iloc[i-1]['duplicate_interatomics'][j]])) 
            # write input file for electronic structure theory package 
            with open("{}/{}".format(str(i), self.input_obj.keywords['input_name']), 'w') as f:
                f.write(xyz)

        print("Your PES inputs are now generated. Run the jobs in the PES_data directory and then parse.")
        
        
