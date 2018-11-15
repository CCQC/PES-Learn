"""
A class for building PES geometries 
"""
from . import geometry_transform_helper as gth
from . import permutation_helper as ph
from .data_sampler import DataSampler 
from collections import OrderedDict
import os
import json
import timeit
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
        self.n_disps = len(self.disps)  # updated if redundancies are removed
        self.n_atoms = self.mol.n_atoms - self.mol.n_dummy
        self.n_interatomics =  int(0.5 * (self.n_atoms * self.n_atoms - self.n_atoms))
        self.bond_columns = []
        for i in range(self.n_interatomics):
            self.bond_columns.append("r%d" % (i))

    def generate_geometries(self):
        start = timeit.default_timer()
        print("Number of displacements: {}".format(self.n_init_disps))
        print("Number of interatomic distances: {}".format(self.n_interatomics))
        # grab cartesians, internals, interatomics representations of geometry
        cartesians = []
        internals = []
        interatomics = []
        # this loop of geometry transformations/saving is pretty slow, but scales linearly at least
        for i, disp in enumerate(self.disps):
            self.mol.update_intcoords(disp)
            cart = self.mol.zmat2xyz()
            cartesians.append(cart)
            internals.append(disp)
            idm = gth.get_interatom_distances(cart)
            # remove float noise for duplicate detection
            idm = np.round(idm[np.tril_indices(len(idm),-1)],10)
            interatomics.append(idm)
        # preallocate dataframe space 
        df = pd.DataFrame(index=np.arange(0, len(self.disps)), columns=self.bond_columns)
        df[self.bond_columns] = interatomics
        df['cartesians'] = cartesians
        df['internals'] = internals 
        self.all_geometries = df
        print("Geometry grid generated in {} seconds".format(round((timeit.default_timer() - start),2)))

    def remove_redundancies(self):
        """
        Very fast algorithm for removing redundant geometries from a configuration space
        Has been confirmed to work for C3H2, H2CO, H2O. 
        Not proven.
        """
        start = timeit.default_timer()
        nrows_before = len(self.all_geometries.index)
        df = self.all_geometries.copy()
        og_cols = df.columns.tolist()
        # sort interatomic distance columns according to alphabetized bond types
        # e.g. OH HH CH --> CH HH OH
        alpha_bond_cols = [og_cols[i] for i in self.mol.alpha_bond_types_indices]
        alpha_bond_cols.append('cartesians')
        alpha_bond_cols.append('internals')
        df = df[alpha_bond_cols]
        df_cols = df.columns.tolist()
        # sort values of each 'bondtype' subpartition of interatomic distance columns
        # subpartitions are defined by the index of the first occurance of each 
        # bond_type label.  CH CH CH HH HH OH would be [0,3,5]. These define partition bounds.
        ind = self.mol.alpha_bond_types_first_occur_indices
        K = len(ind)
        # sort each subpartition
        for i in range(K):
            if i < (K - 1):
                cut = slice(ind[i], ind[i+1])
                mask = df_cols[cut]
                df.loc[:,mask] = np.sort(df.loc[:,mask].values, axis=1)
            else:
                mask = df_cols[i+1:self.n_interatomics]
                #mask = df_cols[i+1:]
                df.loc[:,mask] = np.sort(df.loc[:,mask].values, axis=1)

        # Remove duplicates
        self.unique_geometries = df.drop_duplicates(subset=self.bond_columns)
        self.n_disps = len(self.unique_geometries.index)
        print("Redundancy removal took {} seconds".format(round((timeit.default_timer() - start),2)))
        print("Removed {} redundant geometries from a set of {} geometries".format(nrows_before-self.n_disps, nrows_before))

    def filter_configurations(self):
        """
        Filters the configuration space by computing the norms between geometries.
        Accepts the first point, then the point furthest from that point.
        Each subsequently added point is the one which has the longest distance 
        into the set of currently accepted points 
        """
        start = timeit.default_timer()
        npoints = self.input_obj.keywords['filter_geoms']
        print("Reducing size of configuration space from {} datapoints to {} datapoints".format(self.n_disps, npoints))
        df = self.unique_geometries.copy()
        df = df[self.bond_columns]
        df['E'] = "" 
        # pandas saved as objects, convert to floats so numpy doesnt reject it
        df = df.apply(pd.to_numeric)
        sampler = DataSampler(df, npoints, accept_first_n=None)
        sampler.structure_based()
        accepted_indices, rejected_indices = sampler.get_indices()
        self.unique_geometries = self.unique_geometries.iloc[accepted_indices] 
        print("Configuration space reduction complete in {} seconds".format(round((timeit.default_timer() - start),2)))

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
            print("Removing symmetry-redundant geometries...", end='  ')
            self.remove_redundancies()
            # keep track of redundant geometries for later?
            if self.input_obj.keywords['remember_redundancy'].lower().strip() == 'true':
                self.add_redundancies_back()
            #else:
            #    df = self.unique_geometries 
            if self.input_obj.keywords['filter_geoms']:
                self.filter_configurations()
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
        
        
    #def norm_based_remove_redundancies(self, max_dupes=5):
    #    """
    #     Removes "redundancies" based on norms. Can set tolerance so that cluster of geometries get filtered.
    #    Near-linear scaling wrt size of dataset. Assumes only 'max_dupes' duplicates of a given geometry at most
    #    """
    #    start = timeit.default_timer()
    #    nrows_before = len(self.all_geometries.index)
    #    # Take the interatomic distances, identify their bond type (CH, OH, etc)  
    #    # this may be good for speed up: droppoing angular redudnandt duplicates
    #    #tmp = self.all_geometries.drop_duplicates(subset=self.bond_columns)
    #    #tmp  = self.all_geometries.drop_duplicates(subset=self.bond_columns)
    #    #self.all_interatomics = tmp[self.bond_columns]
    #    #self.all_interatomics = self.all_geometries.drop_duplicates(subset=self.bond_columns)[self.bond_columns]
    #    self.all_interatomics = self.all_geometries[self.bond_columns]
    #    og_cols = self.all_interatomics.columns.tolist()
    #    # sort interatomic distance columns according to alphabetized bond types
    #    newcols = [og_cols[i] for i in self.mol.alpha_bond_types_indices]
    #    df = self.all_interatomics[newcols]
    #    df_cols = df.columns.tolist()
    #    # sort values of each 'bondtype' subpartition of interatomic distance columns
    #    # subpartitions are defined by the index of the first occurance of each 
    #    # bond_type label.  CH CH CH HH HH OH would be [0,3,5]. These define partition bounds.
    #    ind = self.mol.alpha_bond_types_first_occur_indices
    #    K = len(ind)
    #    # sort each subpartition
    #    for i in range(K):
    #        if i < (K - 1):
    #            cut = slice(ind[i], ind[i+1])
    #            mask = df_cols[cut]
    #            # may need to copy df?
    #            df.loc[:,mask] = np.sort(df.loc[:,mask].values, axis=1)
    #            #df.loc[:,mask] = np.sort(newdata.loc[:,mask].values, axis=1)
    #        else:
    #            mask = df_cols[i+1:]
    #            df.loc[:,mask] = np.sort(df.loc[:,mask].values, axis=1)
    #            #df.loc[:,mask] = np.sort(newdata.loc[:,mask].values, axis=1)

        # define function to compute norms between one geometry and all geometries below it
        #def norm(vec, df):
        #    """Assumes dataframe with only interatomic distances """
        #    tmp1 = np.tile(vec, (df.shape[0], 1))
        #    diff = tmp1 - df.values[:]
        #    diff = diff.astype(float)
        #    #print(diff.dtype)
        #    norm_vector = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        #    return norm_vector

        #redundancy_indices = [] 
        #size = df.shape[0]
        #max_dupes = 5   # assumes at most 5 duplicates for any given geometry!!!
#       #                  this assumption improves scaling dramatically.
        #for i in range(1, size + 1):
        #    n = norm(df.values[i-1], df[i:])
        #    if n.shape[0] == max_dupes:
        #        max_dupes -= 1
        #    # have to include last one, +1
        #    indices = np.argpartition(n, max_dupes)[:max_dupes+1]
        #    count = 0
        #    for k in indices:
        #        #if n[k] < 1e-8:
        #        if n[k] == 0.0:
        #            count += 1
        #            # additively adjust since k is based on norm array, which shrinks over time
        #            redundancy_indices.append(k + i)
        ## remove duplicate redundancy indices
        #redundancy_indices = list(set(redundancy_indices))
        #r = np.asarray(redundancy_indices)
        ## now obtain unique indices
        #all_indices = np.arange(nrows_before)
        #unique_indices = np.delete(all_indices, all_indices[r])
        #self.unique_geometries = self.all_geometries.iloc[unique_indices]
        #self.unique_geometries = df.drop_duplicates(subset=self.bond_columns)
        #nrows_after = len(self.unique_geometries.index)
        #print("Removed {} redundant geometries from a set of {} geometries".format(nrows_before-nrows_after, nrows_before))
        #print("Redundancy removal took {} seconds".format(round((timeit.default_timer() - start),2)))


    def old_remove_redundancies(self):
        """
        Deprecated. Theoretically rigorous, but slow.
        Handles the removal of redundant geometries arising from 
        angular scans and like-atom position permutations
        """
        start = timeit.default_timer()
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
        print("Redundancy removal complete {} seconds".format(round((timeit.default_timer() - start),2)))
        print("Removed {} redundant geometries from a set of {} geometries".format(nrows_before-nrows_after, nrows_before))

