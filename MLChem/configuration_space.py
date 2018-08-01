"""
A class for building PES geometries 
"""
from . import geometry_transform_helper as gth
from . import permutation_helper as ph
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
            #print(len(self.unique_geometries.index))
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
        self.unique_geometries['duplicates'] = np.empty((len(self.unique_geometries), 0)).tolist()
        # grab interatomic distance equivalent permutation operations
        bond_indice_permutations = ph.permute_bond_indices(self.mol.atom_count_vector)
        bond_permutation_vectors = ph.induced_permutations(self.mol.atom_count_vector, bond_indice_permutations) 
        # list of lists of bond interatomics from self.all_geometries
        tmp_geoms = self.all_geometries[self.bond_columns].values.tolist() 
        # for every permutation on every unique geometry, apply the permutation and see if it exists in the original geom dataset
        # if it does, add the internal coordinates of duplicate from original geom dataset to duplicates column in self.unique_geometries
        for perm in bond_permutation_vectors:
            permuted_rows = []
            for row in self.unique_geometries.itertuples(index=False):
                # apply permutation, check if it changed, if it did, check if it is in original geom dataset 
                # if it is in original dataset, and not already in the duplicates column of self.unique_geometries, add it 
                new = [row[0:-3][i] for i in perm]  
                if new != list(row[0:-3]):
                    if new in tmp_geoms:
                        x = self.all_geometries.iloc[tmp_geoms.index(new)]['internals']  #grab internal coords
                        if x not in row[-1]:
                            row[-1].append(x) 


#   def remove_redundancies(self):
#       """
#       Automated Redundant Geometry Removal
#       Handles the removal of redundant geometries arising from 
#       angular scans and like-atom position permutations
#       Removed redundant geometries internal coordinates are kept in a final column of the DataFrame called "duplicates"
#       """
#       nrows_before = len(self.all_geometries.index)
#       bond_indice_permutations = ph.permute_bond_indices(self.mol.atom_count_vector)
#       bond_permutation_vectors = ph.induced_permutations(self.mol.atom_count_vector, bond_indice_permutations) 
#       print("Interatomic distances equivalent permutations: ", bond_permutation_vectors)
#       # create new dataframe for unique geometries, with an additional empty column to save the removed duplicates
#       self.unique_geometries = self.all_geometries.copy()
#       print("Done with copying")
#       self.unique_geometries['duplicates'] = np.empty((len(self.unique_geometries), 0)).tolist()
#       new_df = []
#       duplicate_indices = []
#       # for every row, we first collect all the allowed bond permutations arising from like-atom permutations
#       # next, we look at every row after the present row. 
#       # If a latter row matches a permutation of the original row, we save the latter rows internal coordinates in 'duplicates' column of original row.
#       # A duplicate is only saved if it is unique
#       # we also save the index of the duplicate. We only add a row to the final dataframe if it is not detected as a duplicate.
#       for row in self.unique_geometries.itertuples(index=True): # for every row
#           new = row[1:] # <- is this needed?
#           permutations = []
#           permutations.append(list(row[1:-3]))
#           # apply all possible bond permutations, collect them
#           for perm in bond_permutation_vectors:
#               bond_column_perm = [row[1:-3][i] for i in perm]
#               permutations.append(bond_column_perm)
#           for i in range(row[0] + 1, len(self.unique_geometries)): # look at every row after
#               dupes = []
#               for p in permutations:
#                    # look at all the permutations of the current geometry, if its the same as a latter geometry, save it 
#                    if p == list(self.unique_geometries.iloc[i][self.bond_columns]):  # row[-3,-2,-1] is cartesians, internals, duplicates, row[0] is index
#                        if self.unique_geometries.iloc[i]['internals'] not in dupes: # different permutations may lead to the same duplicate detection, make sure its unique
#                            new[-1].append(self.unique_geometries.iloc[i]['internals'])  # add to duplicates column
#                        duplicate_indices.append(i)
#                        dupes.append(self.unique_geometries.iloc[i]['internals'])
#            if row[0] not in duplicate_indices:
#                new_df.append(new)
#            
#        print("Done removing redundancies")
#        all_columns = self.bond_columns + ['cartesians','internals','duplicates']
#        self.unique_geometries = pd.DataFrame(new_df, columns=all_columns)
#        nrows_after = len(self.unique_geometries.index)
#        print("Removed {} redundant geometries from a set of {} geometries".format(nrows_before-nrows_after, nrows_before))


    def generate_PES(self, template_obj):
        # generate the full geometry set or the removed redundancy geometry set?
        self.generate_geometries()
        if self.input_obj.keywords['remove_redundancy'].lower() == 'true':
            self.remove_redundancies()
        if self.input_obj.keywords['pes_print'].lower() == 'all':
            self.add_redundancies_back()
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
                #f.write(json.dumps([df.iloc[i-1,-2]])) 
                #for j in range(len(df.iloc[i-1,-1])):
                #    f.write("\n") 
                #    f.write(json.dumps([df.iloc[i-1,-1][j]])) 
                f.write(json.dumps([df.iloc[i-1]['internals']])) 
                if 'duplicates' in df:
                    for j in range(len(df.iloc[i-1]['duplicates'])):
                        f.write("\n") 
                        f.write(json.dumps([df.iloc[i-1]['duplicates'][j]])) 
            # grab interatomic distances 
            #df.iloc[i-1,0:-3].to_json("{}/interatomics".format(str(i)))
            df.loc[i-1,self.bond_columns].to_json("{}/interatomics".format(str(i)))
            # write input file 
            with open("{}/{}".format(str(i), self.input_obj.keywords['input_name']), 'w') as f:
                f.write(xyz)
        print("Your PES inputs are now generated. Run the jobs in the PES_data directory and then parse.")
        
        
