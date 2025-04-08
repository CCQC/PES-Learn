"""
A class for building PES geometries 
"""
from ..utils import geometry_transform_helper as gth
from ..utils import permutation_helper as ph
from ..ml.data_sampler import DataSampler 

from collections import OrderedDict
import gc
import os
import json
import timeit
import pandas as pd
import numpy as np
pd.set_option('display.width',200)
pd.set_option('display.max_colwidth',200)
pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',1000)

class ConfigurationSpace(object):
    """
    Class for generating PES geometries, removing redundancies, reducing grid size.

    Parameters
    ----------
    molecule_obj : :class:`~peslearn.datagen.molecule.Molecule`. 
        Instance of PES-Learn Molecule class. Required for basic information about the molecule; 
        internal coordinates, xyz coordinates, number of atoms.
    input_obj :  :class:`~peslearn.input_processor.InputProcessor`
        Instance of InputProcessor class. Required for user keyword considerations.
    """
    def __init__(self, molecule_obj, input_obj):
        self.mol = molecule_obj
        self.input_obj = input_obj
        self.n_atoms = self.mol.n_atoms - self.mol.n_dummy
        self.n_interatomics =  int(0.5 * (self.n_atoms * self.n_atoms - self.n_atoms))
        self.bond_columns = []
        for i in range(self.n_interatomics):
            self.bond_columns.append("r%d" % (i))

    def generate_displacements(self):
        """
        Generates internal coordinate displacements according to internal coordinate ranges.
        """
        start = timeit.default_timer()
        self.input_obj.extract_intcos_ranges()
        d = self.input_obj.intcos_ranges
        np.random.seed(self.input_obj.keywords['rseed'])
        for key, value in d.items():
            # Draw fixed intervals of values or uniform random between geometry parameter bounds?
            if len(value) == 3 and self.input_obj.keywords['grid_generation'] == 'fixed':
                d[key] = np.linspace(value[0], value[1], value[2])
            elif len(value) == 3 and self.input_obj.keywords['grid_generation'] == 'uniform':
                d[key] = np.random.uniform(value[0], value[1], value[2])
            elif len(value) == 3: # if bad keyword specification, default to fixed
                d[key] = np.linspace(value[0], value[1], value[2])
            elif len(value) == 1:
                d[key] = np.asarray(value[0])    
            else:
                raise Exception("Internal coordinate range improperly specified")
        grid = np.meshgrid(*d.values())
        # 2d array (ngridpoints x ndim) each row is one datapoint
        intcos = np.vstack(map(np.ravel, grid)).T
        print("{} internal coordinate displacements generated in {} seconds".format(intcos.shape[0], round((timeit.default_timer() - start),3)))
        return intcos

    def generate_geometries(self):
        """
        Generates internal coordinates, converts them to Cartesians, and converts them to interatomic distances.
        Stores them into a Pandas DataFrame with columns ['r0', 'r1', 'r2', 'r3', ..., 'rn', 'cartesians', 'internals']
        Where each row of 'cartesians' contain 2d NumPy arrays and 'internals' contain 1d NumPy arrays.
        """
        t1 = timeit.default_timer()
        intcos = self.generate_displacements()
        eq = self.input_obj.keywords['eq_geom']
        if eq:
            intcos = np.vstack((np.array(eq), intcos))
        self.n_disps = intcos.shape[0]
        # Make NumPy array of complete internal coordinates, including dummy atoms (values only). 
        # If internal coordinates have duplicate entries, a different, slightly slower method is needed; the internal coordinates
        # must be expanded to their redundant full definition before Cartesian coordinate conversion
        if self.mol.unique_geom_parameters != self.mol.geom_parameters:
            indices = []
            for p1 in self.mol.geom_parameters:
                for i, p2 in enumerate(self.mol.unique_geom_parameters):
                    if p1==p2:
                        indices.append(i)
            intcos = intcos[:, np.array(indices)]

        # Make NumPy array of cartesian coordinates 
        cartesians = gth.vectorized_zmat2xyz(intcos, self.mol.zmat_indices, self.mol.std_order_permutation_vector, self.mol.n_atoms)
        print("Cartesian coordinates generated in {} seconds".format(round((timeit.default_timer() - t1), 3)))
        t2 = timeit.default_timer()
        # Find invalid Cartesian coordinates which were constructed with invalid Z-Matrices (3 Co-linear atoms)
        colinear_atoms_bool = np.isnan(cartesians).any(axis=(1,2))
        n_colinear = np.where(colinear_atoms_bool)[0].shape[0]
        if n_colinear > 0:
            print("Warning: {} configurations had invalid Z-Matrices with 3 co-linear atoms, tossing them out! Use a dummy atom to prevent.".format(n_colinear))
        # Remove bad Z-Matrix geometries
        cartesians = cartesians[~colinear_atoms_bool]
        intcos = intcos[~colinear_atoms_bool]
        # Pre-allocate memory for interatomic distances array
        interatomics = np.zeros((cartesians.shape[0], self.n_interatomics))
        for atom in range(1, self.n_atoms):
            # Create an array of duplicated cartesian coordinates of this particular atom, for every geometry, which is the same shape as 'cartesians'
            tmp1 = np.broadcast_to(cartesians[:,atom,:], (cartesians.shape[0], 3))
            tmp2 = np.tile(tmp1, (self.n_atoms,1,1)).transpose(1,0,2)
            # Take the non-redundant norms of this atom to all atoms after it in cartesian array
            diff = tmp2[:, 0:atom,:] - cartesians[:, 0:atom,:]
            norms = np.sqrt(np.einsum('...ij,...ij->...i', diff , diff))
            # Fill in the norms into interatomic distances 2d array , n_interatomic_distances)
            if atom == 1:
                idx1, idx2 = 0, 1
            if atom > 1:
                x = int((atom**2 - atom) / 2)
                idx1, idx2 = x, x + atom
            interatomics[:, idx1:idx2] = norms 
        print("Interatomic distances generated in {} seconds".format(round((timeit.default_timer() - t2), 3)))
        # Round all coordinates for nicer printing and redundancy removal.
        intcos.round(10)
        interatomics.round(10)
        cartesians.round(10)
        self.n_disps = cartesians.shape[0]
        # Build DataFrame of all geometries 
        self.all_geometries = pd.DataFrame(index=np.arange(0, cartesians.shape[0]), columns=self.bond_columns)
        self.all_geometries[self.bond_columns] = interatomics
        self.all_geometries['cartesians'] = [cartesians[i,:,:] for i in range(self.n_disps)]
        self.all_geometries['internals'] = [intcos[i,:] for i in range(self.n_disps)]
        print("Geometry grid generated in {} seconds".format(round((timeit.default_timer() - t1),3)))
        if self.input_obj.keywords['tooclose'] != 0:
            self.too_close(tooclose=self.input_obj.keywords['tooclose'])
        return self.all_geometries
        # Memory is expensive to evaluate
        #print("Peak memory usage estimate (GB): ", 3*(self.all_geometries.memory_usage(deep=True).sum() + cartesians.nbytes + interatomics.nbytes + intcos.nbytes)* (1/1e9))

    def too_close(self, tooclose=0.1):
        """
        Check to ensure no interatomic distances are too close.
        """
        start = timeit.default_timer()
        nrows_before = len(self.all_geometries.index)
        df = self.all_geometries.copy()
        df = df.round(10)
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
                mask = df_cols[i:self.n_interatomics]
                df.loc[:,mask] = np.sort(df.loc[:,mask].values, axis=1)
        # remove row if interatomic ditance is less than tooclose        
        for j in range(len(self.bond_columns)):
            df.drop(df[df[self.bond_columns[j]] < tooclose].index, inplace=True)
        self.all_geometries = df
        self.n_tooclose = len(self.all_geometries.index)
        print("Removed {} geometries where atoms were too close from a set of {} geometries in {} seconds.".format(nrows_before-self.n_tooclose, nrows_before, round((timeit.default_timer() - start), 2)))


    def remove_redundancies(self):
        """
        Very fast algorithm for removing redundant geometries from a configuration space
        Has been confirmed to work for C3H2, H2CO, H2O, CH4, C2H2 
        Not proven.
        """
        start = timeit.default_timer()
        nrows_before = len(self.all_geometries.index)
        df = self.all_geometries.copy()
        df = df.round(10)
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
                mask = df_cols[i:self.n_interatomics]
                df.loc[:,mask] = np.sort(df.loc[:,mask].values, axis=1)

        # Remove duplicates
        # take opposite of duplicate boolean Series (which marks duplicates as True)
        mask = -df.duplicated(subset=self.bond_columns)
        self.unique_geometries = self.all_geometries.loc[mask] 
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
        npoints = self.input_obj.keywords['grid_reduction']
        if npoints > self.unique_geometries.shape[0]:
            raise Exception("grid_reduction number of points is greater than the number of points in dataset")
        print("Reducing size of configuration space from {} datapoints to {} datapoints".format(self.n_disps, npoints))
        df = self.unique_geometries.copy()
        df = df[self.bond_columns]
        df['E'] = "" 
        # pandas saved as objects, convert to floats so numpy doesnt reject it
        df = df.apply(pd.to_numeric)
        #sampler = DataSampler(df, npoints, accept_first_n=None)
        sampler = DataSampler(df, npoints, accept_first_n=None)
        sampler.structure_based()
        accepted_indices, rejected_indices = sampler.get_indices()
        self.unique_geometries = self.unique_geometries.iloc[accepted_indices] 
        print("Configuration space reduction complete in {} seconds".format(round((timeit.default_timer() - start),2)))

    def add_redundancies_back(self):
        """
        #TODO currently does not account for simulataneous permutations
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
            

    def generate_templates(self, template_obj):
        # generate the full geometry set or the removed redundancy geometry set?
        self.generate_geometries()
        if self.input_obj.keywords['remove_redundancy'].lower().strip() == 'true':
            print("Removing symmetry-redundant geometries...", end='  ')
            self.remove_redundancies()

            if self.input_obj.keywords['grid_reduction']:
                self.filter_configurations()
            if self.input_obj.keywords['remember_redundancy'].lower().strip() == 'true':
                self.add_redundancies_back()
            df = self.unique_geometries 
        elif self.input_obj.keywords['remove_redundancy'].lower().strip() == 'false':
            df = self.all_geometries
        
        pes_dir_name = self.input_obj.keywords['pes_dir_name']
        if not os.path.exists("./" +  pes_dir_name):
            os.mkdir("./" +  pes_dir_name)
        os.chdir("./" +  pes_dir_name)

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
                tmp_dict = OrderedDict(zip(self.mol.geom_parameters, list(df.iloc[i-1]['internals'])))
                f.write(json.dumps([tmp_dict]))
                #f.write(json.dumps([df.iloc[i-1]['internals']])) 
                if 'duplicate_internals' in df:
                    for j in range(len(df.iloc[i-1]['duplicate_internals'])):
                        f.write("\n")
                        tmp_dict = OrderedDict(zip(self.mol.geom_parameters, df.iloc[i-1]['duplicate_internals'][j]))
                        f.write(json.dumps([tmp_dict]))
                        #f.write(json.dumps([df.iloc[i-1]['duplicate_internals'][j]])) 
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

        os.chdir("../")
        print("Your PES inputs are now generated. Run the jobs in the {} directory and then parse.".format(pes_dir_name))

    def generate_schema(self):

        self.generate_geometries()
        if self.input_obj.keywords['remove_redundancy'].lower().strip() == 'true':
            print("Removing symmetry-redundant geometries...", end='  ')
            self.remove_redundancies()

            if self.input_obj.keywords['grid_reduction']:
                self.filter_configurations()
            if self.input_obj.keywords['remember_redundancy'].lower().strip() == 'true':
                self.add_redundancies_back()
            df = self.unique_geometries 
        elif self.input_obj.keywords['remove_redundancy'].lower().strip() == 'false':
            df = self.all_geometries  

        pes_dir_name = self.input_obj.keywords['pes_dir_name']
        if not os.path.exists("./" +  pes_dir_name):
            os.mkdir("./" +  pes_dir_name)
        os.chdir("./" +  pes_dir_name)    

        for i, cart_array in enumerate(df['cartesians'], start=1):
            xyz = ''
            # check the contents of the input string for keywords necessary for schema generation
            driver = self.input_obj.keywords['schema_driver']
            if driver not in ['energy','hessian','gradient','properties']:
                raise Exception("{} is not a valid option for 'schema_driver', entry must be 'energy', 'hessian', 'gradient', 'properties'".format(driver))            
            method = self.input_obj.keywords['schema_method']
            if method == None:
                raise Exception("'schema_method' cannot be blank, please enter a method.")
            basis = self.input_obj.keywords['schema_basis']
            if basis == None:
                raise Exception("'schema_basis' cannot be blank, please enter a basis.")
            if self.input_obj.keywords['schema_keywords'] == None:
                keywords = '{}'
            else:
                keywords = self.input_obj.keywords['schema_keywords']
            prog = self.input_obj.keywords['schema_prog']
            if prog == None:
                raise Exception("'schema_prog' must be defined, please enter a program.")
            units = self.input_obj.keywords['schema_units']
            if units == 'bohr':
                from .. import constants
    
            if not os.path.exists(str(i)):
                os.mkdir(str(i))

            # tag with internal coordinates, include duplicates if requested
            with open("{}/geom".format(str(i)), 'w') as f:
                tmp_dict = OrderedDict(zip(self.mol.geom_parameters, list(df.iloc[i-1]['internals'])))
                f.write(json.dumps([tmp_dict]))
                #f.write(json.dumps([df.iloc[i-1]['internals']])) 
                if 'duplicate_internals' in df:
                    for j in range(len(df.iloc[i-1]['duplicate_internals'])):
                        f.write("\n")
                        tmp_dict = OrderedDict(zip(self.mol.geom_parameters, df.iloc[i-1]['duplicate_internals'][j]))
                        f.write(json.dumps([tmp_dict]))
                        #f.write(json.dumps([df.iloc[i-1]['duplicate_internals'][j]])) 
            # tag with interatomic distance coordinates, include duplicates if requested
            with open("{}/interatomics".format(str(i)), 'w') as f:
                f.write(json.dumps([OrderedDict(df.iloc[i-1][self.bond_columns])]))
                if 'duplicate_interatomics' in df:
                    for j in range(len(df.iloc[i-1]['duplicate_interatomics'])):
                        f.write("\n") 
                        f.write(json.dumps([df.iloc[i-1]['duplicate_interatomics'][j]])) 

            os.chdir(str(i))

            # write the input files to run with qcengine
            with open('input.py', 'w') as f:
                f.write("import qcengine as qcng\nimport qcelemental as qcel\nimport pprint\n\n")
                f.write('molecule = qcel.models.Molecule.from_data("""\n')
                for j in range(len(self.mol.std_order_atoms)):
                    if units == 'bohr':
                        xyz += "%s %10.10f %10.10f %10.10f\n" % (self.mol.std_order_atom_labels[j], cart_array[j][0] * constants.bohr2angstroms, cart_array[j][1] * constants.bohr2angstroms, cart_array[j][2] * constants.bohr2angstroms)
                    elif units == 'angstrom':
                        xyz += "%s %10.10f %10.10f %10.10f\n" % (self.mol.std_order_atom_labels[j], cart_array[j][0], cart_array[j][1], cart_array[j][2])
                f.write(xyz)
                f.write('""",\nfix_com=True,\nfix_orientation=True)\n')
                if units == 'bohr':
                    f.write('# The above geometry is in Angstroms for QCEngine input purposes.\n\n')
                f.write("driver = '%s'\nmodel = {'method':'%s', 'basis':'%s'}\nkeywords = %s\nprog = '%s'\n\n" % (driver, method, basis, keywords, prog))
                f.write("atomic_inp = qcel.models.AtomicInput(molecule=molecule, driver=driver, model=model, keywords=keywords)\n\n")
                f.write("atomic_res = qcng.compute(atomic_inp, prog)\n\n")
                f.write("with open('%s','w') as f:\n\tpprint.pprint(atomic_res.dict(), f)" % (self.input_obj.keywords['output_name']))
            os.chdir("../")

        print("Your PES inputs are now generated. Run the jobs in the {} directory and then parse.".format(pes_dir_name))

    def generate_PES(self, template_obj=None, schema_gen='false'):
        if self.input_obj.keywords['schema_generate'].lower().strip() == 'true' or schema_gen == 'true':
            self.generate_schema()
        elif template_obj == None and self.input_obj.keywords['schema_generate'].lower().strip() == 'false' and schema_gen == 'false':
            raise Exception("template_obj not found, check your path.")
        else:
            self.generate_templates(template_obj)
            

    def old_remove_redundancies(self):
        """
        Deprecated. Currently a bug: does not consider combined permutation operations,
        just one at a time. 
        Theoretically rigorous, but slow.
        Handles the removal of redundant geometries arising from 
        angular scans and like-atom position permutations
        """
        start = timeit.default_timer()
        nrows_before = len(self.all_geometries.index)
        # first remove straightforward duplicates using interatomic distances
        # (e.g., angular, dihedral equivalencies)
        self.all_geometries[self.bond_columns] = self.all_geometries[self.bond_columns].round(decimals=10)
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


    def old_generate_geometries(self):
        """
        Deprecated. Generates geometries in serial, converting Z-Matrices to cartesians to interatomics one at a time.
        Current implementation converts these to array operations, converting all geometries coordinates simultaneously.
        """
        start = timeit.default_timer()
        intcos = self.generate_displacements() 
        self.disps = []
        for gridpoint in intcos:
            tmp = OrderedDict([(self.mol.unique_geom_parameters[i], gridpoint[i])  for i in range(intcos.shape[1])])
            self.disps.append(tmp)
        # grab cartesians, internals, interatomics representations of geometry
        cartesians = []
        internals = []
        interatomics = []
        failed = 0 # keep track of failed 3 co-linear atom configurations
        # this loop of geometry transformations/saving is pretty slow, but scales linearly at least
        for i, disp in enumerate(self.disps):
            self.mol.update_intcoords(disp)
            try:
                cart = self.mol.zmat2xyz()
            except:
                failed += 1
                continue
            cartesians.append(cart)
            internals.append(disp)
            idm = gth.get_interatom_distances(cart)
            # remove float noise for duplicate detection
            idm = np.round(idm[np.tril_indices(len(idm),-1)],10)
            interatomics.append(idm)
        # preallocate dataframe space 
        if failed > 0:
            print("Warning: {} configurations had invalid Z-Matrices with 3 co-linear atoms, tossing them out! Use a dummy atom to prevent.".format(failed))
        df = pd.DataFrame(index=np.arange(0, len(self.disps)-failed), columns=self.bond_columns)
        df[self.bond_columns] = interatomics
        df['cartesians'] = cartesians
        df['internals'] = internals 
        self.all_geometries = df
        print("Geometry grid generated in {} seconds".format(round((timeit.default_timer() - start),2)))
