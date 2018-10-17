"""
Data Generation Driver
"""
import timeit
import sys
import os
import json
# python 2 and 3 command line input compatibility
from six.moves import input
from collections import OrderedDict
from geometry_transform_helper import get_interatom_distances 
from permutation_helper import permute_bond_indices, induced_permutations
# find MLChem module
sys.path.insert(0, "../../")
import MLChem
import numpy as np
import pandas as pd


input_obj = MLChem.input_processor.InputProcessor("./input.dat")
template_obj = MLChem.template_processor.TemplateProcessor("./template.dat")
mol = MLChem.molecule.Molecule(input_obj.zmat_string)
text = input("Do you want to 'generate' or 'parse' your data? Type one option and hit enter: ")
start = timeit.default_timer()

if text == 'generate':
    config = MLChem.configuration_space.ConfigurationSpace(mol, input_obj)
    config.generate_PES(template_obj)


print("Data generation finished in {} seconds".format(round((timeit.default_timer() - start),2)))

if text == 'parse':
    os.chdir("./PES_data")
    ndirs = sum(os.path.isdir(d) for d in os.listdir("."))
        
    # separate this from driver later
    # define energy extraction routine based on user keywords
    if input_obj.keywords['energy'] == 'cclib':
        if input_obj.keywords['energy_cclib']: 
            def extract_energy(input_obj, output_obj):
                energy = output_obj.extract_energy_with_cclib(input_obj.keywords['energy_cclib'])
                return energy
        #TODO add flag for when cclib fails to parse, currently just outputs a None 
        else:                                                                  
            raise Exception("\n Please indicate which cclib energy to parse; e.g. energy_cclib = 'scfenergies', energy_cclib = 'ccenergies' ")
    
    elif input_obj.keywords['energy'] == 'regex': 
        if input_obj.keywords['energy_regex']: 
            def extract_energy(input_obj, output_obj):
                energy = output_obj.extract_energy_with_regex(input_obj.keywords['energy_regex'])
                return energy
        else:
            raise Exception("\n energy_regex value not assigned in input. Please add a regular expression which captures the energy value, e.g. energy_regex = 'RHF Final Energy: \s+(-\d+\.\d+)'")
    # TODO add JSON schema support  
    
    # define gradient extraction routine based on user keywords
    if input_obj.keywords['gradient'] == 'cclib':
        def extract_gradient(output_obj):
            gradient = output_obj.extract_cartesian_gradient_with_cclib() 
            # not needed, (unless it's None when grad isnt found?)
            #gradient = np.asarray(gradient)                                   
            return gradient
    
    elif input_obj.keywords['gradient'] == 'regex':
        header = input_obj.keywords['gradient_header']  
        footer = input_obj.keywords['gradient_footer']  
        grad_line_regex = input_obj.keywords['gradient_line'] 
        if header and footer and grad_line_regex:
            def extract_gradient(output_obj, h=header, f=footer, g=grad_line_regex):
                gradient = output_obj.extract_cartesian_gradient_with_regex(h, f, g)
                #gradient = np.asarray(gradient)
                return gradient
        else:
            raise Exception("For regular expression gradient extraction, gradient_header, gradient_footer, and gradient_line string identifiers are required to isolate the cartesian gradient block. See documentation for details")   
    # Currently outputs internal coordinates with redundancies added back in if they were removed.
    # TODO make adding redundancies back in optional with a keyword
    # May in the future want an interatomic distances PES output, with redundancies included or not.
    # TODO add interatomic distance output support 
    DATA = pd.DataFrame(index=np.arange(0,input_obj.ndisps), columns = mol.unique_geom_parameters)
    # add columns for parsed data
    if input_obj.keywords['energy']: 
        DATA['E'] = ''
    if input_obj.keywords['gradient']: 
        DATA['G'] = ''
    # parse output files 
    for i in range(1, ndirs+1):
        path = str(i) + "/output.dat"
        output_obj = MLChem.outputfile.OutputFile(path)
        if input_obj.keywords['energy']: 
            E = extract_energy(input_obj, output_obj)
        if input_obj.keywords['gradient']: 
            G = extract_gradient(output_obj)
        # get geometry data
        with open(str(i) + "/geom") as f:
            for line in f:
                tmp = json.loads(line, object_pairs_hook=OrderedDict)
        #with open(str(i) + "/interatomics") as f:
        #    tmp = json.loads(f.read(), object_pairs_hook=OrderedDict)
                df = pd.DataFrame(tmp, columns=tmp[0].keys())
                df['E'] = E
                DATA = DATA.append(df)
    os.chdir('../')
    DATA.to_csv("PES.dat", sep=',', index=False, float_format='%12.12f')
    print("Parsed data has been written to PES.dat")

    # new parser that reports redundant geometry data
    #if input_obj.keywords['remove_redundancy'].lower() == 'true':
    #    if input_obj.keywords['PES_redundancy'].lower() == 'true':
    #       # # START block of code which assumes unique geometries configuration space
    #       # config = MLChem.configuration_space.ConfigurationSpace(mol, input_obj)
    #       # config.remove_redundancies()
    #       # # dataframe with interatomics r1,r2,r3..., cartesians (arrays), internals (OrderedDicts) column data
    #       # DATA = config.unique_geometries 
    #       # E = []
    #       # G = []
    #       # for i in range(1, ndirs+1):
    #       #     # get output data (energies and/or gradients)
    #       #     path = str(i) + "/output.dat"
    #       #     output_obj = MLChem.outputfile.OutputFile(path)
    #       #     if input_obj.keywords['energy']: 
    #       #         E.append(extract_energy(input_obj, output_obj))
    #       #     if input_obj.keywords['gradient']: 
    #       #         G.append(extract_gradient(output_obj))
    #       # if E:
    #       #     DATA['E'] = E
    #       # if G:
    #       #     DATA['G'] = G
    #       # # END block of code which assumes unique geometries configuration space

    #        # following code assumes redundancy removal was used for efficiency, but the redundancies are still desired in the dataset 
    #        config = MLChem.configuration_space.ConfigurationSpace(mol, input_obj)
    #        all_geoms = config.all_geometries
    #        if input_obj.keywords['energy']: 
    #            all_geoms['E'] = ""
    #        if input_obj.keywords['gradient']: 
    #            all_geoms['G'] = ""
    #        
    #        # for every PES data directory, check if geom file matches a row element 'internals' in
    #        # config.all_geometries dataframe
    #        for i in range(1, ndirs+1):
    #            with open(str(i) + "/geom") as f:
    #                tmp = json.loads(f.read(), object_pairs_hook=OrderedDict)
    #            for row in all_geoms.itertuples(index=False):
    #                if row['internals'] == tmp:
    #                    # extraction routine
    #                    output_obj = MLChem.outputfile.OutputFile(str(i) + "/output.dat")
    #                    if input_obj.keywords['energy']: 
    #                        row['E'] = extract_energy(input_obj, output_obj)
    #                    if input_obj.keywords['gradient']: 
    #                        row['G'] = extract_gradient(output_obj)

    #        # now for remaining geometries without assigned energy values, check if a permutation operation fills it
    #        bond_indice_permutations = permute_bond_indices(mol.atom_count_vector)
    #        bond_permutation_vectors = induced_permutations(mol.atom_count_vector, bond_indice_permutations)
    #        for perm in bond_permutation_vectors:
    #            permuted_rows = []
    #            for row in all_geoms.itertuples(index=False):
    #                new = [row[0:-2][i] for i in perm]
    #                # if this permutation exists in original geometry dataset, add it to DATA  
    #                if (all_geoms[all_geoms.columns[0:-2]] == new).all(1).any():


    #        os.chdir('../')
    #        DATA.to_csv("PES.dat", sep=',', index=False, float_format='%12.12f')
    #        print("Parsed data has been written to PES.dat")
    #             

    #    if input_obj.keywords['PES_redundancy'].lower() == 'false':



