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
sys.path.insert(0, "../../../")
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

    # parse original internals or interatomics?
    if input_obj.keywords['pes_format'] == 'internals':
        data = pd.DataFrame(index=None, columns = mol.unique_geom_parameters)
        geom_path = "/geom"
    else: 
        data = pd.DataFrame(index=None, columns = mol.interatomic_labels)
        geom_path = "/interatomics"

    if input_obj.keywords['energy']: 
        data['E'] = ''
    if input_obj.keywords['gradient']: 
        data['G'] = ''

    # parse output files 
    for i in range(1, ndirs+1):
        path = str(i) + "/output.dat"
        output_obj = MLChem.outputfile.OutputFile(path)
        if input_obj.keywords['energy']:
            E = extract_energy(input_obj, output_obj)
        if input_obj.keywords['gradient']:
            G = extract_gradient(output_obj)
        with open(str(i) + geom_path) as f:
            for line in f:
                tmp = json.loads(line, object_pairs_hook=OrderedDict)
                df = pd.DataFrame(data=tmp, index=None, columns=tmp[0].keys())
                df['E'] = E
                data = data.append(df)
                if input_obj.keywords['pes_redundancy'] == 'true':
                    continue
                else:
                    break
    os.chdir('../')
    data.to_csv("PES.dat", sep=',', index=False, float_format='%12.12f')
    print("Parsed data has been written to PES.dat")


