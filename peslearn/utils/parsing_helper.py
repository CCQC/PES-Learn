import pandas as pd
import numpy as np
import os
import json
from collections import OrderedDict
from ..datagen.outputfile import OutputFile

def parse(input_obj, mol): 
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
    elif input_obj.keywords['pes_format'] == 'interatomics':
        data = pd.DataFrame(index=None, columns = mol.interatomic_labels)
        geom_path = "/interatomics"
    else:
        raise Exception("pes_format keyword value invalid. Must be 'internals' or 'interatomics'")

    if input_obj.keywords['energy']: 
        data['E'] = ''
    if input_obj.keywords['gradient']: 
        data['G'] = ''

    # parse output files 
    os.chdir("./PES_data")
    dirs = [i for i in os.listdir(".") if os.path.isdir(i) ]
    for d in dirs: 
        path = d + "/output.dat"
        #output_obj = MLChem.outputfile.OutputFile(path)
        output_obj = OutputFile(path)
        if input_obj.keywords['energy']:
            E = extract_energy(input_obj, output_obj)
        if input_obj.keywords['gradient']:
            G = extract_gradient(output_obj)
        with open(d + geom_path) as f:
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

    if input_obj.keywords['sort_pes'] == 'true': 
        data = data.sort_values("E")
    data.to_csv("PES.dat", sep=',', index=False, float_format='%12.12f')
    print("Parsed data has been written to PES.dat")
