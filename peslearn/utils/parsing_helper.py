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
            raise Exception("\n energy_regex value not assigned in input. Please add a regular expression which captures the energy value, see docs for more info.")
        
    if input_obj.keywords['energy'] == 'schema':
        def extract_energy(input_obj, output_obj):
            energy = output_obj.extract_from_schema(driver='energy')
            return energy
    
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

    elif input_obj.keywords['gradient'] == 'schema':
        def extract_gradient(output_obj, input_obj):
                gradient = output_obj.extract_from_schema(driver='gradient')
                return gradient
    
    if input_obj.keywords['hessian'] == 'schema':
        def extract_hessian(input_obj, output_obj):
            hessian = output_obj.extract_from_schema(driver='hessian')
            return hessian

    #add function to parse properties from schema

    # parse original internals or interatomics?
    if input_obj.keywords['pes_format'] == 'zmat':
        data = pd.DataFrame(index=None, columns = mol.unique_geom_parameters)
        geom_path = "/geom"
    elif input_obj.keywords['pes_format'] == 'interatomics':
        data = pd.DataFrame(index=None, columns = mol.interatomic_labels)
        geom_path = "/interatomics"
    else:
        raise Exception("pes_format keyword value invalid. Must be 'zmat' or 'interatomics'")

    if input_obj.keywords['energy']: 
        data['E'] = ''
    if input_obj.keywords['gradient']: 
        ngrad = 3*(mol.n_atoms - mol.n_dummy) 
        grad_cols = ["g%d" % (i) for i in range(ngrad)]
        for i in grad_cols:
            data[i] = ''
    if input_obj.keywords['hessian']:
        nhess = (3*(mol.n_atoms - mol.n_dummy))*(3*(mol.n_atoms - mol.n_dummy))
        hess_cols = ["h%d" % (i) for i in range(nhess)]
        for i in hess_cols:
            data[i] = ''
        

    # parse output files 
    E = 0
    G = 0
    H = 0
    os.chdir("./" + input_obj.keywords['pes_dir_name'])
    dirs = [i for i in os.listdir(".") if os.path.isdir(i) ]
    dirs = sorted(dirs, key=lambda x: int(x))
    for d in dirs:  
        path = d + "/" + input_obj.keywords['output_name']
        output_obj = OutputFile(path)
        if input_obj.keywords['energy']:
            E = extract_energy(input_obj, output_obj)
        if input_obj.keywords['gradient']:
            G = extract_gradient(output_obj, input_obj)
            ngrad = 3*(mol.n_atoms - mol.n_dummy) 
            grad_cols = ["g%d" % (i) for i in range(ngrad)]
        if input_obj.keywords['hessian']:
            H = extract_hessian(input_obj, output_obj)
            nhess = (3*(mol.n_atoms - mol.n_dummy))*(3*(mol.n_atoms - mol.n_dummy))
            hess_cols = ["h%d" % (i) for i in range(nhess)]
                
        if E == 'False' or G == 'False' or H == 'False':
            with open('errors.txt','a') as e:
                    error_string = 'File in dir {} returned an error, the parsed output has been omitted from {}.\n'.format(d, input_obj.keywords['pes_name'])
                    e.write(error_string)
        else:
            with open(d + geom_path) as f:
                for line in f:
                    tmp = json.loads(line, object_pairs_hook=OrderedDict)
                    df = pd.DataFrame(data=tmp, index=None, columns=tmp[0].keys())
                    if input_obj.keywords['energy']:
                        df['E'] = E
                    if input_obj.keywords['gradient']:
                        df2 = pd.DataFrame(data=[G.flatten().tolist()],index=None, columns=grad_cols)
                        df = pd.concat([df, df2], axis=1)
                    if input_obj.keywords['hessian']:
                        df3 = pd.DataFrame(data=[H.flatten().tolist()], index=None, columns=hess_cols)
                        df = pd.concat([df,df3], axis=1)
                    data = pd.concat([data, df])
                    if input_obj.keywords['pes_redundancy'] == 'true':
                        continue
                    else:
                        break
    os.chdir('../')

    if input_obj.keywords['sort_pes'] == 'true': 
        if input_obj.keywords['gradient'] or input_obj.keywords['hessian']:
            if input_obj.keywords['energy']:
                data = data.sort_values("E")
            else:
                print("Keyword 'sort_pes' is set to 'true' (default), this only applies to energies and your data has NOT been sorted")
        else:
            data = data.sort_values("E")
    data.to_csv(input_obj.keywords['pes_name'], sep=',', index=False, float_format='%12.12f')
    print("Parsed data has been written to {}".format(input_obj.keywords['pes_name']))
    # if num_errors > 0:
    #     print("One or more output files returned an error, refer to {}/errors.txt for more information".format(input_obj.keywords['pes_dir_name']))


    error_path = "./" + input_obj.keywords['pes_dir_name'] + "/errors.txt"
    if os.path.exists(error_path):
        print("One or more output files returned an error, refer to {}/errors.txt for more information".format(input_obj.keywords['pes_dir_name']))
