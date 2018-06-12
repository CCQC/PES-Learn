"""
Data Generation Driver
"""
import sys
import os
import json
# python 2 and 3 command line input compatibility
from six.moves import input
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
disps = input_obj.generate_displacements()

text = input("Do you want to 'generate' or 'parse' your data? Type one option and hit enter: ")

if text == 'generate':
    # get data from input file containing internal coordinate configuration space
    input_obj = MLChem.input_processor.InputProcessor("./input.dat")
    # get template file data
    template_obj = MLChem.template_processor.TemplateProcessor("./template.dat")
    # create a molecule
    mol = MLChem.molecule.Molecule(input_obj.zmat_string)
    # take internal coordinate ranges, expand them, generate displacement dictionaries
    disps = input_obj.generate_displacements()

    # Automated Redundant Geometry Removal
    n_interatomics =  int(0.5 * (mol.n_atoms * mol.n_atoms - mol.n_atoms))
    bond_columns = []
    for i in range(n_interatomics):
        bond_columns.append("r%d" % (i))
    # preallocate df space, much faster
    df = pd.DataFrame(index=np.arange(0, len(disps)), columns=bond_columns)
    # grab cartesians for input file construction later
    cartesians = []
    for i, disp in enumerate(disps):
        mol.update_intcoords(disp)
        cart = mol.zmat2xyz()
        cartesians.append(cart)
        idm = get_interatom_distances(cart)
        idm = idm[np.tril_indices(len(idm),-1)]
        #idm = np.around(np.asarray(idm), decimals=8)
        # remove float noise for duplicate detection
        df.iloc[i] = np.round(idm.astype(np.double),10) 
        #df.iloc[i] = idm
    nrows_before = len(df.index)
    # we add cartesians here, so that when we drop duplicates we also delete the corresponding cartesian geometries.
    df['cartesians'] = cartesians
    # remove standard duplicates 
    df.drop_duplicates(subset=bond_columns, inplace=True)
    # remove like-atom permutation duplicates
    bond_indice_permutations = permute_bond_indices(mol.atom_count_vector)
    bond_permutation_vectors = induced_permutations(mol.atom_count_vector, bond_indice_permutations) 
    # for each permutation, and each geometry, apply the permutation and check if it already exists in the dataframe
    new_df = [] 
    permuted_rows = []
    for perm in bond_permutation_vectors:
        for row in df.itertuples():
            new = [row[1:-1][i] for i in perm]                  # apply induced bond permutation derived from like-atom permutations
            permuted_rows.append(new)
            # if its unaffected by the permutation, we want to keep one copy
            if new == list(row[1:-1]):
                new_df.append(row)
            # uniqueness check
            if list(row[1:-1]) not in permuted_rows:
                new_df.append(row)

    new_df = pd.DataFrame(new_df)
    nrows_after = len(new_df.index)
    print("Removed {} redundant geometries".format(nrows_before-nrows_after))
    # create a "PES_data" directory and move into it
    if not os.path.exists("./PES_data"):
        os.mkdir("./PES_data")
    os.chdir("./PES_data")

    for i, cart_array in enumerate(new_df['cartesians'], start=1):
        # build xyz input file
        xyz = ''
        xyz += template_obj.header_xyz()
        for j in range(len(mol.std_order_atoms)):
            xyz += "%s %10.10f %10.10f %10.10f\n" % (mol.std_order_atom_labels[j], cart_array[j][0], cart_array[j][1], cart_array[j][2])
        xyz += template_obj.footer_xyz()

        if not os.path.exists(str(i)):
            os.mkdir(str(i))
        os.chdir(str(i))
        # keep internal coordinates handy
        with open("geom", 'w') as f:
            f.write(json.dumps(disps[i-1]))
        # write input file 
        with open("input.dat", 'w') as f:
            f.write(xyz)
        os.chdir("../.")


    #for i, disp in enumerate(disps, start=1):
    #    mol.update_intcoords(disp)
    #    cart_array = mol.zmat2xyz()
    #    # build xyz input file
    #    xyz = ''
    #    xyz += template_obj.header_xyz()
    #    for j in range(len(mol.atom_labels)):
    #        xyz += "%s %10.10f %10.10f %10.10f\n" % (mol.atom_labels[j], cart_array[j][0], cart_array[j][1], cart_array[j][2])
    #    xyz += template_obj.footer_xyz()

    #    if not os.path.exists(str(i)):
    #        os.mkdir(str(i))
    #    os.chdir(str(i))
    #    # keep internal coordinates handy
    #    with open("geom", 'w') as f:
    #        f.write(json.dumps(disp))
    #    # write input file 
    #    with open("input.dat", 'w') as f:
    #        f.write(xyz)
    #    os.chdir("../.")

    print("Your PES inputs are now generated. Run the jobs in the PES_data directory and then parse.")


if text == 'parse':
    import pandas as pd
    import numpy as np
    # get geom labels, intialize data frame
    input_obj = MLChem.input_processor.InputProcessor("./input.dat")
    mol = MLChem.molecule.Molecule(input_obj.zmat_string)
    DATA = pd.DataFrame(columns = mol.geom_parameters)

    os.chdir("./PES_data")
    ndirs = sum(os.path.isdir(d) for d in os.listdir("."))

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
    
    E = []
    G = []
    # parse output files 
    for i in range(1, ndirs+1):
        # get geometry data
        with open(str(i) + "/geom") as f:
            tmp = json.load(f)
        new = []
        for l in mol.geom_parameters:
            new.append(tmp[l])
        df = pd.DataFrame([new], columns = mol.geom_parameters)
        path = str(i) + "/output.dat"
        # get output data (energies and/or gradients)
        output_obj = MLChem.outputfile.OutputFile(path)
        if input_obj.keywords['energy']: 
            E.append(extract_energy(input_obj, output_obj))
        if input_obj.keywords['gradient']: 
            G.append(extract_gradient(output_obj))
        DATA = DATA.append(df)
    if E:
        DATA['E'] = E
    if G:
        DATA['G'] = G
    os.chdir('../')
    DATA.to_csv("PES.dat", sep=',', index=False, float_format='%12.12f')
    # this method skips data that is too long
    #with open('./PES.dat', 'w') as f:
    #    f.write(DATA.__repr__())
    
    print("Parsed data has been written to PES.dat")




