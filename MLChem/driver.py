"""
Temporary, quick and dirty driver for PES scans
"""

import sys
import os
import json
# python 2 and 3 command line input compatibility
from six.moves import input
# find MLChem module
sys.path.insert(0, "../../")
import MLChem

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
    
    # create displacement input files
    # this should maybe be implemented in it a class

    # create a "data" directory and move into it
    if not os.path.exists("./PES_data"):
        os.mkdir("./PES_data")
    os.chdir("./PES_data")
    
    for i, disp in enumerate(disps, start=1):
        mol.update_intcoords(disp)
        cart_array = mol.zmat2xyz()
        xyz = ''
        xyz += template_obj.header_xyz()
        for j in range(len(mol.atom_labels)):
            xyz += "%s %10.10f %10.10f %10.10f\n" % (mol.atom_labels[j], cart_array[j][0], cart_array[j][1], cart_array[j][2])
        xyz += template_obj.footer_xyz() 
    
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
        os.chdir(str(i))
        with open("geom", 'w') as f:
            f.write(json.dumps(disp))
        with open("input.dat", 'w') as f:
            f.write(xyz)
        os.chdir("../.")

    print("Your PES inputs are now generated. Run the jobs in the PES_data directory and then parse.")

if text == 'parse':
    import pandas as pd
    import numpy as np
    # get geom labels, intialize data frame
    input_obj = MLChem.input_processor.InputProcessor("./input.dat")
    mol = MLChem.molecule.Molecule(input_obj.zmat_string)
    geom_labels = mol.geom_parameters
    DATA = pd.DataFrame(columns = geom_labels)

    os.chdir("./PES_data")
    ndirs = sum(os.path.isdir(d) for d in os.listdir("."))


    #TODO remove all these logic statements from for loop, they are not necessary and slowing it down. Most only need to be checked once
    E = []
    G = []
    # parse output files after running jobs
    for i in range(1, ndirs+1):
        # get geometry data
        with open(str(i) + "/geom") as f:
            tmp = json.load(f) 
        new = []
        for l in geom_labels:
            new.append(tmp[l])
        df = pd.DataFrame([new], columns = geom_labels)
        path = str(i) + "/output.dat"
        # get output data (energies and/or gradients)
        output_obj = MLChem.outputfile.OutputFile(path)


        # parse energies
        if input_obj.keywords['energy'] == 'cclib':
            if input_obj.keywords['energy_cclib']:
                try:
                    energy = output_obj.extract_energy_with_cclib(input_obj.keywords['energy_cclib'])
                    E.append(energy)
                except:
                    raise Exception("\n Looks like cclib failed to parse your data. Try using regular expressions instead.") 
            else:
                raise Exception("\n Please select which cclib energy to parse; e.g. energy_cclib = 'scfenergies', energy_cclib = 'ccenergies' ")

        if input_obj.keywords['energy'] == 'regex':
            if input_obj.keywords['energy_regex']:
                energy = output_obj.extract_energy_with_regex(input_obj.keywords['energy_regex'])
                E.append(energy)
            else:
                raise Exception("\n energy_regex value not assigned in input. Please add a regular expression which captures the energy value, e.g. energy_regex = 'RHF Final Energy is: \s+(-\d+\.\d+)'")

        # parse gradients 
        if input_obj.keywords['gradient'] == 'cclib':
            try:
                gradient = output_obj.extract_cartesian_gradient_with_cclib()
                gradient = np.asarray(gradient)
                G.append(gradient)
            except:
                raise Exception("cclib failed to parse your gradient. Try using regular expressions instead.")

        if input_obj.keywords['gradient'] == 'regex':
            header = input_obj.keywords['gradient_header']
            footer = input_obj.keywords['gradient_footer']
            grad_line_regex = input_obj.keywords['gradient_line']
            if header and footer and grad_line_regex:
                try:
                    gradient = output_obj.extract_cartesian_gradient_with_regex(header, footer, grad_line_regex)
                    gradient = np.asarray(gradient)
                    G.append(gradient)
                except:
                #TODO
                    raise Exception("")
            else:
                raise Exception("For regular expression gradient extraction, gradient_header, gradient_footer, and gradient_line string identifiers are required to isolate the cartesian gradient block. See documentation for details")

        #gradient = output_obj.extract_cartesian_gradient_with_regex(
        #"Total Gradient:", "\*\*\* tstop() called on", "\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")

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
    
