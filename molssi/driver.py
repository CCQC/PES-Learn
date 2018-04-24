"""
Temporary, quick and dirty driver for PES scans
"""

import sys
import os
import json
# python 2 and 3 command line input compatibility
from six.moves import input
# find molssi module
sys.path.insert(0, "../../")
import molssi

input_obj = molssi.input_processor.InputProcessor("./input.dat")
print(input_obj.keywords['energy'])
template_obj = molssi.template_processor.TemplateProcessor("./template.dat")
mol = molssi.molecule.Molecule(input_obj.zmat_string)
disps = input_obj.generate_displacements()

text = input("Do you want to 'generate' or 'parse' your data? Type one option and hit enter: ")

if text == 'generate':
    # get data from input file containing internal coordinate configuration space
    input_obj = molssi.input_processor.InputProcessor("./input.dat")
    # get template file data
    template_obj = molssi.template_processor.TemplateProcessor("./template.dat")
    # create a molecule
    mol = molssi.molecule.Molecule(input_obj.zmat_string)
    # take internal coordinate ranges, expand them, generate displacement dictionaries
    disps = input_obj.generate_displacements()
    
    # create displacement input files
    # this maybe should be a method in the TemplateProcessor class, but idk it needs InputProcessor disps
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
    input_obj = molssi.input_processor.InputProcessor("./input.dat")
    mol = molssi.molecule.Molecule(input_obj.zmat_string)
    geom_labels = mol.geom_parameters
    DATA = pd.DataFrame(columns = geom_labels)

    os.chdir("./PES_data")
    ndirs = sum(os.path.isdir(d) for d in os.listdir("."))

    E = []
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
        output_obj = molssi.outputfile.OutputFile(path)
        if input_obj.keywords['extract'] == 'cclib':
            try:
                energy = output_obj.extract_energy_with_cclib(input_obj.keywords['energy'])
            except:
                raise Exception("Either the cclib option is not valid, or cclib is having trouble parsing your data. Try using regular expressions instead.")
        if input_obj.keywords['extract'] == 'regex':
            energy = output_obj.extract_energy_with_regex(input_obj.keywords['energy'])

        #TODO add gradient keyword checks 
        #gradient = output_obj.extract_cartesian_gradient_with_regex(
        #"Total Gradient:", "\*\*\* tstop() called on", "\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
        E.append(energy)

        # handle cases when gradient doesn't exist
        #try:
        #    grad_std_dev = np.std(gradient)
        #    grad_stds.append(grad_std_dev)
        #except:
        #    grad_std_dev = None
        #    grad_stds.append(grad_std_dev)
        DATA = DATA.append(df)
        
    DATA['E'] = E
    os.chdir('../')
    DATA.to_csv("PES.csv", sep=',', index=False, float_format='%12.12f')
    with open('./PES.dat', 'w') as f:
        f.write(DATA.__repr__())

    print("Parsed data has been written to PES.dat")
    
