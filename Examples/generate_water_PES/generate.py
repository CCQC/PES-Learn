"""
Temporary, quick and dirty driver for data generation
"""

import sys
import os
# find molssi module
sys.path.insert(0, "../../")
import molssi


# get data from input file containing internal coordinate configuration space
input_obj = molssi.input_processor.InputProcessor("./input.dat")
# get template file data
template_obj = molssi.template_processor.TemplateProcessor("./template.dat")
# create a molecule
mol = molssi.molecule.Molecule(input_obj.zmat_string)
# take internal coordinate ranges, expand them, generate displacement dictionaries
disps = input_obj.generate_displacements()

# create a "data" directory and move into it
if not os.path.exists("./data"):
    os.mkdir("./data")
os.chdir("./data")

# create displacement input files
# this maybe should be a method in the TemplateProcessor class, but idk it needs InputProcessor disps
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
    with open("input.dat", 'w') as f:
        f.write(xyz)
    os.chdir("../.")
    
