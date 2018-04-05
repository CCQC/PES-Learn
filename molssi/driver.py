# a simple driver structure for generating input files over a PES 
# import itertools as it
with open("generate.py", 'r') as f:
    inputstring = f.read()

with open("template.dat", 'r') as f:
    templatestring = f.read()

inp = InputProcessor(inputstring)
template = TemplateProcessor(templatestring)

mol = Molecule(inp.zmatstring)
disps = it.combinations(inp.parameter_disps)

for disp in disps:
    cartesian_disp = mol.displace(disp)
    # create directory, move into directory
    # new = template.header + str(cartesian_disp) + template.footer
    # write input file 
    # move out of directory
