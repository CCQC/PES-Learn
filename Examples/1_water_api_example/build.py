import peslearn

input_string = ("""
               O 
               H 1 r1
               H 1 r2 2 a2 
            
               r1 = [0.85,1.20, 5]
               r2 = [0.85,1.20, 5]
               a2 = [90.0,120.0, 5]

               energy = 'regex'
               use_pips = true
               energy_regex = 'Total Energy\s+=\s+(-\d+\.\d+)'
               hp_max_evals = 15
               training_points = 40
               sampling = smart_random
               """)

input_obj = peslearn.InputProcessor(input_string)
template_obj = peslearn.datagen.Template("./template.dat")
mol = peslearn.datagen.Molecule(input_obj.zmat_string)
config = peslearn.datagen.ConfigurationSpace(mol, input_obj)
config.generate_PES(template_obj)

# run single point energies with Psi4
import os
os.chdir("PES_data")
dirs = [i for i in os.listdir(".") if os.path.isdir(i) ]
for d in dirs:
    os.chdir(d)
    if "output.dat" not in os.listdir('.'):
        print(d, end=', ')
        os.system("psi4 input.dat")
    os.chdir("../")
os.chdir("../")


print('\nParsing ab initio data...')
peslearn.utils.parsing_helper.parse(input_obj, mol)

print('\nBeginning GP optimization...')
#gp = peslearn.ml.GaussianProcess("PES.dat", input_obj, molecule=mol)
gp = peslearn.ml.gaussian_process.GaussianProcess("PES.dat", input_obj, 'A2B')
gp.optimize_model()
#gp.build_model(params =  {'morse_transform': {'morse': True, 'morse_alpha': 1.7000000000000002}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'std', 'scale_y': None})
##
#
