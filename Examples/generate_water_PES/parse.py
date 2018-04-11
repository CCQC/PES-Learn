"""
Temporary, quick and dirty driver for output parsing
"""

import sys
import os
import numpy as np
# find molssi module
sys.path.insert(0, "../../")
import molssi


os.chdir("./PES_data")

ndirs = sum(os.path.isdir(d) for d in os.listdir("."))

E_cclib = []
E_regex = []
grads_cclib = []
grads_regex = []

# try parsing in every way currently supported
for i in range(1, ndirs+1):
    path = str(i) + "/output.dat"
    output_obj = molssi.outputfile.OutputFile(path)
    energy_cclib = output_obj.extract_energy_with_cclib("scfenergies")
    energy_regex = output_obj.extract_energy_with_regex("Final Energy:\s+(-\d+\.\d+)")
    grad_cclib = output_obj.extract_cartesian_gradient_with_cclib()
    grad_regex = output_obj.extract_cartesian_gradient_with_regex(
    "Total Gradient:", "\*\*\* tstop() called on", "\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
    E_regex.append(energy_regex)
    E_cclib.append(energy_cclib)
    grads_cclib.append(grad_cclib)
    grads_regex.append(grad_regex)

# check that the energies and gradients are parsed the same    
print(np.allclose(E_regex, E_cclib))
print(np.allclose(grads_regex, grads_cclib))

# use timeit to race cclib vs regex
def energy_cclib(ndirs):
    for i in range(1, ndirs+1):
        path = str(i) + "/output.dat"
        output_obj = molssi.outputfile.OutputFile(path)
        energy_cclib = output_obj.extract_energy_with_cclib("scfenergies")

def energy_regex(ndirs):
    for i in range(1, ndirs+1):
        path = str(i) + "/output.dat"
        output_obj = molssi.outputfile.OutputFile(path)
        energy_regex = output_obj.extract_energy_with_regex("Final Energy:\s+(-\d+\.\d+)")

def gradient_cclib(ndirs):
    for i in range(1, ndirs+1):
        path = str(i) + "/output.dat"
        output_obj = molssi.outputfile.OutputFile(path)
        grad_cclib = output_obj.extract_cartesian_gradient_with_cclib()

def gradient_regex(ndirs):
    for i in range(1, ndirs+1):
        path = str(i) + "/output.dat"
        output_obj = molssi.outputfile.OutputFile(path)
        grad_regex = output_obj.extract_cartesian_gradient_with_regex(
        "Total Gradient:", "\*\*\* tstop() called on", "\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


# TIMEIT RESULTS (obtained by loading into ipython)
# 
# First note that cclib is always the same, because cclib always parses EVERYTHING everytime its called (yeah...)
# As a result, using regex searches is 80x faster for energies and 20x faster for gradients. 
# Regex will likely become even more favorable as output files and the amount of parsed data gets larger 

# %timeit energy_cclib(ndirs)
# 1 loop, best of 3: 1.32 s per loop
# 
# %timeit energy_regex(ndirs)
# 100 loops, best of 3: 16.5 ms per loop

# %timeit gradient_cclib(640)
# 1 loop, best of 3: 1.32 s per loop
# 
# %timeit gradient_regex(640)
# 10 loops, best of 3: 48.4 ms per loop



