"""
Test output parsing timings with different methods 
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
    energy_cclib = output_obj.extract_energy_with_cclib("mpenergies")
    # bug in cclib, prints arrays instead of floats... for mp2 but not scf
    energy_cclib = float(energy_cclib)  
    energy_regex = output_obj.extract_energy_with_regex("\s\sTotal Energy\s+=\s+(-\d+\.\d+)")
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
        energy_cclib = output_obj.extract_energy_with_cclib("mpenergies")

def energy_regex(ndirs):
    for i in range(1, ndirs+1):
        path = str(i) + "/output.dat"
        output_obj = molssi.outputfile.OutputFile(path)
        energy_regex = output_obj.extract_energy_with_regex("\s+Total Energy\s+=\s+(-\d+\.\d+)")

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


# TIMEIT RESULTS (obtained by loading this script into ipython)
# 
# First note that cclib is always the same, because cclib always parses EVERYTHING everytime its called (yeah...)
# Gradient regex is faster because it searches over a smaller portion of the file since it takes header and footer arguments
# Regex will likely become even more favorable compared to cclib as output files and the amount of parsed data gets larger 

#In [3]: %timeit energy_cclib(ndirs)
#10 loops, best of 3: 55.2 ms per loop
#
#In [4]: %timeit energy_regex(ndirs)
#100 loops, best of 3: 8.96 ms per loop
#
#In [5]: %timeit gradient_cclib(ndirs)
#10 loops, best of 3: 55.1 ms per loop
#
#In [6]: %timeit gradient_regex(ndirs)
#1000 loops, best of 3: 1.64 ms per loop

