# script used to run psi4 on input files
import os
import glob

ndisps = len(glob.glob("data/*"))

os.chdir("data/")

# run psi4
for i in range(1, ndisps + 1):
    os.chdir(str(i))
    os.system("psi4 input.dat")
    os.chdir("../.")
