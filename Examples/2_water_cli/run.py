import os

os.chdir("PES_data")

dirs = [i for i in os.listdir(".") if os.path.isdir(i) ]
for d in dirs:
    os.chdir(d)
    if "output.dat" not in os.listdir('.'):
        print("in" + d)
        os.system("psi4 input.dat")
    os.chdir("../")

