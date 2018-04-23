# Data Generation

For automated data generation across a PES, one only needs two files:
    1. `input.dat`  
    2. `template.dat`

The file `template.dat` is an input file for any electronic structure theory package that computes a single point energy or gradient.
Currently, the only constraint is that it uses an xyz-style geometry definition.

The file `input.dat` defines the molecular configuration space you wish to scan over as well as other keyword options. 
The configuration space is specified with internal coordinates, a "Z-Matrix." 
Parameter ranges such as bond lengths, angles, and dihedrals are indicated by a bracketed list of the form `param =  [start, stop, number of points]`. 
Parameters can be fixed by setting them equal to a value instead. 
Labels for the parameters (e.g. `R1`,`ROH`,`D180`) are flexible. 
Geometries will be generated with every possible combination of parameters.

The keyword `extract` dictates whether one wishes to use cclib `extract = cclib` or regular expressions `extract = regex` to obtain the energies and gradients from output files. 
The keyword `energy` specifies which energy to parse. 
If using cclib, the appropriate [cclib parsing variable must be used](https://cclib.github.io/data.html), such as `scfenergies` for DFT and Hartree-Fock, `mpenergies` for MP2, and `ccenergies` for coupled cluster.
If using regular expressions, a regex identifier for the energy desired from the output file is needed. 
For example, if a electronic structure theory package prints the energy in the following manner, `  @DF-RHF Final Energy:   -75.91652851796150`
then one could input `energy = 'Final Energy:\s+(-\d+\.\d+)'`. 
   Notice we use a capture group `()` to obtain the energy value.
The software will take the last match, so uniqueness is only important for the tail end of the output (log) file.
Regular expressions can be easily checked beforehand with online utilities such as [pythex](https://pythex.org/).


If one wishes to generate a portion of the potential energy surface of water, the `input.dat` file may look like the following:


```
O  
H 1 r1  
H 1 r2 2 a1  

r1 = [0.7, 1.4, 8]  
r2 = [0.7, 1.4, 8]
a1 = [90, 180, 10]

extract = 'cclib'
energy  = 'scfenergies'

```
