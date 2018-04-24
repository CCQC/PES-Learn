# Data Generation

# Initial set up
For automated data generation across a PES, one only needs two files:    
    1. `input.dat`   
    2. `template.dat`  

The file `template.dat` is an input file for any electronic structure theory package that computes a single point energy or gradient.
Currently, the only constraint is that it uses an xyz-style geometry definition.

The file `input.dat` defines the molecular configuration space you wish to scan over as well as other keyword options. 
### Running the software
To run the software, simply run `python /path/to/molssi/molssi/driver.py` while in the directory containing `input.dat` and `template.dat` and follow the instructions.
### Defining a configuration space
The configuration space is specified with internal coordinates, a "Z-Matrix." 
Parameter ranges such as bond lengths, angles, and dihedrals are indicated by a bracketed list of the form `param =  [start, stop, number of points]`. 
Parameters can be fixed by setting them equal to a value instead. 
Labels for the parameters (e.g. `R1`,`ROH`,`D180`) are flexible. 
Geometries will be generated with every possible combination of parameter values.
See below for an example configuration space definition.

### Extracting energy values
The keyword `energy` dictates whether one wishes to use cclib `energy = 'cclib'` or regular expressions `energy = 'regex'` to obtain the energies from output files. 
If using cclib, one must set `energy_cclib` to the appropriate [cclib parsing variable](https://cclib.github.io/data.html), such as `energy_cclib = 'scfenergies'` for DFT and Hartree-Fock, `'mpenergies'` for perturbation theory methods, or `'ccenergies'` for coupled cluster theory methods.

If using regular expressions, a regex identifier for the energy desired from the output file is needed.  
This is assigned to the `energy_regex` keyword. 
For example, suppose an electronic structure theory package prints the energy in the following manner, `  @DF-RHF Final Energy:   -75.91652851796150`
then one could input `energy_regex = 'Final Energy:\s+(-\d+\.\d+)'`. 
Notice we use a capture group `()` to obtain the energy value.
The software will take the last match, so uniqueness of the regular expression match is not entirely crucial.
Regular expressions can be easily checked beforehand with online utilities such as [pythex](https://pythex.org/).


If one wishes to generate a portion of the potential energy surface of water, the `input.dat` file may look like the following:

```
O  
H 1 r1  
H 1 r2 2 a1  

r1 = [0.7, 1.4, 8]  
r2 = [0.7, 1.4, 8]
a1 = [90, 180, 10]

energy = 'cclib'
energy_cclib  = 'scfenergies'

```

alternatively one can use regular expressions with: 

```
energy = 'regex'
energy_regex = 'Final Energy:\s+(-\d+\.\d+)'. 
```

### Extracting gradient values
The software supports extracting cartesian gradients from output files to improve machine learning models of potential energy surfaces.
For the handful of codes which cclib supports, gradient extraction is very easy. Just add the keyword `gradient = 'cclib'` to the input file.
For everything other code, regular expressions are the only way to automate gradient extraction.
This can be tedious and messy, but it works and it's general. 
The software requires three keywords to extract a gradient:
    1. `gradient_header`
    2. `gradient_footer`
    3. `gradient_line`

The `gradient_header` is just some string that matches text that is before and close to the gradient data, and is *unique*.
The `gradient_footer` is just some string that matches text that is after and close to the gradient data, but does not need to be unique.
The `gradient_line` is a regular expression for identifying a line the of gradient, with capture groups around the x, y, and z values. 
For example, if the output file gradient is printed as 
```
    1    0.00000 -0.23410 0.32398 
    2   -0.02101 0.09233 0.01342   
    3   -0.01531 -0.04813 -0.06118
```
A valid argument for `grad_line_regex` would be `"\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"`.
Note the three capture groups corresponding to the x, y, and z floats, and the allowence of negative values `-?`

A more complicated case,
```
Atom 1 Cl 0.00000 -0.23410 0.32398 
Atom 2 H  -0.02101 0.09233 0.01342   
Atom 3 N  -0.01531 -0.04813 -0.06118
```
A valid argument for `grad_line_regex` would be `"Atom\s+\d+\s+[A-Z,a-z]+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"`


    


