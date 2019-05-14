# PES-Learn Command-Line Interface Tutorial

PES-Learn is designed to work similarly to standard electronic structure theory packages: users can generate an input file with appropriate keywords, run the software, and get a result. This tutorial covers how to do exactly that.
Here we generate a machine-learning model of the PES of water from start to finish (no knowledge of Python required!).

## 1. Generating Data

### 1.1 Defining an internal coordinate grid
Currently PES-Learn supports generating points across PESs by displacing in simple internal coordinates (a 'Z-Matrix'). To do this, we must define a Z-Matrix in the input file. We first create an input file called `input.dat`:

```console
home:~$ vi input.dat
```

in the input file we define the Z-Matrix and the displacements:
```python
O
H 1 r1
H 1 r2 2 a1

r1 = [0.85, 1.30, 10]
r2 = [0.85, 1.30, 10]
a1 = [90.0, 120.0, 10] 
```

The syntax defining the internal coordinate ranges is of the form [start, stop, number of points], with the bounds included in the number of points. The angles and dihedrals are always specified in degrees. The units of length can be anything, though typically Angstrom or Bohr. Dummy atoms are supported (and in fact, are required if there are 3 or more co-linear atoms, otherwise in that case those internal coordinate configurations will just be deleted!). Labels for geometry parameters can be anything (RDUM, ROH1, A120, etc) as long as they do not start with a number. Parameters can be fixed with `r1 = 1.0`, etc. An equilibruim geometry can be specified in the order the internal coordinates appear with
```python
eq_geom = [0.96,0.96,104.5]
```
and this will also be included. 

### 1.2 Creating a Template file

Normally we would go on to build our input file by adding keywords controlling the program, but first let's talk about **template input files**. A template input file is a file named `template.dat` which is a **cartesian coordinate input file for an electronic structure theory package** such as Gaussian, Molpro, Psi4, CFOUR, QChem, NWChem, and so on. It does not matter what package you want to use, it only matters that the `template.dat` contains Cartesian coordinates, and computes an electronic energy by whatever means you wish. PES-Learn will use the template file to generate a bunch of (Guassian, Molpro, Psi4, etc) input files, each with different Cartesian geometries corresponding to the above internal coordinate grid. The template input file we will use in this example is a Psi4 input file which computes a CCSD(T)/cc-pvdz energy:
```python
molecule h2o {
0 1
H 0.00 0.00 0.00
H 0.00 0.00 0.00
O 0.00 0.00 0.00
}

set {
reference rhf
basis cc-pvdz
}
energy('ccsd(t)')
```

The actual contents of the Cartesian coordinates does not matter. Later on when we run the code, the auto-generated input files with Cartesian geometries corresponding to our internal coordinate grid will be put into their own newly-created sub-directories likes this:
```
PES_data/1/
PES_data/2/
PES_data/3/
...
```

This PES_data folder can then be zipped up and sent to whatever computing resources you want to use.

### 1.3 Data Generation Keywords

Let's go back to our PES-Learn input file, add a few keywords, and discuss them.
```python
O
H 1 r1
H 1 r2 2 a1

r1 = [0.85, 1.30, 10]
r2 = [0.85, 1.30, 10]
a1 = [90.0, 120.0, 10] 

# Data generation-relevant keywords
eq_geom = [0.96,0.96,104.5]
input_name = 'input.dat'
remove_redundancy = true
remember_redundancy = false
grid_reduction = 300
```
Comments (ignored text) can be specified with a `#` sign. All entries are case-insensitive. Multiple word phrases are seperated with an underscore. Text that doesn't match any keywords is simply ignored (in this way, the use of comment lines is really not necessary unless your are commenting out keyword options). **This means if you spell a keyword or its value incorrectly it will be ignored**. The first occurance of a keyword will be used.

* We discussed **`eq_geom`** before, it is a geometry forced into the dataset, and it would typically correspond to the global minimum at the level of theory you are using. It is often a good idea to create your dataset such that the minimum of the dataset is the true minimum of the surface, especially for vibrational levels applications. 
* **`input_name`** tells PES-Learn what to call the electronic structure theory input files. `'input.dat'` is the default value, no need to set it normally.  Note that it is surrounded in quotes; this is so PES-Learn doesn't touch it or change anything about it, such as lowering the case of all the letters. 

* **`remove_redundancy`** removes symmetry-redundant geometries from the internal coordinate grid. In this case, there is redundancy in the equivalent OH bonds and they will be removed. 

* **`remember_redundancy`** keeps a cache of redundant-geometry pairs, so that when the energies are parsed from the output files and the dataset is created later on, all of the original geometries are kept in the dataset, with duplicate energies for redundant geometries. If one does not use a permutation-invariant geometry for ML later, this may be useful.

* **`grid_reduction`** reduces the grid size to the value entered. In this case it means only 300 geometries will be created. This is done by finding the Euclidean distances between all the points in the dataset, and extracting a maximally spaced 'sub-grid' of the size specified. 

### 1.4 Running the code and generating the data

In the directory containing the PES-Learn input file `input.dat` and `template.dat`, simply run 
```console
home:~$ python path/to/PES-Learn/peslearn/driver.py
```
The code will then ask what you want to do, here we type `g` or `generate` and hit enter, and this is the output:
```
Do you want to 'generate' data, 'parse' data, or 'learn'?g

1000 internal coordinate displacements generated in 0.00741 seconds
Total displacements: 1001
Number of interatomic distances: 3
Geometry grid generated in 0.06 seconds
Removing symmetry-redundant geometries...  Redundancy removal took 0.01 seconds
Removed 450 redundant geometries from a set of 1001 geometries
Reducing size of configuration space from 551 datapoints to 300 datapoints
Configuration space reduction complete in 0.05 seconds
Your PES inputs are now generated. Run the jobs in the PES_data directory and then parse.
Data generation finished in 0.41 seconds
Total run time: 0.41 seconds
```

The 300 Psi4 input files with Cartesian coordinates corresponding to our internal coordinate grid are now placed into a directory called `PES_data`:
```
PES_data/1/
PES_data/2/
PES_data/3/
...
```

**Quick note:** you do not have to use the command line to specify whether you want to `generate`, `parse`, or `learn`, you can instead specify the `mode` keyword in the input file:
```python
mode = generate # or 'parse' or 'learn', or shorthand: 'g', 'p', 'l'
```
this is at times convenient if computations are submitted remotely in an automated fashion, and the users are not directly interacting with a command line.

We are now ready run the Psi4 energy computations.

## 2. Parsing Output files to collect electronic energies

Now that every Psi4 input file has been run, and there is a corresponding `output.dat` in each sub-directory of `PES_data`, we are ready to use PES-Learn to grab all of the energies, match them with the appropriate geometries, and create a dataset. 

There are two schemes for parsing output files with PES-Learn:
  * User-supplied Python regular expressions
  * cclib
  
**Regular expressions** are a pattern-matching syntax. Though they are somewhat tedious to use, they are completely general. Using the regular expression scheme requires
  1. Inspecting the electronic structure theory software output file
  2. Finding the line where the desired energy is 
  3. Writing a regular expression to match the line's text and grab the desired energy.
    
**cclib** is a Python library of hard-coded parsing routines. It works in a lot of cases. At the time of writing, cclib supports parsing `scfenergies`, ``mpenergies``, and `ccenergies`. These different modes attempt to find the highest level of theory SCF energy (Hartree-Fock or DFT), highest level of Moller-Plesset perturbation theory energy, or the highest level of theory coupled cluster energy. Since these are hard-coded routines that are version-dependent, there is no gurantee it will work! It is also a bit slower than regular expressions (i.e. milliseconds --> seconds slower)


### 2.1 Setting the appropriate parsing keywords in the PES-Learn input file

It is often a good idea to take a look at a successful output file in `PES_data/`. Here is the output file in `PES_data/1/`, which is the geometry corresponding to `eq_geom` defined earlier:

```
            **************************
            *                        *
            *        CCTRIPLES       *
            *                        *
            **************************


    Wave function   =    CCSD_T
    Reference wfn   =      RHF

    Nuclear Rep. energy (wfn)                =    9.168193296244223
    SCF energy          (wfn)                =  -76.026653661887252
    Reference energy    (file100)            =  -76.026653661887366
    CCSD energy         (file100)            =   -0.213480496782495
    Total CCSD energy   (file100)            =  -76.240134158669861

    Number of ijk index combinations:               35
    Memory available in words        :        65536000
    ~Words needed per explicit thread:            2048
    Number of threads for explicit ijk threading:    1

    MKL num_threads set to 1 for explicit threading.

    (T) energy                                =   -0.003068821713392
      * CCSD(T) total energy                  =  -76.243202980383259


    Psi4 stopped on: Thursday, 09 May 2019 01:51PM
    Psi4 wall time for execution: 0:00:01.05

*** Psi4 exiting successfully. Buy a developer a beer!
```

If we were to use cclib, we would put into our PES-Learn `input.dat` file:
```python
# Parsing-relevant keywords
energy = cclib
energy_cclib = ccenergies
```
to grab coupled cluster energies. **Unfortunately, at the time of writing, this only grabbed the CCSD energies and not the CCSD(T) energies (it's a good idea to always check**). Let's use regular expressions instead.

### 2.1.1 Regular expressions

One fact is always very important to keep in mind when using regular expressions in PES-Learn:  
**PES-Learn always grabs the last matching entry in the output file**  
This is good to know, since a pattern may match multiple entries in the output file, but it's okay as long as you want the *last one*.

We observe that the energy we want is always contained in a line like 
```
      * CCSD(T) total energy                  =  -76.243202980383259
```

So the general pattern we want to match is `total energy` (whitespace) `=` (whitespace) (negative floating point number). We may put into our PES-Learn input file the following regular expression:
```python
# Parsing-relevant keywords
energy = regex
energy_regex = 'total energy\s+=\s+(-\d+\.\d+)'
```

Here we have taken advantage of the fact that the pattern `total energy` does not appear anymore after the CCSD(T) energy in the output file. The above `energy_regex` line matches the words 'total energy' followed by one or more whitespaces `\s+`, an equal sign `=`, one or more whitespaces `\s+`, and then a negative floating point number `-\d+\.\d+` which we have necessarily enclosed in parentheses to indicate that we only want to capture the number itself, not the whole line. This is a bit cumbersome to use, so if this in foreign to you I recommend trying out various regular expressions via trial and error using Pythex https://pythex.org/ to ensure that the pattern is matched.

A few other valid `enegy_regex` lines would be
```python
energy_regex = 'CCSD\(T\) total energy\s+=\s+(-\d+\.\d+)'
```
```python
energy_regex = '=\s+(-\d+\.\d+)'
```
Note above that we had to "escape" the parentheses with backward slashes [since it is a reserved character.](https://www.debuggex.com/cheatsheet/regex/python)
If you want to be safe from parsing the wrong energy, more verbose is probably better.

### 2.2 Setting up the input file
Here we have added our parsing keywords to our PES-Learn input file. (We could have had these keywords defined earlier as well, but to keep things simple I am only adding them when needed.)
```python
O
H 1 r1
H 1 r2 2 a1

r1 = [0.85, 1.30, 10]
r2 = [0.85, 1.30, 10]
a1 = [90.0, 120.0, 10] 

# Data generation-relevant keywords
eq_geom = [0.96,0.96,104.5]
input_name = 'input.dat'
remove_redundancy = true
remember_redundancy = false
grid_reduction = 300

# Parsing-relevant keywords
energy = regex
energy_regex = 'total energy\s+=\s+(-\d+\.\d+)'
pes_name = 'PES.dat'
sort_pes = true           # sort in terms of increasing energy
pes_format = interatomics # could also choose internal coordinates r1, r2, a1
```

### 2.3 Parsing the output files and creating a dataset
Just as before, we run PES-Learn, and this time choose `parse` by typing `p` and hitting enter:
```console
home:~$ python path/to/PES-Learn/peslearn/driver.py
Do you want to 'generate' data, 'parse' data, or 'learn'?p
Parsed data has been written to PES.dat
Total run time: 0.38 seconds
```

The dataset `PES.dat` looks like this:
```python
r0,r1,r2,E
1.518123981600,0.960000000000,0.960000000000,-76.243202980383
1.455484441900,0.950000000000,0.950000000000,-76.242743191056
1.494132369500,1.000000000000,0.950000000000,-76.242037809799
1.568831329800,1.000000000000,1.000000000000,-76.241196021922
1.494050142500,1.000000000000,1.000000000000,-76.240995054410
```


## 3. Creating Auto-Generated Machine Learning Models of the Potential Energy Surface

### 3.1 Gaussian Process Regression

We now have in our working directory a file called `PES.dat`, created with the routine above. An auto-optimized machine learning model of the surface can be produced by this dataset. Below we have added keywords to our PES-Learn input file which are relevant to training a ML model


```python
O
H 1 r1
H 1 r2 2 a1

r1 = [0.85, 1.30, 10]
r2 = [0.85, 1.30, 10]
a1 = [90.0, 120.0, 10] 

# Data generation-relevant keywords
eq_geom = [0.96,0.96,104.5]
input_name = 'input.dat'
remove_redundancy = true
remember_redundancy = false
grid_reduction = 300

# Parsing-relevant keywords
energy = regex
energy_regex = 'total energy\s+=\s+(-\d+\.\d+)'
pes_name = 'PES.dat'
sort_pes = true           # sort in terms of increasing energy
pes_format = interatomics # could also choose internal coordinates r1, r2, a1

# ML-relevant keywords
ml_model = gp              # Use Gaussian Process regression
pes_format = interatomics  # Geometry values in the PES file
use_pips = true            # Transform interatomic distances into permutation invariant polynomials
hp_maxit = 15              # Train 15 models with hyperparameter optimization, select the best
training_points = 200      # Train with 200 points (out of 300 total)
sampling = structure_based # Sample training set by maximizing Euclidean distances
n_low_energy_train = 1     # Force lowest energy point into training set
```

We note that a **minimal working input file** would only need the internal coordinate definition (because we are using PIPs and need to know the atom types!) and the ML relevant keywords:

```python
O
H 1 r1
H 1 r2 2 a1

# ML-relevant keywords
ml_model = gp              # Use Gaussian Process regression
pes_format = interatomics  # Geometry values in the PES file
use_pips = true            # Transform interatomic distances into permutation invariant polynomials
hp_maxit = 15              # Train 15 models with hyperparameter optimization, select the best
training_points = 200      # Train with 200 points (out of 300 total)
sampling = structure_based # Sample training set by maximizing Euclidean distances
n_low_energy_train = 1     # Force lowest energy point into training set
```

running the following tries out several models and prints the performance statistics in units of wavenumbers (cm$^{-1}$):

<details>
  <summary>Click to expand and see all hyperparameter iterations</summary>
  
```console
home:~$ python path/to/PES-Learn/peslearn/driver.py
Do you want to 'generate' data, 'parse' data, or 'learn'?l
Using permutation invariant polynomial transformation for molecule type  A2B
Beginning hyperparameter optimization...
Trying 15 combinations of hyperparameters
Training with 200 points (Full dataset contains 300 points).
Using structure_based training set point sampling.
Errors are root-mean-square error in wavenumbers (cm-1)
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'mm01', 'scale_y': None}
Test Dataset 5.38
Full Dataset 5.36
Median error: 4.18
Max 5 errors: [11.3 11.4 11.9 12.7 13.9]
Hyperparameters: 
{'morse_transform': {'morse': True, 'morse_alpha': 2.0}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': 'mm11', 'scale_y': 'mm11'}
Test Dataset 0.94
Full Dataset 0.77
Median error: 0.53
Max 5 errors: [1.9 2.  2.  2.  2.2]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': 'mm11', 'scale_y': 'mm11'}
Test Dataset 0.55
Full Dataset 0.51
Median error: 0.33
Max 5 errors: [1.2 1.3 1.3 1.5 1.8]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'mm01', 'scale_y': 'std'}
Test Dataset 0.52
Full Dataset 0.42
Median error: 0.26
Max 5 errors: [1.1 1.1 1.1 1.2 1.2]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'std', 'scale_y': None}
Test Dataset 5.38
Full Dataset 5.36
Median error: 4.17
Max 5 errors: [11.4 11.5 11.9 12.7 13.9]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': 'std', 'scale_y': 'mm01'}
Test Dataset 0.86
Full Dataset 0.81
Median error: 0.52
Max 5 errors: [2.  2.  2.2 2.3 3. ]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'mm11', 'scale_y': 'mm11'}
Test Dataset 0.54
Full Dataset 0.49
Median error: 0.35
Max 5 errors: [1.2 1.2 1.2 1.2 1.4]
Hyperparameters: 
{'morse_transform': {'morse': True, 'morse_alpha': 1.2000000000000002}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': 'mm01', 'scale_y': None}
Test Dataset 9.84
Full Dataset 8.51
Median error: 6.29
Max 5 errors: [19.4 20.4 20.5 21.1 25.2]
Hyperparameters: 
{'morse_transform': {'morse': True, 'morse_alpha': 1.8}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': None, 'scale_y': 'std'}
Test Dataset 0.28
Full Dataset 0.24
Median error: 0.15
Max 5 errors: [0.7 0.8 0.8 1.2 1.4]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': None, 'scale_y': 'mm01'}
Test Dataset 0.91
Full Dataset 0.87
Median error: 0.57
Max 5 errors: [2.2 2.2 2.3 2.4 2.8]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': None, 'scale_y': 'mm01'}
Test Dataset 0.97
Full Dataset 0.9
Median error: 0.61
Max 5 errors: [2.2 2.3 2.4 2.5 2.8]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'mm11', 'scale_y': None}
Test Dataset 5.38
Full Dataset 5.37
Median error: 4.15
Max 5 errors: [11.4 11.5 12.  12.7 13.9]
Hyperparameters: 
{'morse_transform': {'morse': True, 'morse_alpha': 1.3}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'mm01', 'scale_y': 'mm01'}
Test Dataset 1.37
Full Dataset 1.06
Median error: 0.57
Max 5 errors: [3.2 3.2 3.3 3.5 3.7]
Hyperparameters: 
{'morse_transform': {'morse': False}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': 'mm11', 'scale_y': None}
Test Dataset 5.7
Full Dataset 5.68
Median error: 4.33
Max 5 errors: [13.  13.8 14.7 14.7 15.1]
Hyperparameters: 
{'morse_transform': {'morse': True, 'morse_alpha': 1.5}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': 'mm01', 'scale_y': 'mm01'}
Test Dataset 1.3
Full Dataset 1.03
Median error: 0.65
Max 5 errors: [2.9 2.9 3.  3.  3.1]

###################################################
#                                                 #
#     Hyperparameter Optimization Complete!!!     #
#                                                 #
###################################################

Best performing hyperparameters are:
[('morse_transform', {'morse': True, 'morse_alpha': 1.8}), ('pip', {'degree_reduction': True, 'pip': True}), ('scale_X', None), ('scale_y', 'std')]
Fine-tuning final model architecture...
Hyperparameters:  {'morse_transform': {'morse': True, 'morse_alpha': 1.8}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': None, 'scale_y': 'std'}
Final model performance (cm-1):
Test Dataset 0.28  Full Dataset 0.24     Median error: 0.15  Max 5 errors: [0.7 0.8 0.8 1.2 1.4] 

Saving ML model data...
Total run time: 66.25 seconds
```
</details>

  
  
Training with just 200 points, the best model had a RMSE on the 100-point test set of 0.28 cm$^{-1}$, and the full 300 point dataset had a RMSE of 0.24 cm$^{-1}$. This is absurdly accurate; it's a good thing we used `grid_reduce` back when generating our data to reduce our dataset from 551 points to just 300! We clearly did not need more than a few hundred points to model this portion of the PES of water; any more computations would have been unnecessary! This is why it is important to probe how much data one needs along the surface at a **meaningful but low level of theory**. 

### 3.2 Using the GP model

After running the above, PES-Learn creates a directory `model1data` (subsequent calls will not overwrite this, but instead create `model2data`, `model3data`, etc.). Inside this directory is a variety of files which are self-explanatory.  

The most important file is the auto-generated Python script `compute_energy.py` which can be used to evaluate new energies using the model. It needs to be in the same directory as `PES.dat` and `model.json` to work. It contains a function `pes()` which takes one or more cartesian or internal coordinate arguments and outputs one or more energies corresponding to the geometries. If the argument `cartesian=False` is set, you must supply coordinates in the exact same format and exact same order as the model was trained on (i.e. the format in `PES.dat`). If the argument `cartesian=True` is set, cartesian coordinates are supplied in the same order as given in a typical PES_data input file (not the template.dat file). **Cartesians can only be supplied if the model was trained on interatomic distances or PIPs of the interatomic distances.**

The `compute_energy.py` file can be imported and used. Here's an example python script `use_model.py` which imports the `pes` function and evaluates some energies at some cartesian geometries.

```python
from compute_energy import pes

cart_geoms = [[0.0000000000, 0.0000000000, 1.1000000000, 0.0000000000, 0.7361215932, -0.4250000000, 0.0000000000, 0.0000000000, 0.0000000000],
              [0.0000000000, 0.2000000000, 1.2000000000, 0.0000000000, 0.7461215932, -0.4150000000, 0.0000000000, 0.0000000000, 0.1000000000],
              [0.0000000000, 0.1000000000, 1.3000000000, 0.0000000000, 0.7561215932, -0.4350000000, 0.0000000000, 0.0000000000, 0.2000000000]]

energies1 = pes(cart_geoms)
print(energies1)

interatomic_geoms = [[1.494050142500,1.000000000000,1.000000000000],
                     [1.597603916000,1.000000000000,0.950000000000],
                     [1.418793563200,1.000000000000,0.950000000000]]

energies2 = pes(interatomic_geoms, cartesian=False)
print(energies2)
```
The print statements yield the following output. The energy is in units of Hartrees (which are the unit of energy in our PES.dat which the model was trained on).

```
[[-76.20462724]
 [-76.21835841]
 [-76.21467994]]
[[-76.24099496]
 [-76.24031118]
 [-76.24024971]]
```


### 3.3 Neural Network Regression

Todo!
