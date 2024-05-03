# PES-Learn Keyword Options 

### How to assign PES-Learn keywords...
* Using command line interface:
    *  In your PES-Learn input file, `input.dat`, all keywords are assigned the same way:  
      
        ```
        keyword1 = option1
        keyword2 = option2
        ```
    

* Using Python API:
    * In your Python code, after importing PES-Learn, you may freely dump keywords into a multi-line string and assign them just as you would when using the command line interface, and then create an InputProcessor object, which holds all of the keyword options and is passed to most other objects in PES-Learn:
    
```python
import peslearn
     
input_string =  """
                keyword1 = option1
                keyword2 = option2
                """
input_obj = peslearn.InputProcessor(input_string)
```
   * Once an `InputProcessor` object is created, new keywords can be set with the `set_keyword` method:
    
        ```python
        inp_obj = peslearn.InputProcessor("")
        inp_obj.set_keyword({'keyword':'option'})
        ```
    
    
**Note:** Some keywords must be surrounded in single `'` or double `"` quotes. These keywords all assign either regular expressions or user-specified names of files/directories for the software to use, such as the name of the dataset, the name of written electronic structure theory code input files, name of electronic structure theory code output files, etc. Rule of thumb: if its a name assignment or regex, use quotes.

**Note:** PES-Learn just looks for the keywords in your input, case-insensitive. If you spell something wrong, it will just be ignored. If you want to comment something out, use the pound sign `#` before the text you want removed.

When using command line interface `python path/to/peslearn/driver.py`, to specify which mode to run the software in, use:
* `mode = generate`, `mode = parse`, or `mode = learn`, or the corresponding shorthand `mode = g` `mode = p`, `mode = l`  
If this keyword is not used, the software will ask what you want to do.

## Data Generation Keywords (in alphabetical order)

* `energy`  
  **Description:** Energy parsing method, regular expressions, cclib, or schema. 
    * **Type**: string
    * **Default**: None
    * **Possible values**: regex, cclib, schema


* `energy_cclib`  
  **Description:** Use cclib to parse energies from output files. Takes the last occurance captured by cclib.  
    * **Type**: string
    * **Default**: None
    * **Possible values**: scfenergies, mpenergies, ccenergies

    
* `energy_regex`  
  **Description:**  Regular expression pattern which captures electronic energy from an electronic structure theory code output file. Always takes the last occuring match in the output file.  Floating point numbers `(?-\d+\.\d+)` should be surrounded by parentheses to capture just the number. It is recommended to check your regular expressions with [Pythex](https://pythex.org/). Simply copy the part of the output file you are trying to capture as well as your trial regular expression to see if it properly captures the energy. 
    * **Type**: string, surrounded by quotes
    * **Default**: None
    * **Possible values**: Any regular expression string.


* `eq_geom`  
  **Description:** Forces this one geometry (typically equilibrium geometry) into the dataset. Internal coordinates are supplied in the order they appear in the Z-Matrix of the input file.  
    * **Type**: list
    * **Default**: None
    * **Possible values**: `[1.0, 1.0, 104.5, 1.5, 120, 180]`, etc.


* `grid_reduction`  
  **Description:** Reduce the size of the internal coordinate grid to _n_ points. Acts **after** redundancy removal. Analyzes Euclidean distances between all datapoints, and creates a sub-grid of *n* geometries which are maximally far apart from one another.
    * **Type**: int
    * **Default**: None
    * **Possible values**: any integer, less than the total number of points in the internal coordinate grid after redundancies are removed.


* `input_name`  
  **Description:** The name of generated input files for electronic structure theory packages. 
    * **Type**: string, surrounded by quotes
    * **Default**: 'input.dat'
    * **Possible values**: any string


* `output_name`  
  **Description:**  The name of electronic structure theory package output (log) files which PES-Learn will attempt to parse.
    * **Type**: string, surrounded by quotes
    * **Default**: 'output.dat'
    * **Possible values**: any string  


* `pes_dir_name`  
  **Description:**  The name of the directory containing all electronic structure theory package input and/or output files. Used both when generating and parsing data.
    * **Type**: string, surrounded by quotes
    * **Default**: 'PES_data'
    * **Possible values**: any string 


* `pes_format`  
  **Description:** When parsing output file data, should PES-Learn create a dataset in terms of interatomic distances or user-supplied internal coordinates given by the Z-Matrix? 
    * **Type**: string
    * **Default**: interatomics
    * **Possible values**: interatomics, zmat 


* `pes_name`  
  **Description:**  The name of the produced dataset after parsing output files from `pes_dir_name`, as well as the name of the dataset which will be read for building ML models.
    * **Type**: string, surrounded by quotes
    * **Default**: 'PES.dat'
    * **Possible values**: any string 


* `pes_redundancy`  
  **Description:** Include all redundant geometries and assign appropriate redundant energies when creating a dataset with parsing capability. Doesn't do anything unless `remember_redundancy` was set to true when data was generated.
    * **Type**: bool
    * **Default**: false
    * **Possible values**: true, false 


* `remember_redundancy`  
  **Description:**  Remember symmetry-redundant geometries when they are removed using `remove_redundancy`. This is done so that redundant geometries can be included in the dataset created when parsing, and assigned the appropriate energy of its redundant partner whos energy was actually computed. These geometries are included when parsing only if `pes_redundancy` is set to true.
    * **Type**: bool
    * **Default**: false
    * **Possible values**: true, false


* `remove_redundancy`  
  **Description:**  Removes symmetry-redundant geometries from internal coordinate grid 
    * **Type**: bool
    * **Default**: true
    * **Possible values**: true, false


* `sort_pes`  
  **Description:** When parsing to produce a dataset, sort the energies in increasing order.  
    * **Type**: bool
    * **Default**: true
    * **Possible values**: true, false   


* `schema_basis`  
  **Description:** Any basis that can be interpreted by the quantum chemical software of choice. 
    * **Type**: string
    * **Default**: None
    * **Possible values**: any string, e.g. 'sto-3g', 'cc-pvdz', etc.


* `schema_driver`  
  **Description:** The type of computation for QCEngine to run.
    * **Type**: string
    * **Default**: 'energy'
    * **Possible values**: 'energy', 'hessian', 'gradient', 'properties'


* `schema_generate`  
  **Description:** Generate input files that will run with QCEngine to produce QCSchema outputs.  
    * **Type**: bool
    * **Default**: false
    * **Possible values**: true, false


* `schema_keywords`  
  **Description:** A python dictionary surrounded by quotes containing keywords to be used by the quantum chemical software of choice.
    * **Type**: dict, surrounted by quotes
    * **Default**: None
    * **Possible values**: any dict surrounded by quotes e.g. "{e_convergence': '1e-4', 'maxiter': '30'}"


* `schema_method`  
  **Description:** Any method that can be interpreted by the quantum chemical software of choice. 
    * **Type**: string
    * **Default**: None
    * **Possible values**: any string, e.g. 'hf', 'ccsd', etc.


* `schema_prog`  
  **Description:** The quantum chemical program to run the desired computation, must be a program supported by QCEngine. 
    * **Type**: string
    * **Default**: None
    * **Possible values**: any string e.g. 'psi4'
    

* `schema_units`  
  **Description:** The units of the provided Z-Matrix input. QCEngine expects input units of Angstroms so Bohr will be converted. 
    * **Type**: string
    * **Default**: angstrom
    * **Possible values**: bohr, angstrom


## Machine Learning Keywords (in alphabetical order)

* `gp_ard`  
  **Description:** Use auto-relevancy determination (ARD) in Gaussian process regression. If True, a length scale is optimized for each input value. If false, just one length scale is optimized.  If gp_ard = opt, it is treated as a hyperparameter. False is typically better for high-dimensional inputs (>30).
    * **Type**: bool
    * **Default**: true
    * **Possible values**: true, false, or opt (treats as hyperparameter)


* `hp_maxit`  
  **Description:** Maximum number of hyperparameter tuning iterations.  
    * **Type**: int
    * **Default**: 20
    * **Possible values**: Any positive integer


* `ml_model`  
  **Description:** Use Gaussian process regression or neural networks?  
    * **Type**: string
    * **Default**: gp
    * **Possible values**: gp, nn


* `nas_trial_layers`  
  **Description:** Neural network hidden layer structures to try out during neural architecture search. A list of lists of numbers corresponding to the hidden layers and the number of nodes in each hidden layer. Must have at least 3 hidden layer structures. E.g. `[[16,16], [32], [64,64,64]]`
    * **Type**: List
    * **Default**: See `ml/neural_network.py`
    * **Possible values**: any list of lists of positive integers


* `nn_precision`  
  **Description:**  Floating point precision for neural networks. 32 or 64 bit. For high precision use-cases, it is recommended to use 64 bit for stability and maximum fitting performance, though training is a little slower than 32 bit.
    * **Type**: int
    * **Default**: 32
    * **Possible values**: 32, 64


* `rseed`  
  **Description:**  Global random seed. Used for initializing hyperparameter optimization iterations, random training set sampling.
    * **Type**: int
    * **Default**: None
    * **Possible values**: Any integer


* `sampling`  
  **Description:** Training set sampling algorithm  
    * **Type**: string
    * **Default**: structure_based
    * **Possible values**: structure_based, smart_random, random


* `training_points`  
  **Description:** Number of training points 
    * **Type**: int
    * **Default**: 50
    * **Possible values**: any int smaller than total dataset size.


* `use_pips`  
  **Description:** Use software's library of fundamental invariant polynomials to represent the interatomic distances dataset in terms of permutation invariant polynomials. Requires that the dataset is an interatomic distance dataset produced by PES-Learn, or a properly formatted Cartesian coordinate external dataset. 
    * **Type**: bool
    * **Default**: true
    * **Possible values**: any string

   
* `validation_points`  
  **Description:** Number of validation points. Currently only used for neural networks.
    * **Type**: int
    * **Default**: Random set of half the points remaining after setting aside training set.
    * **Possible values**: Any positive integer smaller than (total dataset size - `training_points`).
    
   
* ``  
  **Description:**  
    * **Type**: 
    * **Default**: 
    * **Possible values**: 

