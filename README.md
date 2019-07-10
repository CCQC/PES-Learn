# PES-Learn
[![Build Status](https://travis-ci.org/CCQC/PES-Learn.svg?branch=master)](https://travis-ci.org/CCQC/PES-Learn)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

PES-Learn is a Python library designed to fit system-specific Born-Oppenheimer potential energy surfaces using modern machine learning models. PES-Learn assists in generating datasets, and features Gaussian process and neural network model optimization routines. The goal is to provide high-performance models for a given dataset without requiring user expertise in machine learning.

This project is young and under active development. It is recommended to take a look at the [Tutorials](1_Tutorials), the [FAQ page](2_FAQ) page, and the list of [keyword options](3_Keywords) before using PES-Learn for research purposes. More documentation will be added periodically. Questions and comments are encouraged; please consider submitting an issue. 

## Features

* **Ease of Use**
  * PES-Learn can be run by writing an input file and running the code (much like most electronic structure theory packages)
  * PES-Learn also features a Python API for more advanced workflows
  * Once ML models are finished training, PES-Learn automatically writes a Python file containing a function for evaluating the energies at new geometries. 
  
* **Data Generation**
  * PES-Learn supports input file generation and output file parsing for arbitrary electronic structure theory packages such as Psi4, Molpro, Gaussian, NWChem, etc. 
  * Data is generated with user-defined internal coordinate displacements with support for:
    * Redundant geometry removal
    * Configuration space filtering

* **Automated Data Transformation**
  * Rotation, translation, and permutation invariant molecular geometry representations

* **Automated Machine Learning Model Generation**
  * Neural network models are built using PyTorch
  * Gaussian process models are built using GPy

* **Hyperparameter Optimization**


## Installation Instructions 
PES-Learn has been tested and developed on Linux, Mac, and Windows 10 through the Windows Subsystem for Linux. To install using `pip`:   
Clone the repository:    
`git clone https://github.com/adabbott/PES-Learn.git`  
Change into top-level directory:  
`cd PES-Learn`  
Install PES-Learn and all dependencies:  
`python setup.py install`  
To avoid having to re-install the package whenever a change is made to the code, run
`pip install -e .`  
### Install using an Anaconda environment
The above procedure works just fine, however for performance and stability, we recommend installing all PES-Learn dependencies in a clean Anaconda environment. 
After installing [Anaconda for Python3](https://www.anaconda.com/distribution/), create and activate an environment:  
```conda create -n peslearn python=3.6```  
```conda activate peslearn```  
The required dependencies can be installed in one line:  
```conda install -c conda-forge -c pytorch -c omnia gpy pytorch scikit-learn pandas hyperopt cclib```   
Then install the PES-Learn package:  
`git clone https://github.com/adabbott/PES-Learn.git`   
`python setup.py install`   
`pip install -e .`  


To update PES-Learn in the future, run `git pull` while in the top-level directory `PES-Learn`.

To run the test suite, you need pytest: `pip install pytest-cov` 
To run tests, in the top-level directory called `PES-Learn`, run: `py.test -v tests/`

## Citing PES-Learn
[PES-Learn: An Open-Source Software Package for the Automated Generation of Machine Learning Models of Molecular Potential Energy Surfaces ](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00312)

Bibtex:
```
```




## Funding 
This project is a collaboration with the [Molecular Sciences Software Institute](http://molssi.org).
The author gratefully acknowledges MolSSI for funding the development of this software.
