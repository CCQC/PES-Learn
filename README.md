# PES-Learn
[![Build Status](https://travis-ci.org/CCQC/PES-Learn.svg?branch=master)](https://travis-ci.org/CCQC/PES-Learn)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

PES-Learn is a Python library designed to fit system-specific Born-Oppenheimer potential energy surfaces using modern machine learning models. PES-Learn assists in generating datasets, and features Gaussian process and neural network model optimization routines. The goal is to provide high-performance models for a given dataset without requiring user expertise in machine learning.

This project is young and under active development. It is recommended to take a look at the Tutorial and FAQ pages before using PES-Learn for research purposes. More documentation will be added periodically. Questions and comments are encouraged; please consider submitting an issue. 

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
First, git clone with either HTTPS or SSH: `git clone https://github.com/adabbott/PES-Learn.git`

Then, change directories into the PES-Learn directory and install the required dependencies with: `python setup.py install`

Finally install peslearn locally: `pip install -e .` 

To run the test suite, you need pytest: `pip install pytest-cov` 

To run the test suite: `py.test -v tests/`

## Citing PES-Learn
The paper is currently in review, and will be posted here once published. 


## Funding 
This project is a collaboration with the [Molecular Sciences Software Institute](http://molssi.org).
The author gratefully acknowledges MolSSI for funding the development of this software.
