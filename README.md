# PES-Learn
[![Build Status](https://travis-ci.org/adabbott/PES-Learn.svg?branch=master)](https://travis-ci.org/adabbott/PES-Learn)
[![codecov](https://codecov.io/gh/adabbott/PES-Learn/branch/master/graph/badge.svg)](https://codecov.io/gh/adabbott/PES-Learn)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


PES-Learn is a Python library designed to facilitate the investigation molecular potential energy surfaces with machine learning.

This project is very young and under active development.

### Installation Instructions ### 
First, git clone with either HTTPS or SSH: `git clone https://github.com/adabbott/PES-Learn.git`

Then, change directories into the PES-Learn directory and install the required dependencies with: `python setup.py install`

Finally install peslearn locally: `pip install -e .` 

To run the test suite, you need pytest: `pip install pytest-cov` 

To run the test suite: `py.test -v tests/`



Anticipated features include:

* Data Generation for Electronic Structure Theory Software
    * Generalized input file construction over geometrical configuration spaces
    * Generalized output file parsing of energies and gradients 

* Automated Data Transformation  
    * Rotation, translation, and permutation invariant molecular geometry representations

* Local Machine Learning Algorithms
    * Neural networks 
    * Gaussian process regression
    * Kernel ridge regression

* Interfaces to External Machine Learning Algorithms

* Hyperparameter optimization


This project is a collaboration with the [Molecular Sciences Software Institute](http://molssi.org).
The author gratefully acknowledges MolSSI for funding the development of this software.
