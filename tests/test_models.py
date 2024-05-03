"""
Test automated neural networks and Gaussian process optimizations 
"""

import peslearn
import pytest

datasets = ['tests/datafiles/H2CO.dat','tests/datafiles/H2O.dat']
mol_strings = ['A2BC', 'A2B']

input_string = ("""
               hp_maxit = 3
               training_points = 700
               rseed = 3
               use_pips = true
               sampling = structure_based
               """)
input_obj = peslearn.InputProcessor(input_string)

krr_input_string = ("""
               hp_maxit = 50
               training_points = 700
               rseed = 3
               use_pips = true
               sampling = structure_based
               """)
krr_input_obj = peslearn.InputProcessor(krr_input_string)

def test_gp():
    errors = []
    for i in range(len(datasets)):
        gp = peslearn.ml.GaussianProcess(datasets[i], input_obj, mol_strings[i])
        gp.optimize_model()
        errors.append(gp.test_error)
    # Test set error < 50 cm-1
    assert errors[0] < 50
    assert errors[1] < 50

def test_nn():
    nn = peslearn.ml.NeuralNetwork(datasets[1], input_obj, mol_strings[1])
    nn.optimize_model()
    # Test set error < 50 cm-1
    assert nn.test_error < 50

def test_krr():
    errors = []
    for i in range(len(datasets)):
        krr = peslearn.ml.KernelRidgeReg(datasets[i], krr_input_obj, mol_strings[i])
        krr.optimize_model()
        errors.append(krr.test_error)
    # Test set error < 200 cm-1
    assert errors[0] < 200
    assert errors[1] < 200


