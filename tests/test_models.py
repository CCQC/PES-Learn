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

def test_gp():
    errors = []
    for i in range(len(datasets)):
        gp = peslearn.ml.GaussianProcess(datasets[i], input_obj, mol_strings[i])
        gp.optimize_model()
        errors.append(gp.test_error)
    # Test set error < 15 cm-1
    assert errors[0] < 15
    assert errors[1] < 15

def test_nn():
    nn = peslearn.ml.NeuralNetwork(datasets[1], input_obj, mol_strings[1])
    nn.optimize_model()
    # Test set error < 15 cm-1
    assert nn.test_error < 15



