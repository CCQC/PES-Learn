######################################################
Training models with datasets Not created by PES-Learn
######################################################

PES-Learn supports builting machine learning (ML) models from user-supplied datasets in many flexible formats.
This tutorial covers all of the different kinds of datasets which can be loaded in and used. 

***************************
**Supported Dataset Types**
***************************

Cartesian Coordinates
#####################

When PES-Learn imports Cartesian coordinate files, it re-orders the atoms to its standard ordering scheme. 
This was found to be necessary in order to enable the use of permutation invariant polynomials with externally 
supplied datasets. PES-Learn's standard atom order sorts elements by most common occurance, with an alphabetical 
tiebraker. For example, if the Cartesian coordinates of acetate (:math:`C_2H_3O_2`) were given in the order 
C,C,H,H,H,O,O, they would be automatically re-ordered to :math:`H_3C_2O_2`.

PES-Learn uses the set of interatomic distances for the geometries, which are defined to be the row-wise order 
of the interatomic distance matrix in standard order:

.. code-block::

       H   H   H   C   C   O   O
    H  
    H  r0
    H  r1  r2
    C  r3  r4  r5
    C  r6  r7  r8  r9
    O  r10 r11 r12 r13 r14
    O  r15 r16 r17 r18 r19 r20

Thus, in all the following water examples, the HOH atom order is internally reordered to HHO.

The "standard" way to express geometry, energy pairs with Cartesian coordinates is the following:

.. code-block::

    3
    -76.02075832627291
    H            0.000000000000    -0.671751442127     0.596572464600
    O           -0.000000000000     0.000000000000    -0.075178977527
    H           -0.000000000000     0.671751442127     0.596572464600

    3
    -76.0264333762269331
    H            0.000000000000    -0.727742220982     0.542307610016
    O           -0.000000000000     0.000000000000    -0.068340619196
    H           -0.000000000000     0.727742220982     0.542307610016

    3
    -76.0261926533675592
    H            0.000000000000    -0.778194442078     0.483915467021
    O           -0.000000000000     0.000000000000    -0.060982147482
    H           -0.000000000000     0.778194442078     0.483915467021

Here, there is a number indicating the number of atoms, an energy on its own line in Hartrees, 
and Cartesian coordinates in Angstroms.

Flexibility of Cartesian Coordinate input
-----------------------------------------

* The **atom number** is optional

.. code-block::

    -76.02075832627291
    H            0.000000000000    -0.671751442127     0.596572464600
    O           -0.000000000000     0.000000000000    -0.075178977527
    H           -0.000000000000     0.671751442127     0.596572464600

    -76.0264333762269331
    H            0.000000000000    -0.727742220982     0.542307610016
    O           -0.000000000000     0.000000000000    -0.068340619196
    H           -0.000000000000     0.727742220982     0.542307610016

    -76.0261926533675592
    H            0.000000000000    -0.778194442078     0.483915467021
    O           -0.000000000000     0.000000000000    -0.060982147482
    H           -0.000000000000     0.778194442078     0.483915467021

* **Blank lines** between each datablock are optional

.. code-block::

    -76.02075832627291
    H            0.000000000000    -0.671751442127     0.596572464600
    O           -0.000000000000     0.000000000000    -0.075178977527
    H           -0.000000000000     0.671751442127     0.596572464600
    -76.0264333762269331
    H            0.000000000000    -0.727742220982     0.542307610016
    O           -0.000000000000     0.000000000000    -0.068340619196
    H           -0.000000000000     0.727742220982     0.542307610016
    -76.0261926533675592
    H            0.000000000000    -0.778194442078     0.483915467021
    O           -0.000000000000     0.000000000000    -0.060982147482
    H           -0.000000000000     0.778194442078     0.483915467021

* Your **whitespace delimiters** do not matter at all, and can be completely erratic, if you're into that:

.. code-block::

    -76.02075832627291
    H   0.000000000000  -0.671751442127   0.596572464600
    O           -0.000000000000     0.000000000000    -0.075178977527
    H           -0.000000000000  0.671751442127     0.596572464600
    -76.0264333762269331
    H                     0.000000000000 -0.727742220982     0.542307610016
    O  -0.000000000000     0.000000000000    -0.068340619196
    H           -0.000000000000     0.727742220982     0.542307610016
    -76.0261926533675592
    H    0.000000000000    -0.778194442078   0.483915467021
    O      -0.000000000000  0.000000000000   -0.060982147482
    H           -0.000000000000     0.778194442078          0.483915467021

* You can use Bohr instead of Angstroms (just remember the model is trained in terms of Bohr when using it in the future!), and you can use whatever energy unit you want (though, keep in mind PES-Learn assumes it is Hartrees when converting units to wavenumbers (cm :math:`^{-1}`))

Note that you don't need to use the ``units=bohr`` keyword when training a ML model on this dataset, this keyword is for using Bohr units
when generating schemas.

Arbitrary Internal Coordinates
##############################

.. note::

    The keyword option ``use_pips`` should be set to ``false`` when using your own internal coordinates, 
    unless the coordinates correspond to the standard order PES-Learn uses for interatomic distances, described above.

For internal coordinates, the first line requires a series of geometry parameter labels, with the last column being 
the energies labeled with E. One can use internal coordinates with comma or whitespace delimiters. A few examples:



.. code-block::

    a1,r1,r2,E
    104.5,0.95,0.95,-76.026433
    123.0,0.95,0.95,-76.026193
    95.0,0.95,0.95,-76.021038

.. code-block::

    a1 r1 r2 E
    104.5 0.95 0.95 -76.026433
    123.0 0.95 0.95 -76.026193
    95.0 0.95 0.95 -76.021038

.. code-block::

    r0  r1  r2  E
    1.4554844420    0.9500000000    0.9500000000    -76.0264333762
    1.5563888842    0.9500000000    0.9500000000    -76.0261926534
    1.6454482672    0.9500000000    0.9500000000    -76.0210378425

****************************************
**Creating ML models with the datasets**
****************************************

Using an external dataset called ``dataset_name`` is the same whether it is a Cartesian coordinate or 
internal coordinate file.

With the Python API:

.. code-block:: python 

    import peslearn

    input_string = ("""
                use_pips = false
                hp_maxit = 15
                training_points = 500
                sampling = structure_based
                """)

    gp = peslearn.ml.GaussianProcess("dataset_name", input_obj)
    gp.optimize_model()

Using a Neural Network:

.. code-block:: python 

    nn = peslearn.ml.NeuralNetwork("dataset_name", input_obj)
    nn.optimize_model()

Using the commane line interface:

.. code-block::

    use_pips = false
    hp_maxit = 15
    training_points = 1000
    sampling = smart_random
    ml_model = gp
    pes_name = 'dataset_name'

Using the Python API, one can even partition and supply their own training, validation, and testing datasets:

.. code-block:: python

    nn = peslearn.ml.NeuralNetwork('full_dataset_name', input_obj, train_path='my_training_set', valid_path='my_validation_set', test_path='my_test_set')
    nn.optimize_model()