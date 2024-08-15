######################################################
PES-Learn Application Program Interface (CLI) Tutorial
######################################################

The following tutorial goes over an example of how to use PES-Learn with Python API. This allows the user to
import the peslearn python package and use python to excecute instead of the peslearn driver as with the CLI
option. Some of the differences between ML models and parsing methods are left out of this tutorial. For a 
more detailed description on these differences, check out the `CLI <cli.html>`_ tutorial.

First, import the PES-Learn python package ``peslearn``: 

.. note::

    Check out the `Installation <../started/installation.html>`_ guide for information on how to install the 
    ``peslearn`` package.

.. code-block:: python

    import peslearn

Here we will generate a simple potential energy surface of water. First, we need to create an input object 
which contains information such as the grid of geometries we wish to generate and the various keyword options. 
The input object is initialized from a string. Here we use a multiline string with triple quotes. Anything can 
go in this multiline string; only text patterns which match PES-Learn keyword options will be considered. Because 
of this, if a keyword is spelled wrong it will be ignored. We choose to scan over the OH bond distances from 0.85 
to 1.2 angstroms and the bond angle from 90 to 120 degrees. Also, we will remove redundant geometries arising from 
equivalent values of r1 and r2.

.. code-block:: python 

    input_string = ("""
    O 
    H 1 r1
    H 1 r2 2 a2 

    r1 = [0.85,1.20, 5]
    r2 = [0.85,1.20, 5]
    a2 = [90.0,120.0, 5]

    remove_redundancy = true
    input_name = input.dat
                    """)

The ``input_name`` value tells PES-Learn what to call the produced input files based on our template file ``template.dat``. 
Alternatively, schema can be used in the same way as described in the `CLI <cli.html>`_ tutorial, with all of the necessary 
keywords in the ``input_string``. In this example, however, we will be using a template.dat, which will be a Psi4 input 
file that computes a density-fitted MP2 energy with a 6-31g basis set. The only relevant part of this file from 
PES-Learn's perspective is the Cartesian coordinates, which will be found and replaced by the Cartesian coordinates 
corresponding to the internal coordinate grid given above to create a series of input files. The example ``template.dat`` 
file looks like this:

.. code-block:: 

    molecule h2o {
    0 1
    H 0.00 0.00 0.00
    H 0.00 0.00 0.00
    O 0.00 0.00 0.00
    }

    set basis 6-31g
    energy('mp2')

We instantiate a PES-Learn InputProcessor object with the input string given above, as well as a template object with 
the template file ``template.dat``:

.. code-block:: python

    input_object = peslearn.InputProcessor(input_string)
    template_object = peslearn.datagen.Template("./template.dat")

If you are using schemas, ``template_object`` does not need to be instantiated. The input_object holds the Z-Matrix 
connectivity information as well as the internal coordinate ranges and keyword options. A PES-Learn Molecule object 
takes in this Z-Matrix information and derives obtains a bunch of information about the molecule, and is able to 
update its internal coordinates and convert to Cartesian coordinates.

.. code-block:: python

    molecule_object = peslearn.datagen.Molecule(input_object.zmat_string)

We are now ready to generate all of the Psi4 input files for these geometries. To do this, we create a ``ConfigurationSpace`` 
object. This object actually creates all of the internal coordinate displacements, uses the Molecule object to obtain 
Cartesian coordinates, and also finds the interatomic distances. Each of the coordinate representations are kept in a 
pandas DataFrame. The interatomic distances are used to remove redundant geometries using permutational symmetry of 
the two identical hydrogen atoms. The ``generate_PES`` method of the Configuration space object creates a directory 
``PES_data`` containing a series of subdirectories with integer values ``1``, ``2``, ``3``... which each contain a 
unique cartesian coordinate Psi4 input file across the PES of water.

.. code-block:: python

    config = peslearn.datagen.ConfigurationSpace(molecule_object, input_object)
    config.generate_PES(template_object)

.. note::

    If you are using schemas instead of template objects, you can instead use the following line instead of the last one:

    .. code-block:: python

        config.generate_PES(template_object=None, schema_gen=true)

Excecution of ``config.generate_PES()`` will generate input files (or python scripts with schemas) in a directory ``PES_data``,
along with the following output:

.. code-block::

    125 internal coordinate displacements generated in 0.00139 seconds
    Total displacements: 125
    Number of interatomic distances: 3
    Geometry grid generated in 0.01 seconds
    Removing symmetry-redundant geometries...  Redundancy removal took 0.01 seconds
    Removed 50 redundant geometries from a set of 125 geometries
    Your PES inputs are now generated. Run the jobs in the PES_data directory and then parse.

We see here that 50 redundant geometries corresponding to identical molecular configurations were removed, 
leaving just 75 energies needed to be explicitly computed. We proceed to compute the energies with Psi4 by 
moving to each subdirectory created by PES-Learn, ``PES_data/1``, ``PES_data/2`` ..., and running Psi4 with 
the command line:

.. code-block:: python

    import os
    os.chdir('PES_data')
    for i in range(1,76):
        os.chdir(str(i))
        if "output.dat" not in os.listdir('.'):
            print(i, end=', ')
            os.system('psi4 input.dat')
        os.chdir('../')
    os.chdir("../")

.. note::

    See the `Command line tutorial <cli.html>`_ for tips on doing this with schemas.

Once the computations are complete, we wish to create a dataset of geometry, energy pairs for creating a 
machine learning model of the potential energy surface. To do this, we use the parsing capabilities of PES-Learn 
to extract the energies from the Psi4 output files. There are three schemes for doing this: exctracting from schemas,
regular expressions, and cclib. In this case, for my version of Psi4, cclib does not work for parsing MP2 energies. 
Luckily we can use the general regular expression scheme. We first need to come up with a regular expression pattern 
which matches the energy we want from the Psi4 output file. We observe that the MP2 energy in an output file is 
printed as follows:

.. code-block::

     ==================> DF-MP2 Energies <====================
    -----------------------------------------------------------
     Reference Energy          =     -75.9381224063424440 [Eh]
     Singles Energy            =      -0.0000000000000000 [Eh]
     Same-Spin Energy          =      -0.0277202185419175 [Eh]
     Opposite-Spin Energy      =      -0.0919994716794230 [Eh]
     Correlation Energy        =      -0.1197196902213406 [Eh]
     Total Energy              =     -76.0578420965637889 [Eh]


A regular expression which grabs the energy we want is ``Total Energy\s+=\s+(-\d+\.\d+)`` which matches the words 
'Total Energy' followed by one or more whitespaces ``\s+``, an equal sign ``=``, one or more whitespaces ``\s+``, 
and then a negative floating point number ``-\d+\.\d+`` which we have necessarily enclosed in parentheses to indicate 
that we only want to capture the number itself, not the whole line. This is a bit cumbersome to use, so in practice 
it is recommend trying out various regular expressions via trial and error using `Regex101 <https://regex101.com/>`_  
or `Pythex <https://pythex.org/>`_ to ensure that the pattern is matched. In the context of PES-Learn, we would set 
the following keywords in the input:

.. code-block::

    energy = 'regex'
    energy_regex = 'Total Energy\s+=\s+(-\d+\.\d+)'

However, Psi4 will print out this same line 'Total Energy = (float)' for the Hartree-Fock, MP2, SCS-MP2,
and all other energies:

.. code-block::

        @DF-RHF Final Energy:   -75.93812240634244

        => Energetics <=

        Nuclear Repulsion Energy =             10.4012001939225183
        One-Electron Energy =                -124.9779212700375410
        Two-Electron Energy =                  38.6385986697725912
        Total Energy =                        -75.9381224063424298
    ...
    ...
    ...
        ==================> DF-MP2 Energies <====================
        -----------------------------------------------------------
        Reference Energy          =     -75.9381224063424440 [Eh]
        Singles Energy            =      -0.0000000000000000 [Eh]
        Same-Spin Energy          =      -0.0277202185419175 [Eh]
        Opposite-Spin Energy      =      -0.0919994716794230 [Eh]
        Correlation Energy        =      -0.1197196902213406 [Eh]
        Total Energy              =     -76.0578420965637889 [Eh]
        -----------------------------------------------------------
        ================> DF-SCS-MP2 Energies <==================
        -----------------------------------------------------------
        SCS Same-Spin Scale       =       0.3333333333333333 [-]
        SCS Opposite-Spin Scale   =       1.2000000000000000 [-]
        SCS Same-Spin Energy      =      -0.0092400728473058 [Eh]
        SCS Opposite-Spin Energy  =      -0.1103993660153076 [Eh]
        SCS Correlation Energy    =      -0.1196394388626135 [Eh]
        SCS Total Energy          =     -76.0577618452050643 [Eh]
        -----------------------------------------------------------

We note that PES-Learn by default takes the *last match occurance* of the regex pattern as the energy.
Thus, the Hartree-Fock line is not relavent as it occurs earlier. However, with our above regex we will 
accidentally match the 'SCS Total Energy' line. To fix this, we just input some spaces before the word 
'Total' to ensure the correct energy is matched. Using the set_keyword method, we can directly modify 
our input_object with the new parsing-relevant keywords. We note here that these could have just as 
easily been included at the very beginning in our multi-line input string, but this method is valid as well:

.. code-block:: python

    input_object.set_keyword({'energy':'regex'})
    input_object.set_keyword({'energy_regex':'\s+Total Energy\s+=\s+(-\d+\.\d+)'})

Now at the begining of our regex we have ``\s+`` which will look for any amount of whitespace before 
'Total Energy'. We will also choose to create a PES file using interatomic distances as the geometry 
representation instead of the internal coordinates. The reason is because we plan to use a permutation-invariant 
geometry representation when we do machine learning, and this requires the interatomic distances format.

.. code-block:: python

    input_object.set_keyword({'pes_format':'interatomics'})

After a bit of work, we are ready to parse the output files and create the dataset, which si a simple csv file.

.. code-block:: python

    peslearn.utils.parsing_helper.parse(input_object, molecule_object)

Let's take a look at this dataset with the Python module pandas:

.. code-block:: python 

    import pandas as pd
    data = pd.read_csv('PES.dat')
    print(data)

.. code-block:: 

              r0      r1      r2          E
    0   1.559006  0.9375  0.9375 -75.985033
    1   1.487538  0.9375  0.9375 -75.983615
    2   1.623798  0.9375  0.9375 -75.983490
    3   1.632483  1.0250  0.9375 -75.979965
    4   1.557867  1.0250  0.9375 -75.979436
    5   1.409700  0.9375  0.9375 -75.978781
    6   1.700138  1.0250  0.9375 -75.977620
    7   1.476613  1.0250  0.9375 -75.975593
    8   1.626374  1.0250  1.0250 -75.975385
    9   1.704513  1.0250  1.0250 -75.975163
    10  1.541272  1.0250  1.0250 -75.972390
    11  1.775352  1.0250  1.0250 -75.972145
    12  1.487047  0.9375  0.8500 -75.971909
    13  1.548639  0.9375  0.8500 -75.971168
    14  1.325825  0.9375  0.9375 -75.970199
    15  1.419119  0.9375  0.8500 -75.969622
    16  1.389076  1.0250  0.9375 -75.968116
    17  1.562034  1.0250  0.8500 -75.966560
    18  1.449569  1.0250  1.0250 -75.965868
    19  1.629860  1.1125  0.9375 -75.965474
    20  1.491347  1.0250  0.8500 -75.965295
    21  1.707283  1.1125  0.9375 -75.965147
    22  1.626153  1.0250  0.8500 -75.964895
    23  1.345151  0.9375  0.8500 -75.963847
    24  1.545585  1.1125  0.9375 -75.962611
    25  1.777507  1.1125  0.9375 -75.962048
    26  1.696629  1.1125  1.0250 -75.961554
    27  1.414414  1.0250  0.8500 -75.960651
    28  1.777931  1.1125  1.0250 -75.960609
    29  1.608093  1.1125  1.0250 -75.959401
    30  1.472243  0.8500  0.8500 -75.959264
    31  1.413498  0.8500  0.8500 -75.959080
    32  1.851646  1.1125  1.0250 -75.956962
    33  1.454841  1.1125  0.9375 -75.956255
    34  1.348701  0.8500  0.8500 -75.955794
    35  1.265467  0.9375  0.8500 -75.954234
    36  1.512707  1.1125  1.0250 -75.953858
    37  1.331587  1.0250  0.8500 -75.952297
    38  1.638263  1.1125  0.8500 -75.951463
    39  1.565135  1.1125  0.8500 -75.951179
    40  1.278128  0.8500  0.8500 -75.948932
    41  1.704635  1.1125  0.8500 -75.948923
    42  1.765211  1.1125  1.1125 -75.947854
    43  1.485602  1.1125  0.8500 -75.947648
    44  1.703305  1.2000  0.9375 -75.946490
    45  1.672844  1.1125  1.1125 -75.946405
    46  1.850020  1.1125  1.1125 -75.946316
    47  1.783240  1.2000  0.9375 -75.945385
    48  1.616351  1.2000  0.9375 -75.944547
    49  1.768423  1.2000  1.0250 -75.942713
    50  1.926907  1.1125  1.1125 -75.942161
    51  1.573313  1.1125  1.1125 -75.941700
    52  1.855776  1.2000  0.9375 -75.941615
    53  1.676818  1.2000  1.0250 -75.941350
    54  1.852573  1.2000  1.0250 -75.941111
    55  1.400056  1.1125  0.8500 -75.940553
    56  1.522796  1.2000  0.9375 -75.939274
    57  1.202082  0.8500  0.8500 -75.938122
    58  1.928892  1.2000  1.0250 -75.936908
    59  1.578171  1.2000  1.0250 -75.936758
    60  1.640272  1.2000  0.8500 -75.932031
    61  1.715568  1.2000  0.8500 -75.931421
    62  1.558452  1.2000  0.8500 -75.929538
    63  1.835403  1.2000  1.1125 -75.929147
    64  1.739586  1.2000  1.1125 -75.928363
    65  1.783956  1.2000  0.8500 -75.928100
    66  1.923388  1.2000  1.1125 -75.927074
    67  1.636355  1.2000  1.1125 -75.924477
    68  1.470544  1.2000  0.8500 -75.923650
    69  2.003162  1.2000  1.1125 -75.922477
    70  1.904048  1.2000  1.2000 -75.910576
    71  1.804416  1.2000  1.2000 -75.910336
    72  1.995527  1.2000  1.2000 -75.908080
    73  1.697056  1.2000  1.2000 -75.907144
    74  2.078461  1.2000  1.2000 -75.903146

As expected, we obtain 75 geometry, energy pairs (interatomic distances in Angstroms, Hartrees) 
with the energies sorted in increasing order. We are now ready to do some machine learning on this 
dataset. However, we did not set any keywords related to ML so lets do that here:

.. code-block::

    input_object.set_keyword({'use_pips':'true'})
    input_object.set_keyword({'training_points':40})
    input_object.set_keyword({'sampling':'structure_based'})
    input_object.set_keyword({'hp_maxit':10})
    input_object.set_keyword({'rseed':0})

We set the use of permutation invariant polynomials (pips). We also choose 40 training points out 
of our 75 point dataset. We sample the 40 training points with the structure-based sampling algorithm, 
and train over 10 different hyperparamter configurations. For reproduciblity, we fix the random seed 
of the hyperparameter search.

We use Gaussian process regression here. We supply a dataset, and input_object for access to the various 
keywords we have set, and a ``molecule_type`` which is required for using PIPs. The ``molecule_type`` must be a 
string given in the order of most common element first, e.g. A3B2C, A4B, etc. We could alternatively supply 
our Molecule object from before by passing ``molecule=molecule_object`` instead.

.. code-block::

    gp = peslearn.ml.GaussianProcess("PES.dat", input_object, molecule_type='A2B')
    gp.optimize_model()

.. code-block::

    Using permutation invariant polynomial transformation for molecule type  A2B
    Beginning hyperparameter optimization...
    Trying 10 combinations of hyperparameters
    Training with 40 points (Full dataset contains 75 points).
    Using structure_based training set point sampling.
    Hyperparameters: 
    {'morse_transform': {'morse': False}, 'pip': {'degree_reduction': True, 'pip': True}, 'scale_X': None, 'scale_y': 'mm11'}
    Test Dataset 84.45
    Full Dataset 73.93
    Median error: 48.46
    Max 5 errors: [148.2 153.1 154.5 171.6 225.3]
    Hyperparameters: 
    {'morse_transform': {'morse': True, 'morse_alpha': 1.9000000000000001}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'std', 'scale_y': 'mm01'}
    Test Dataset 1323.87
    Full Dataset 978.36
    Median error: 606.51
    Max 5 errors: [1405.8 2066.5 3306.  3561.  3768.6]
    ...
    ...
    ...

    ###################################################
    #                                                 #
    #     Hyperparameter Optimization Complete!!!     #
    #                                                 #
    ###################################################

    Best performing hyperparameters are:
    [('morse_transform', {'morse': False}), ('pip', {'degree_reduction': False, 'pip': True}), ('scale_X', None), ('scale_y', 'mm01')]
    Fine-tuning final model architecture...
    Hyperparameters:  {'morse_transform': {'morse': False}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': None, 'scale_y': 'mm01'}
    Final model performance (cm-1):
    Test Dataset 9.38  Full Dataset 6.44     Median error: 1.09  Max 5 errors: [11.8 13.5 18.4 24.8 34.8] 

    Saving ML model data...

We have found a Gaussian process model which has a 9.38 cm-1 RMS prediction error on the test set of 
35 points, and 6.44 cm-1 RMSE for the full 75 point dataset. Very nice! The model information is saved 
in a directory called ``model1_data``, and if further models are trained (perhaps with different random 
seeds and maybe constrained hyperparameters) additional models will be saved in this same format but with 
increasing integer values, ``model2_data``, ``model3_data``, etc.

A neural network can be trained with nearly identical syntax, though one may want to specify additional keywords.

.. code-block::

    nn = peslearn.ml.NeuralNetwork("PES.dat", input_object, molecule_type='A2B')
    nn.optimize_model()

More information on this can be found in the `CLI <cli.html>`_ tutorial page.