#######################
Kernel Ridge Regression
#######################

Kernel Ridge Regression in PES-Learn is done via interface to `scikit-learn <https://scikit-learn.org/stable/>`_ . 
At the time of writing this, scikit-learn has six options for kernel functions to use with kernel ridge regresstion (KRR). 
PES-Learn implements five of these options, polynomial, RBF, Laplacian, Sigmoid, and cosine. When chosing a verbose 
kernel with the keyword ``kernel = verbose``, PES-Learn will search the hyperparameter space of all five of these kernels, some of which 
have additional options (such as degree of polynomial) which makes the hyperparameter space very large. Because of this it is recommended
to do an initial search with the verbose space and then narrow down the search with a ``precomputed`` kernel. The following
example covers just this. If you would like to work along with this example, the ``PES.dat`` file is available `here <pes.html>`_ to copy.

.. note::

    PES-Learn does not support the sixth kernel available with scikit-learn, chi^2, with the ``kernel`` keyword set to ``verbose``,
    but if the user so chooses the option is still available with a ``precomputed kernel``. This kernel is not in the ``verbose`` set
    because it typically is a poor description of potential energy surfaces, and was left out to reduce the hyperparameter space.

*********************
Verbose Kernel Search 
*********************

Let us assume that we have already generated data and parsed it to a file ``PES.dat`` (see `CLI <cli.hmtl>`_ for tips on doing this).
In this example we are examining the PES of the water dimer at MP2/6-31+G**, our single point energies were run with Psi4, and our ``PES.dat``
looks like this:

.. code-block::

    r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,E
    0.107393696300,1.511209005600,0.958000000000,0.960000000000,0.960000000000,0.965000000000,4.000000000000,4.358877312500,4.421347611400,4.963684516800,5.230357990000,5.312408594200,5.375381443600,5.694425695800,5.744200549400,-148.80832244761734
    0.107393696300,1.511209005600,0.958000000000,0.960000000000,0.960000000000,0.965000000000,4.000000000000,4.358877312500,4.421347611400,4.957842687500,5.168189668400,5.317748361000,5.379703357400,5.650727044700,5.699458003200,-148.80832802926432
    0.107393696300,1.511209005600,0.958000000000,0.960000000000,0.960000000000,0.965000000000,4.000000000000,4.358877312500,4.421347611400,4.964853815800,5.259243598200,5.308560352700,5.372087089700,5.726738564900,5.778198657600,-148.80833921708614
    0.107393696300,1.511209005600,0.958000000000,0.960000000000,0.960000000000,0.965000000000,3.892857142900,4.254231044800,4.316513987700,4.856548656300,5.125227560200,5.206358341000,5.270299165700,5.580693108000,5.648377814800,-148.80842398247543
    0.107393696300,1.511209005600,0.958000000000,0.960000000000,0.960000000000,0.965000000000,3.892857142900,4.254231044800,4.316513987700,4.850737828400,5.063478368600,5.210650713200,5.274891032200,5.534120397300,5.601491987300,-148.80842822074243
    0.107393696300,1.511209005600,0.958000000000,0.960000000000,0.960000000000,0.965000000000,3.892857142900,4.254231044800,4.316513987700,4.846094655800,5.030456998800,5.211506356100,5.272612122600,5.510508312600,5.557029710400,-148.80842874416982
    ...

In the generation of this data, the number of data points was reduced to 1500. Let's examine this dataset with KRR using a ``verbose`` 
hyperparameter space since we don't know much about the PES initially. We add the following keywords to our ``input.dat`` file to generate a 
KRR surface:

.. code-block::

    # Machine learning keywords
    ml_model = krr
    hp_maxit = 500
    kernel = verbose
    sampling = structure_based
    use_pips = true

Here we tell PES-Learn that we want to create a machine learning model with KRR, allow it to run over 500 hyperparameter optimizations, do a verbose
kernel hyperparameter search, use structure based sampling to split training and test sets, and use permutationally invariant polynomials (PIPs). 

We run this with PES-Learn and it will print the hyperparameter optimizations and run over the iterations given in ``hp_maxit``. Of the iterations 
it will find the one with the lowest dataset error and return that collection of hyperparameters at the end:

.. code-block::

    Best performing hyperparameters are:
    [('alpha', 1e-06), ('kernel', {'degree': None, 'gamma': None, 'ktype': 'laplacian'}), ('morse_transform', {'morse': True, 'morse_alpha': 1.0}), ('pip', {'degree_reduction': False, 'pip': True}), ('scale_X', None), ('scale_y', 'std')]
    Fine-tuning final model...
    Hyperparameters:  {'alpha': 1e-06, 'kernel': {'degree': None, 'gamma': None, 'ktype': 'laplacian'}, 'morse_transform': {'morse': True, 'morse_alpha': 1.0}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': None, 'scale_y': 'std'}
    Final model performance (cm-1):
    R^2 0.9999999878503739
    Test Dataset 22.87  Full Dataset 10.23     Median error: 0.25  Max 5 errors: [ 80.1 139.7 149.3 150.1 155.9] 

    Model optimization complete. Saving final model...
    Saving ML model data...
    Total run time: 495.87 seconds

The errors are printed in wavenumbers (cm^-1) and we see that the best performing hyperparameters tested give an average error of 10.23 cm^-1 for the full dataset.

PES-Learn generated a model that uses these hyperparameters that can be used to make predictions about the given PES. Before we get to that, however, lets see if we can 
generate a better ML model with a bit of fine-tuning.

Let's now build a ``precomputed`` kernel with some of the best performing hyperparameters to narrow our hyperparameter space and hopefully build a 
better model. From the hyperparameter optimizations, (not shown because of length) it appears the polynomial and Laplacian kernels performed well 
with respect to errors, so lets examine them separately. It is important to note that when using a precomputed kernel that includes a polynomial type 
kernel it is recommended to optimize the hyperparameters for that kernel separately. The polynomial kernel takes another hyperparameter, the degree 
of the polynomial. If you try to optimize multiple kernel functions at once with the degree hyperparameter, the optimization scheme will try and 
find trends between degree and performance with other kernels being used. This will a.) build a larger hyperparameter space and b.) could skew 
results with trends that don't exist. This is explicitly accounted for with the ``verbose`` kernel option, but not with a ``precomputed`` kernel.
As such we will try to build two separate models, one with a polynomial kernel and one with a Laplacian. Let us first do this for a polynomial kernel, 
we change our input to the following for a precomputed kernel:

.. code-block::

    # Machine learning keywords
    ml_model = krr
    hp_maxit = 500
    kernel = precomputed
    precomputed_kernel = {'kernel': ['polynomial'], 'degree': ['uniform', 1, 6, 1]}
    sampling = structure_based
    use_pips = true

We have changed ``kernel`` to ``precomputed`` and set the ``precomputed_kernel`` option to a dicitonary of options for our kernel. By setting the 
first option in degree to 'uniform' that tells PES-Learn (and by extension HyperOpt) to use degrees from 1 to 6, stepping by 1 each time. This 
means that hyperparameter optimizations will examine polynomials of degree 1, 2, 3, 4, 5, and 6. Equivalently, we could leave out the 'uniform' option 
and set ``precomputed_kernel = {'kernel': ['polynomial'], 'degree': [1, 2, 3, 4, 5, 6]}``. Leaving out 'uniform' allows specifications for exactly 
which degree(s) to examine. 

Let's now run this with the precomputed polynomial kernel and see what that results us.

.. code-block::

    Best performing hyperparameters are:
    [('alpha', 1e-06), ('degree', 6.0), ('gamma', None), ('kernel', 'polynomial'), ('morse_transform', {'morse': True, 'morse_alpha': 1.2000000000000002}), ('pip', {'degree_reduction': False, 'pip': True}), ('scale_X', 'std'), ('scale_y', 'mm01')]
    Fine-tuning final model...
    Hyperparameters:  {'alpha': 1e-06, 'degree': 6.0, 'gamma': None, 'kernel': 'polynomial', 'morse_transform': {'morse': True, 'morse_alpha': 1.2000000000000002}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': 'std', 'scale_y': 'mm01'}
    Final model performance (cm-1):
    R^2 0.999999887725274
    Test Dataset 69.53  Full Dataset 45.16     Median error: 21.14  Max 5 errors: [162.8 183.3 205.6 293.8 740.3] 

    Model optimization complete. Saving final model...
    Saving ML model data...
    Total run time: 494.81 seconds

It looks like this didn't do quite as well as we had hoped, so let's try the Laplacian kernel now.  Let's change the keywords in our input.dat again:

.. code-block::

    ...
    precomputed_kernel = {'kernel': ['laplacian']}
    ...

Let's run it and see what it gets us:

.. code-block::

    Best performing hyperparameters are:
    [('alpha', 1e-06), ('degree', 1), ('gamma', None), ('kernel', 'laplacian'), ('morse_transform', {'morse': True, 'morse_alpha': 1.0}), ('pip', {'degree_reduction': False, 'pip': True}), ('scale_X', None), ('scale_y', 'std')]
    Fine-tuning final model...
    Hyperparameters:  {'alpha': 1e-06, 'degree': 1, 'gamma': None, 'kernel': 'laplacian', 'morse_transform': {'morse': True, 'morse_alpha': 1.0}, 'pip': {'degree_reduction': False, 'pip': True}, 'scale_X': None, 'scale_y': 'std'}
    Final model performance (cm-1):
    R^2 0.9999999878503739
    Test Dataset 22.87  Full Dataset 10.23     Median error: 0.25  Max 5 errors: [ 80.1 139.7 149.3 150.1 155.9] 

    Model optimization complete. Saving final model...
    Saving ML model data...
    Total run time: 515.5 seconds

We get the same answer as we initially did. This is a good indication that this may be the best model KRR can make, unless we drastically expand the hyperparameter space.
You may notice some of the other hyperparameters, like gamma and alpha which can also be set with a precomputed kernel. show how to do this along with other hps bellow ian




KRR example with precomputed_kernel (then link from cli (and maybe api))

{'kernel': ['rbf','polynomial']



