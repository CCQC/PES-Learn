# PES-Learn usage FAQ

## 1. How to install PES-Learn...
  * from source?
    *  In a command line environment on Linux, OSX, or [Windows 10 Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10): 
      * `git clone https://github.com/CCQC/PES-Learn.git`
      * `cd PES-Learn`
      * `python setup.py install`
      * `pip install -e .`
  * with pip?
      * _coming soon_
  * with conda?
      * _coming soon_

## 2. How do I use PES-Learn?
  * The code can be used in two formats, either with an input file `input.dat` or with the Python API. See Tutorials section for examples. If an input file is created, one just needs to run `python path/to/PES-Learn/peslearn/driver.py` while in the directory containing the input file. To use the Python API, create a python file which imports peslearn `import peslearn`. This requires the package to be in your Python path: `export PYTHONPATH="absolute/path/to/directory/containing/peslearn"`. This can be executed on the command line or added to your shell intializer (e.g. `.bashrc`) for more permanent access. 
    
## 3. Why is data generation so slow?
  * This is probably because you're generating a lot of points.  Also, that part of the code isn't very efficient. My bad. Multiplying the grid increments together gives the total number of points, so if there are 6 geometry parameters with 10 increments each thats 10^6 internal coordinate configurations.  If you are not removing redundancies (`remove_redundancy=false`) or reducing the grid size to some value (e.g. `grid_reduction=1000`)  it is recommended to only generate tens of thousands of points at a time. If you are removing redundancies and/or filtering geometries, it is not recommended to generate more than a few million internal coordinate configurations. Finally, the algorithm behind `remember_redundancies=true` is very slow for large datasets with a lot of permutational symmetry, as like-geometries have to be paired together and stored.
    
## 4. Why is my machine learning model so bad?
  * 95% of the time it means your dataset sucks. Open the dataset and look at the energies. If it is a PES-Learn-generated dataset, the energies are in increasing order by default (can be disabled with `sort_pes=false`. Scrolling through the dataset, the energies should be smoothly increasing. If there are large jumps in the energy values (typcially towards the end of the file) these points are probably best deleted. If the dataset looks good, the ML algorithm probably just needs more training points in order to model the dimensionality and features of the surface. Either that, or PES-Learn's automated ML model optimization routines are just not working for your use case.
    
## 5. Why is training ML models so slow?
  * A few things you can do:
    * Train over less hyperparameter optimization iterations
    * Ensure multiple cores/threads are being used by your CPU. This can be done by checking which BLAS library NumPy is using:
    * open an interactive python session with `python` and then `import numpy as np` followed by `np.__show_config()`. If this displays a bunch of references to `mkl`, then NumPy is using Intel MKL. If this displays a bunch of references to 'openblas' then Numpy is using OpenBLAS. If Numpy is using MKL, you can control CPU usage with the environment variable MKL_NUM_THREADS=4 or however many physical cores your CPU has (this is recommended by Intel; do not use hyperthreading).   If Numpy is using OpenBLAS, you can control CPU usage with OMP_NUM_THREADS=8 or however many threads are available. In bash, environment variables can be set by typing `export OMP_NUM_THREADS=8` into the terminal. Note that instabilities such as memory leaks due to thread-overallocation can occur if _both_ of these environment variables are set depending on your configuration (i.e., if one is set to 4 or 8 or whatever, make sure to set the other to =1).
  * You also may be using a lot of training points
    * Gaussian processes scale poorly with the number of training points. Any more than 1000-2000 is unreasonable on a personal computer. If submitting to some external computing resource, anything less than 5000 or so is reasonable. Use neural networks for large training sets. If it is still way too slow, you can try to constrain the neural networks to use the Adam optimizer instead of the BFGS optimizer.  
      
      
## 6. How do I use this machine learning model?
  * When a model is finished training PES-Learn exports a folder `model1_data` which contains a bunch of stuff including a Python code `compute_energy.py` with convenience function `pes()` for evaluating energies with the ML model. Directions for use are written directly into the `compute_energy.py` file. The convenience function can be imported into other Python codes that are in the same directory with `from compute_energy import pes`.  This is also in principle accessible from codes written in other programming languages such as C, C++ through their respective Python APIs, though these can be tricky to use.
    
## 7. What are all these Hyperparameters?
  * `scale_X` is how each individual input (geometry parameter) is scaled. `scale_y` is how the energies (outputs) are scaled. 
    * `std` is standard scaling, each column of data is scaled to a mean of 0 and variance of 1. 
    * `mm01` is minmax scaling, each column of data is scaled such that it runs from 0 to 1
    * `mm11` is minmax scaling with a range -1 to 1
  * `morse` is whether interatomic distances are transformed into morse variables $r_1 \rightarrow e^{r_1/\alpha}$
  * `pip` stands for permutation invariant polynomials; i.e. the geometries are being transformed into a permutation invariant representation using the fundamental invariants library. 
  * `degree_reduce` is when each fundamental invariant polynomial result is taken to the $1/n$ power where $n$ is the degree of the polynomial
  * `layers` is a list of the number of nodes in each hidden layer of the neural network

## 8. How many points do I need to generate?

 *  It's very hard to say what size of training set is required for a given target accuracy; it depends on a lot of things. First, the application: if you are doing some variational computation of the vibrational energy levels and only want the fundamentals, you might be able to get away with less points because you really just need a good description of the surface around the minimum. If one wants high-lying vibrational states with VCI, the surface needs a lot more coverage, and therefore more points. If the application involves a reactive potential energy surface across several stationary points, even more points are needed. The structure of the surface itself can also influence the number of points needed.
**You don't know until you try.** For a given system, one should try out a few internal coordinate grids, reduce them to some size with `grid_reduction`, compute the points at a low level of theory, and see how well the models perform.  

## 9. How big can the molecular system be? 

No more than 5-6 atoms for 'full' PESs. Any larger than that, and generating data by displacing in internal coordinates is impractical (if you have 6 atoms and want 5 increments along each internal coordinate, that's already ~240 million points). This is just an unfortunate reality of high-dimensional spaces: ample coverage over each coordinate and all possible coordinate displacement couplings requires an impossibly large grid of points for meaningful results. **One can still do large systems if they only scan over some of the coordinates.** For example, you can do relaxed scans across the surface, fixing just a few internal coordinates and relaxing all others through geometry optimization at each point, and creating a model of this 'sub-manifold' of the surface is no problem (i.e., train on the fixed coordinate parameters and 'learn' the relaxed energies). This is useful for inspecting reaction coordinates/reaction entrance channels, for example. 
Future releases will support including gradient information in training the model, and this may allow for slightly larger systems and smaller dataset sizes. In theory, gradients can give the models more indication of the curvature of the surface with less points.


