import numpy as np

# Scaling functions (flattened to vectors): begin
def morse(x, alpha):
    # THIS IS BIG BOOTY WRONG!!! NEEDS MINUS SIGN IN EXPONENT!!!!!!!!! TODO
    return np.exp(-x/alpha)

def morse_B1(x, alpha=1.0):
    # dm/dr
    return -1.0 * (alpha**-1.0) * morse(x, alpha) #TODO

def morse_B2(x, alpha=1.0):
    return (alpha**-2.0) * morse(x, alpha)

def scale_mean_B1(x, stds):
    return np.array([[stds[i] for i in range(len(stds))] for n in range(len(stds))])
    #return x.std(axis=0)**-1

def scale_mm_B1(x, bmin, bmax):
    xmin = x.min(axis=0)
    xmax = x.max(axis=0)
    return (bmax-bmin)/(xmax-xmin)

def degree_B1(x, degrees):
    pwr = np.power(degrees, -1.0) - 1
    return np.divide(np.power(x, pwr), degrees)

def degree_B2(x, degrees):
    pwr = np.power(degrees, -1.0) - 2
    factor = np.power(degrees,-2.0) - np.power(degrees,-1.0)
    return np.multiply(np.power(x, pwr), factor)

def dist(x):
    pass

def ddist(x):
    pass
