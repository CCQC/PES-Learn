import numpy as np

def cart_dist_B_2(X, do_hess=False):
    ndat, nvar = X.shape
    natom = round(nvar/3)
    X = X.reshape((ndat,natom,3))
    nr = round((natom**2 - natom) / 2)
    R = np.zeros((ndat, nr))
    def cart_dist_inner1(R):
        ctr = 0
        for atomi in range(natom):
            for atomj in range(atomi):
                R[:,ctr] = np.linalg.norm(X[:,atomi,:] - X[:,atomj,:], axis=1)
                ctr += 1
    cart_dist_inner1(R=R)
    B1 = np.zeros((ndat, nr, nvar))
    B2 = np.zeros((ndat, nr, nvar, nvar))
    def cart_dist_inner2(B1):
        for atomi in range(natom):
            for atomj in range(atomi):
                r = round((atomi*(atomi-1))/2) + atomj
                B1[:, r, 3*atomi:3*(atomi+1)] = np.divide((X[:,atomi,:] - X[:,atomj,:]), R[:,r][:,None])
                B1[:, r, 3*atomj:3*(atomj+1)] = -B1[:, r, 3*atomi:3*(atomi+1)]
    cart_dist_inner2(B1=B1)
    if do_hess:
        for atomi in range(natom):
            for atomj in range(atomi):
                r = round((atomi*(atomi-1))/2) + atomj
                v = X[:, atomi, :] - X[:, atomj, :]
                matt = np.einsum("ni, nj -> nij", v, v, optimize="optimal")
                matt *= (R[:,r][:,None,None]**-3.0)
                matt -= (np.identity(3) * (R[:,r][:,None,None]**-1.0))
                B2[:, r, 3*atomi:3*(atomi+1), 3*atomi:3*(atomi+1)] = -1.0 * matt
                B2[:, r, 3*atomj:3*(atomj+1), 3*atomj:3*(atomj+1)] = -1.0 * matt
                B2[:, r, 3*atomi:3*(atomi+1), 3*atomj:3*(atomj+1)] = matt
                B2[:, r, 3*atomj:3*(atomj+1), 3*atomi:3*(atomi+1)] = matt
    return R, B1, B2
