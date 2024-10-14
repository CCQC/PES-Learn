import numpy as np
import sympy
import re

class Pip_B():
    def __init__(self, fn, nbonds):
        # Returns PIPs from X (n interatomic distances), degrees of PIPs,
        # Grad. of PIP wrt dist., and Hess. of PIP wrt dist. x dist.

        # Read and prep PIP strings
        with open(fn, 'r') as f:
            data = f.read()
        data = re.sub('\^', '**', data)

        # Define variables for Sympy
        symbols = ""
        for i in range(nbonds):
            symbols += f"x{i} "

        variables = sympy.symbols(symbols)
        for i in range(1, nbonds+1):
            data = re.sub('x{}(\D)'.format(str(i)), 'x{}\\1'.format(i-1), data)

        # Define PIP equations for Sympy
        self.polys = re.findall("\]=(.+)",data)

        # Calculate PIP first and second derivatives and associated "lambdified" functions
        self.grad = []
        self.grad_lambda = []
        self.hess = []
        self.hess_lambda = []
        for p in self.polys:
            lil_grad = []
            lil_grad_lambda = []
            lil_hess = []
            lil_hess_lambda = []
            for x1 in variables:
                d1 = sympy.diff(p, x1)
                lil_grad.append(d1)
                lil_grad_lambda.append(re.sub(r"x(\d+)", r"X[:,\1]", str(d1)))
                liller_hess = []
                liller_hess_lambda = []
                for x2 in variables:
                    d1d2 = sympy.diff(d1, x2)
                    liller_hess.append(d1d2)
                    liller_hess_lambda.append(re.sub(r"x(\d+)", r"X[:,\1]", str(d1d2)))
                lil_hess.append(liller_hess)
                lil_hess_lambda.append(liller_hess_lambda)
            self.grad.append(lil_grad)
            self.grad_lambda.append(lil_grad_lambda)
            self.hess.append(lil_hess)
            self.hess_lambda.append(lil_hess_lambda)
 
        # Determine nonzero second derivatives w.r.t. polynomial and the Hessian row (xi)
        self.grad_not_zero = []
        self.grad_not_const = []
        self.grad_const = []
        self.hess_not_const = []
        self.hess_const = []
        self.hess_not_zero = []
        
        for pi in range(len(self.polys)):
            for xi in range(nbonds):
                if self.grad[pi][xi] != 0:
                    if "x" in str(self.grad[pi][xi]):
                        self.grad_not_const.append((pi, xi))
                    else:
                        self.grad_const.append((pi, xi))
                    self.grad_not_zero.append((pi, xi))
                for xj in range(nbonds):
                    if self.hess[pi][xi][xj] != 0:
                        if "x" in str(self.hess[pi][xi][xj]):
                            self.hess_not_const.append((pi, xi, xj))
                        else:
                            self.hess_const.append((pi, xi, xj))
                        self.hess_not_zero.append((pi, xi, xj))


    def transform(self, X, do_hess=False):
        ndat, nbonds = X.shape
        new_X = np.zeros((ndat, len(self.polys)))
        # Evaluate polynomials
        for i, p in enumerate(self.polys):    # evaluate each FI 
            # convert the FI to a python expression of raw_X, e.g. x1 + x2 becomes raw_X[:,1] + raw_X[:,2]
            eval_string = re.sub(r"(x)(\d+)", r"X[:,\2]", p)
            # evaluate that column's FI from columns of raw_X
            new_X[:,i] = eval(eval_string)

        # Evaluate polynomial derivatives
        egrad = np.zeros((ndat, len(self.polys), nbonds))
        ehess = np.zeros((ndat, len(self.polys), nbonds, nbonds))
        
        for pi, xi in self.grad_not_const:
            egrad[:,pi,xi] = eval(self.grad_lambda[pi][xi])
        for pi, xi in self.grad_const:
            egrad[:,pi,xi] = float(self.grad[pi][xi])

        if do_hess:
            for pi, xi, xj in self.hess_not_const:
                ehess[:,pi,xi,xj] = eval(self.hess_lambda[pi][xi][xj])
            for pi, xi, xj in self.hess_const:
                ehess[:,pi,xi,xj] = float(self.hess[pi][xi][xj])

        degrees = []
        for p in self.polys:
            # just checking first, assumes every term in each FI polynomial has the same degree (seems to always be true)
            tmp = p.split('+')[0]
            # count number of exponents and number of occurances of character 'x'
            exps = [int(i) - 1 for i in re.findall("\*\*(\d+)", tmp)]
            ndegrees = len(re.findall("x", tmp)) + sum(exps)
            degrees.append(ndegrees)

        return new_X, degrees, egrad, ehess # PIP values, degrees, B1, and B2

