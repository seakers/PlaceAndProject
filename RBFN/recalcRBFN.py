from RBFN.kmeansRBFN import kmeansRBFN

import scipy.optimize as spo
import scipy as sp
import numpy as np


class RBFNwithParamTune(kmeansRBFN):
    """ radial basis function network with auto-tuning parameters"""
    def __init__(self, penaltyTerm=1):
        super().__init__(1, sigma=1)
        self.changeSemaphore=False
        self.penalty=penaltyTerm

    def breakAndSetFromOptVect(self, optVars):
        # optVars = (N, alphas, centers, sigmas). Yes, it's a very big optimization problem
        N=optVars[0]
        curBrk=1
        brk=(len(optVars)-1)/3 # assert: must be an integer
        alphas=optVars[curBrk:brk+curBrk]
        curBrk+=brk
        centers=optVars[curBrk:brk+curBrk]
        curBrk+=brk
        sigmas=optVars[curBrk:brk+curBrk]

        self.sigma=sigmas
        self.centers=centers
        self.weights=alphas

    def fit(self, X, Y):
        def minimizationFunction(optVars):
            self.breakAndSetFromOptVect(optVars)

            # update=super().fit(X,Y) #unneeded, already got the weights and stuff
            yHat=self.predict(X)
            return np.linalg.norm(Y-yHat)**2 + self.penalty*np.linalg.norm(alphas,ord=1)

        n=len(Y)
        self.centers=super()._select_centers(X)
        self.sigma=np.ones(n)

        N0=1
        alph0=super().fit(X,Y)
        x0=(N0,alph0, self.centers, self.sigma)
        y0=minimizationFunction(x0)

        NlBnd=1; NuBnd=n
        centLBnd=-np.full(n, np.inf); centUBnd=np.full(n, np.inf)
        alphLBnd=-np.full(n, np.inf); alphUBnd=np.full(n, np.inf)
        sigLBnd=1e-8* np.ones(n); sigUBnd=1e10 * np.ones(n)
        bnds=spo.Bounds((NlBnd, alphLBnd, centLBnd, sigLBnd), (NuBnd, alphUBnd, centUBnd, sigUBnd), keep_feasible=True)

        #optimize
        optRes=spo.minimize(minimizationFunction, x0, bounds=bnds, callback=None, options={'maxiter': 500, 'disp': True})
        self.breakAndSetFromOptVect(optRes.x)
        return optRes.func-y0 # for fun.

    def _select_centers(self, X):
        return self.centers
