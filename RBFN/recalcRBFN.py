from RBFN.kmeansRBFN import kmeansRBFN

import scipy.optimize as spo
import scipy as sp
import numpy as np
import sklearn.model_selection as sklcv


class RBFNwithParamTune(kmeansRBFN):
    """ radial basis function network with auto-tuning parameters"""
    def __init__(self, penaltyTerm=None):
        super().__init__(1, sigma=1)

    def fit(self,X,Y, penalty=None, numNEvals=None, kfolds=None, testSize=None):
        self.centers=super()._select_centers(X)
        super().fit(X,Y)
        if kfolds is not None:
            if numNEvals is None:
                numNEvals=np.inf
            self.optimizeAcrossPenalty(X, Y, numNEvals=numNEvals, kfolds=kfolds, testSize=testSize)
        elif numNEvals is not None or penalty is not None:
            if numNEvals is None:
                numNEvals=np.inf
            self.optimizeRBFNwithVariableLayers(self, X, Y, numEvals=numNEvals, penalty=penalty)
        else:
            self.optimizeRBFN(self, X,Y)
        # assert optimizer already set own variables

    def _select_centers(self, X):
        return self.centers

    def breakAndSetFromOptVect(toOperateOn, optVars, dataShape):
        """ extracts variabes from optVars and sets the corresponding variables in the toOperateOn RBFN object.
        returns the variables at the end too
        optVars = (N, weights, centers, sigmas). Yes, it's a very big optimization problem. N is assumed scalar (number of terms. others are all vectors of length N)
        """
        n=dataShape[0]
        nm=np.prod(dataShape)
        curBrk=0
        endBrk=n+curBrk
        alphas=optVars[curBrk:endBrk]
        curBrk=endBrk
        endBrk+=nm
        centers=np.reshape(optVars[curBrk:endBrk],dataShape)
        curBrk=endBrk
        endBrk+=n
        sigmas=optVars[curBrk:endBrk]

        toOperateOn.sigma=sigmas
        toOperateOn.centers=centers
        toOperateOn.weights=alphas
        toOperateOn.hidden_shape=len(sigmas)
        return alphas, centers, sigmas

    def optimizeRBFN(toOperateOn, X, Y,penalty=0):
        """
        optimizes weights, sigmas and centers of an RBFN network
        :param toOperateOn: RBFN object to optimize parameters of
        :param X: training set X axis 0 is elements, axis 1 is data dimensions
        :param Y: training set Y unidimensional and real
        :param penalty: L1 penalty term for having high N
        :return: sets toOperateOn to the best found values (as determined by the L1 penalized weights added to the L2 loss. returns the optimization object from sciply.optimize
        """
        n=toOperateOn.hidden_shape
        if len(X.shape)==1:
            m=1
        else:
            m=X.shape[1]
        toOperateOn.sigma=np.ones(n)
        toOperateOn.centers=super()._select_centers(X) # initial guess for cetners.
        super().fit(X,Y) # initial guess for alphas.
        def minimizationFunction(optVars):
            alphas,centers, sigmas = toOperateOn.breakAndSetFromOptVect(optVars, (n,m))

            # update=super().fit(X,Y) #unneeded, already got the weights and set in previous function
            yHat=toOperateOn.predict(X)
            return np.linalg.norm(Y-yHat)**2 + penalty*np.linalg.norm(alphas,ord=1)
        x0=np.concatenate((toOperateOn.weights, toOperateOn.centers.flatten(), toOperateOn.sigma.flatten()))
        y0=minimizationFunction(x0)

        centLBnd=-np.full(n, np.inf); centUBnd=np.full(n, np.inf)
        alphLBnd=-np.full(n*m, np.inf); alphUBnd=np.full(n*m, np.inf)
        sigLBnd=1e-8* np.ones(n); sigUBnd=1e10 * np.ones(n)
        bnds=spo.Bounds(np.concatenate((alphLBnd, centLBnd, sigLBnd)), np.concatenate((alphUBnd, centUBnd, sigUBnd)), keep_feasible=True)

        #optimize
        optRes=spo.minimize(minimizationFunction, x0, bounds=bnds, callback=None, options={'maxiter': 500, 'disp': True})
        toOperateOn.breakAndSetFromOptVect(optRes.x, (n, m))
        return optRes

    def optimizeRBFNwithVariableLayers(toOperateOn, X, Y, numEvals=np.inf, penalty=0, returnN=False):
        """
        optimizes weights, sigmas and centers of an RBFN network but also finds best value of the number of hidden layers
        :param toOperateOn: RBFN object to optimize parameters of
        :param X: training set X axis 0 is elements, axis 1 is data dimensions
        :param Y: training set Y unidimensional and real
        :param penalty: L1 penalty term for having high N
        :return: sets toOperateOn to the best found values (as determined by the L1 penalized weights added to the L2 loss. returns the optimization object from sciply.optimize
        """
        if penalty == 0:
            return toOperateOn.optimizeRBFN(X,Y,penalty=penalty) # obviously, will optimize for higher N. This is training data after all
        n=toOperateOn.hidden_shape
        if len(X.shape)==1:
            m=1
        else:
            m=X.shape[1]
        nLBnd=1; nUBnd=len(Y)
        def evalForN(thisN):
            toOperateOn.hidden_shape=thisN
            optRes=toOperateOn.optimizeRBFN(X,Y, penalty=penalty)
            return optRes

        if numEvals>=len(Y):
            nGrid=np.arange(nLBnd,nUBnd+1)
            optResArray=[evalForN(thisN) for thisN in nGrid]
        else:
            nGrid=np.linspace(nLBnd,nUBnd,numEvals, dtype=int)
            optResArray=[evalForN(thisN) for thisN in nGrid]  # TODO: something more intelligent than straight grid search. We fully expect things to usually improve with more n as the penalty term changes

        errs=list(map(lambda x: x.fun, optResArray))
        minErrIndx=errs.index(min(errs))
        bestOptRes=optResArray[minErrIndx]
        bestN=nGrid[minErrIndx]
        toOperateOn.breakAndSetFromOptVect(bestOptRes.x, (bestN,m))
        if returnN:
            return bestOptRes, bestN
        else:
            return bestOptRes

    def optimizeAcrossPenalty(toOperateOn, X,Y, numNEvals=np.inf, kfolds=5, testSize=None, returnLambda=False):
        """
        uses k-fold evaluation to try and estimate a good value for the L1 penalty term from setting variable layers
        optimizes weights, sigmas and centers of an RBFN network but also finds best value of the number of hidden layers.
        :param toOperateOn: RBFN object to optimize parameters of
        :param X: training set X axis 0 is elements, axis 1 is data dimensions
        :param Y: training set Y unidimensional and real
        :param penalty: L1 penalty term for having high N
        :return: sets toOperateOn to the best found values (as determined by the L1 penalized weights added to the L2 loss. returns the optimization object from sciply.optimize
        """
        if testSize is None:
            testSize=1/kfolds
        if len(X.shape)==1:
            m=1
        else:
            m=X.shape[1]

        kval=sklcv.ShuffleSplit(n_splits=kfolds, test_size=testSize)
        kvalList=list(kval.split(Y))
        accumFoldData=[]
        for trnIndx, testIndx in kvalList:
            if len(X.shape)>1:
                trnX=X[trnIndx,:]
                testX=X[testIndx,:]
            else:
                trnX=X[trnIndx]
                trnY=Y[testIndx]
            trnY=Y[trnIndx]
            testY=Y[testIndx]
            accumFoldData.append(((trnX,trnY), (testX, testY)))
        def minimizationFunction(lamb):
            thisError=np.zeros(kfolds)
            for i,fold in enumerate(accumFoldData):
                trnDat=fold[0]
                testDat=fold[1]
                optRes=toOperateOn.optimizeRBFNwithVariableLayers(trnDat[0], trnDat[1], numEvals=numNEvals, penalty=lamb)
                # breakAndSetFromOptVect(toOperateOn,optRes.x) # currently unnecessary, but might make an ok safeguard if things change in the future and not set to best value at end
                thisError[i]=np.linalg.norm(toOperateOn.predict(testDat[0]) - testDat[1])
            return np.mean(thisError)
        lamb0=1
        bnds=spo.Bounds(0, np.inf, True)
        lambOptRes=spo.minimize_scalar(minimizationFunction, bounds=bnds, options={'maxiter': 10, 'disp': True})
        errs=[]; resAccum=[]
        for fold in accumFoldData:
            trnDat=fold[0]
            testDat=fold[1]
            optRes, optN=toOperateOn.optimizeRBFNwithVariableLayers(trnDat[0], trnDat[1], numEvals=numNEvals, penalty=lambOptRes.x, returnN=True)
            # breakAndSetFromOptVect(toOperateOn,optRes.x) # currently unnecessary, but might make an ok safeguard if things change in the future and not set to best value at end
            errs.append(np.linalg.norm(toOperateOn.predict(testDat[0]) - testDat[1]))
            resAccum.append((optRes, optN))
        minErrIndx=errs.index(min(errs))
        bestOptRes=resAccum[minErrIndx]
        toOperateOn.breakAndSetFromOptVect(bestOptRes[0].x, (bestOptRes[1],m))
        if returnLambda:
            return bestOptRes,lambOptRes
        else:
            return bestOptRes
