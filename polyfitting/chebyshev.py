import numpy as np
from numpy import linalg as npl
from numpy.polynomial import chebyshev as cheb, polynomial as poly

from polyfitting.polycommon import PolynomialAnalyzer


def genericChebVander(locations, deg):
    # stolen straight from legvander3d and modified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[cheb.chebvander(locations[:,i], deg[i]) for i in range(n)]
    indexingTuples=[]
    raise NotImplementedError
    # v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
    return v.reshape(v.shape[:-n] + (-1,))


def genericChebVal(locations, coeffs):
    if len(locations.shape)==1:
        return cheb.chebval(locations,coeffs)
    else: # assume dim 0 is points, dim 1 is dimensions
        c=cheb.chebval(locations[:,0], coeffs)
        for i in range(1, locations.shape[1]):
            c=cheb.chebval(locations[:,i],c, tensor=False)
        return c


def chebVander4d(locations, deg):
    # stolen straight from legvander3d and mondified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[cheb.chebvander(locations[:,i], deg[i]) for i in range(n)]
    v = vi[0][..., None, None, None]*vi[1][..., None,:,None, None]*vi[2][..., None, None,:,None]*vi[3][...,None,None,None,:]
    return v.reshape(v.shape[:-n] + (-1,))


def chebVander5d(locations, deg):
    # stolen straight from legvander3d and modified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[cheb.chebvander(locations[:,i], deg[i]) for i in range(n)]
    v = vi[0][..., None, None, None, None]*vi[1][..., None,:,None, None, None]*vi[2][..., None, None,:,None,None]*vi[3][...,None,None,None,:,None]*vi[4][...,None,None,None,None,:]
    return v.reshape(v.shape[:-n] + (-1,))


def chebForwardTransform(orders, locations, functionVals):
    if len(locations.shape)==1:
        return np.array(cheb.chebfit(locations, functionVals, orders[0]))
    else:
        if locations.shape[1]==2:
            V=cheb.chebvander2d(locations[:,0], locations[:,1], orders)
        elif locations.shape[1]==3:
            V=cheb.chebvander3d(locations[:,0],locations[:,1],locations[:,2], orders)
        elif locations.shape[1]==4:
            V=chebVander4d(locations,orders)
        elif locations.shape[1]==5:
            V=chebVander5d(locations,orders)
        else:
            raise NotImplementedError # there's a bad startup joke about this being good enough for the paper.
        ret, _, _, _=npl.lstsq(V, functionVals, rcond=None)
        return np.reshape(ret, (np.array(orders)+1).flatten())


def chebReconstruct(orders, locations, coeffs,unusedNumPts):
    if len(locations.shape)==1:
        return np.array(cheb.chebval(locations, coeffs))
    else:
        if locations.shape[1] == 2:
            return cheb.chebval2d(locations[:,0], locations[:,1], coeffs)
        elif locations.shape[1] == 3:
            return cheb.chebval3d(locations[:,0],locations[:,1],locations[:,2], coeffs)
        else:
            return genericChebVal(locations, coeffs)


def chebReconstructDerivative(freqs, locations, spectrum, numPts):
    """

    :param freqs:
    :param locations:
    :param spectrum:
    :param numPts:
    :return: derivative of iFFT of the spectrum as an array with shape (Npts, Ncomponent) or (Ncomponent,) if 1-d
    """
    deriv=poly.polyder(spectrum, axis=0)
    return poly.polyval(locations, deriv)


class chebDirectAnalyzer(PolynomialAnalyzer):
    def __init__(self,pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None):
        super().__init__(chebForwardTransform, chebReconstruct, chebReconstructDerivative, pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None)