import numpy as np
from numpy import linalg as npl
from numpy.polynomial import legendre as leg, polynomial as poly

from polyfitting.polycommon import PolynomialAnalyzer


def legendreToNewton(legCoeffs):
    return leg.Legendre(legCoeffs).convert(kind=np.polynomial.polynomial.Polynomial)


def newtonToLegendre(newtonCoeffs):
    return np.polynomial.polynomial.Polynomial(newtonCoeffs).convert(kind=leg.Legendre)


def genericNewtVander(locations, deg):
    # stolen straight from legvander3d and modified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[poly.polyvander(locations[:,i], deg[i]) for i in range(n)]
    indexingTuples=[]
    raise NotImplementedError
    # v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
    return v.reshape(v.shape[:-n] + (-1,))


def genericNewtVal(locations, coeffs):
    if len(locations.shape)==1:
        return poly.polyval(locations,coeffs)
    else: # assume dim 0 is points, dim 1 is dimensions
        c=poly.polyval(locations[:,0], coeffs)
        for i in range(1, locations.shape[1]):
            c=poly.polyval(locations[:,i],c, tensor=False)
        return c


def newtVander4d(locations, deg):
    # stolen straight from legvander3d and mondified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[poly.polyvander(locations[:,i], deg[i]) for i in range(n)]
    v = vi[0][..., None, None, None]*vi[1][..., None,:,None, None]*vi[2][..., None, None,:,None]*vi[3][...,None,None,None,:]
    return v.reshape(v.shape[:-n] + (-1,))


def newtVander5d(locations, deg):
    # stolen straight from legvander3d and modified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[poly.polyvander(locations[:,i], deg[i]) for i in range(n)]
    v = vi[0][..., None, None, None, None]*vi[1][..., None,:,None, None, None]*vi[2][..., None, None,:,None,None]*vi[3][...,None,None,None,:,None]*vi[4][...,None,None,None,None,:]
    return v.reshape(v.shape[:-n] + (-1,))


def newtForwardTransform(orders, locations, functionVals):
    if len(locations.shape)==1:
        return np.array(poly.polyfit(locations, functionVals, orders[0]))
    else:
        if locations.shape[1]==2:
            V=poly.polyvander2d(locations[:,0], locations[:,1], orders)
        elif locations.shape[1]==3:
            V=poly.polyvander3d(locations[:,0],locations[:,1],locations[:,2], orders)
        elif locations.shape[1]==4:
            V=newtVander4d(locations,orders)
        elif locations.shape[1]==5:
            V=newtVander5d(locations,orders)
        else:
            raise NotImplementedError # there's a bad startup joke about this being good enough for the paper.
        ret, _, _, _=npl.lstsq(V, functionVals, rcond=None)
        return np.reshape(ret, (np.array(orders)+1).flatten())


def newtReconstruct(orders, locations, coeffs,unusedNumPts):
    if len(locations.shape)==1:
        return np.array(poly.polyval(locations, coeffs))
    else:
        if locations.shape[1] == 2:
            return poly.polyval2d(locations[:,0], locations[:,1], coeffs)
        elif locations.shape[1] == 3:
            return poly.polyval3d(locations[:,0],locations[:,1],locations[:,2], coeffs)
        else:
            return genericNewtVal(locations, coeffs)


def newtReconstructDerivative(freqs, locations, spectrum, numPts):
    """

    :param freqs:
    :param locations:
    :param spectrum:
    :param numPts:
    :return: derivative of iFFT of the spectrum as an array with shape (Npts, Ncomponent) or (Ncomponent,) if 1-d
    """
    deriv=poly.polyder(spectrum, axis=0)
    return poly.polyval(locations, deriv)


class newtonDirectAnalyzer(PolynomialAnalyzer):
    def __init__(self,pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None):
        super().__init__(newtForwardTransform, newtReconstruct, newtReconstructDerivative, pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None)