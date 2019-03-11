import numpy as np
import pandas as pd
import numpy.linalg as npl
import scipy as sp
import functools as ft
import matplotlib.pyplot as plt
import scipy.ndimage.filters as spndf
import numpy.polynomial.legendre as leg
import numpy.polynomial.polynomial as poly
import numpy.polynomial.chebyshev as cheb
import warnings

import common as cmn
import meanPlane as mP
import tradeoffMatrixImage as tMI
import analyticsCommon as aC

from numpy.polynomial import legendre as pl

#nice reference: http://mathfaculty.fullerton.edu/mathews/n2003/LegendrePolyMod.html


class PolynomialAnalyzer():
    """
    """
    def __init__(self,forwardTransform, reconstruct, reconstructDerivative, pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None):
        self.pointHeight=pointHeight
        self.forwardTransform=forwardTransform
        self.pointLocation=pointLocation
        if normalizeRange is None:
            self.normalizeRange=np.ptp(pointLocation, keepdims=True)
        else:
            self.normalizeRange=normalizeRange
        if normalizeMin is None:
            self.normalizeMin=np.min(pointLocation, keepdims=True)
        else:
            self.normalizeMin=normalizeMin
        self.normalizedPointLocation=self.normalize(pointLocation)
        self.reconCall=reconstruct
        self.reconDcall=reconstructDerivative
        if ordersToEval is None:
            if len(pointLocation.shape)==1:
                self.ordersToEval=cmn.numpyze(self.pointHeight.size-1)
                self.ordersToEval=2*np.sqrt(self.ordersToEval) #for stability see: https://en.wikipedia.org/wiki/Runge%27s_phenomenon#Mitigations_to_the_problem
            else:
                totalCapacity=2*np.sqrt(cmn.numpyze(self.pointHeight.size-1))
                nthRoot=np.floor(totalCapacity**(1/(2*pointLocation.shape[1])))
                self.ordersToEval=np.array([nthRoot,]*pointLocation.shape[1])
        else:
            self.ordersToEval=cmn.numpyze(ordersToEval)
        self.ordersToEval=self.ordersToEval.astype(int)
        if len(self.ordersToEval) ==1:
            self.fullOrders=np.arange(self.ordersToEval+1)
        else:
            self.fullOrders=np.array(list((np.arange(d+1) for d in self.ordersToEval)))
        self.spectralFilters=[]
        self.inputFilters=[]

    def addSpectralFilter(self,filter):
        self.spectralFilters.append(filter)
    def removeSpectralFilter(self, filter):
        self.spectralFilters.remove(filter)
    def addInputFilter(self,filter):
        self.inputFilters.append(filter)
    def removeInputFilter(self,filter):
        self.inputFilters.remove(filter)

    def normalize(self,x):
        if np.any(np.logical_or(x<self.normalizeMin, x>(self.normalizeMin+self.normalizeRange))):
            warnings.warn('outside bounds when normalizing points. input to normalize exceeds bounds of original points to interpolate or manually input range and bounds')
        return 2*((x-self.normalizeMin)/self.normalizeRange) - 1

    @property
    def spectrum(self):
        """
        :return: the one-sided spectrum of the pointHeights and locations input when creating the analyzer
        """
        # instead product out the list of frequencies and then calculate
        return self.filteredSpectrum()

    def filteredSpectrum(self):
        def decForward(orders, locations, functionVals):
            return self.forwardTransform(self.ordersToEval, locations, functionVals) # ugly hack to convert full orders to to eval orders at last second
        return aC.filteredSpectrum(self.inputFilters,self.spectralFilters,self.fullOrders, self.normalizedPointLocation, self.pointHeight, decForward)

    def trueSpectrum(self):
        return self.forwardTransform(self.ordersToEval, self.normalizedPointLocation, self.pointHeight)

    def reconstruction(self, locations=None):
        """
        runs fourier series defined by this analysis.
        :param locations: locations to evaluate at. if None (default) evaluates on the input locations
        :return: value of the inverse transform at corresponding locations. if was done on input, returns heights for each point in the order of the fft input when creating the object
        """
        if locations is None:
            locations=self.normalizedPointLocation
        else:
            locations=self.normalize(locations)
        return self.reconCall(self.ordersToEval, locations, self.filteredSpectrum(),self.pointHeight.size)

    def reconstructDerivative(self,locations=None):
        if locations is None:
            locations=self.normalizedPointLocation
        else:
            locations=self.normalize(locations)
        return self.reconDcall(self.ordersToEval, locations, self.filteredSpectrum(), self.pointHeight.size)

    def avgSqrdReconstructionError(self):
        return np.mean((self.reconstruction()-self.pointHeight)**2)

    def freqGrid(self):
        return orderTuples(self.fullOrders)

    @classmethod
    def fromMeanPlane(cls,meanPlane, ordersToEval=None, normalizeMin=None, normalizeRange=None):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return SlowLegendreAnalyzer(meanPlane.inputResidual, meanPlane.inputInPlane, ordersToEval=ordersToEval, normalizeMin=normalizeMin, normalizeRange=normalizeRange)

def orderTuples(orders):
    freqProd=np.array(orders)
    if len(freqProd.shape)>1:
        freqProd=np.array(list(map(lambda arr: arr.flatten(), np.meshgrid(*freqProd))))
    else:
        freqProd.resize((1,len(freqProd)))
    return freqProd

def legendreToNewton(legCoeffs):
    return leg.Legendre(legCoeffs).convert(kind=np.polynomial.polynomial.Polynomial)

def newtonToLegendre(newtonCoeffs):
    return np.polynomial.polynomial.Polynomial(newtonCoeffs).convert(kind=leg.Legendre)

def genericLegVander(locations, deg):
    # stolen straight from legvander3d and modified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[leg.legvander(locations[:,i], deg[i]) for i in range(n)]
    indexingTuples=[]
    raise NotImplementedError
    # v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
    return v.reshape(v.shape[:-n] + (-1,))

def genericLegVal(locations, coeffs):
    if len(locations.shape)==1:
        return leg.legval(locations,coeffs)
    else: # assume dim 0 is points, dim 1 is dimensions
        c=leg.legval(locations[:,0], coeffs)
        for i in range(1, locations.shape[1]):
            c=leg.legval(locations[:,i],c, tensor=False)
        return c

def legvander4d(locations, deg):
    # stolen straight from legvander3d and mondified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[leg.legvander(locations[:,i], deg[i]) for i in range(n)]
    v = vi[0][..., None, None, None]*vi[1][..., None,:,None, None]*vi[2][..., None, None,:,None]*vi[3][...,None,None,None,:]
    return v.reshape(v.shape[:-n] + (-1,))

def legvander5d(locations, deg):
    # stolen straight from legvander3d and modified
    n=locations.shape[1]
    ideg = [int(d) for d in deg]
    is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
    if is_valid != [True,]*n:
        raise ValueError("degrees must be non-negative integers")

    vi=[leg.legvander(locations[:,i], deg[i]) for i in range(n)]
    v = vi[0][..., None, None, None, None]*vi[1][..., None,:,None, None, None]*vi[2][..., None, None,:,None,None]*vi[3][...,None,None,None,:,None]*vi[4][...,None,None,None,None,:]
    return v.reshape(v.shape[:-n] + (-1,))

def legForwardTransform(orders, locations, functionVals):
    if len(locations.shape)==1:
        return np.array(leg.legfit(locations, functionVals, orders[0]))
    else:
        if locations.shape[1]==2:
            V=leg.legvander2d(locations[:,0], locations[:,1], orders)
        elif locations.shape[1]==3:
            V=leg.legvander3d(locations[:,0],locations[:,1],locations[:,2], orders)
        elif locations.shape[1]==4:
            V=legvander4d(locations,orders)
        elif locations.shape[1]==5:
            V=legvander5d(locations,orders)
        else:
            raise NotImplementedError # there's a bad startup joke about this being good enough for the paper.
        ret, _, _, _=npl.lstsq(V, functionVals, rcond=None)
        return np.reshape(ret, (np.array(orders)+1).flatten())

def legReconstruct(orders, locations, coeffs,unusedNumPts):
    if len(locations.shape)==1:
        return np.array(leg.legval(locations, coeffs))
    else:
        if locations.shape[1] == 2:
            return leg.legval2d(locations[:,0], locations[:,1], coeffs)
        elif locations.shape[1] == 3:
            return leg.legval3d(locations[:,0],locations[:,1],locations[:,2], coeffs)
        else:
            return genericLegVal(locations, coeffs)

def legReconstructDerivative(freqs, locations, spectrum, numPts):
    """

    :param freqs:
    :param locations:
    :param spectrum:
    :param numPts:
    :return: derivative of iFFT of the spectrum as an array with shape (Npts, Ncomponent) or (Ncomponent,) if 1-d
    """
    deriv=leg.legder(spectrum, axis=0)
    return leg.legval(locations, deriv)

class SlowLegendreAnalyzer(PolynomialAnalyzer):
    def __init__(self,pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None):
        super().__init__(legForwardTransform, legReconstruct, legReconstructDerivative, pointHeight,pointLocation,ordersToEval, normalizeMin, normalizeRange)

def culledLegForwardTransform(orders, locations, functionVals, threshold=None):
    # inspired by : A Simple Regularization of the Polynomial Interpolation For the Runge Phenomemenon
    if len(locations.shape)==1:
        vandermonde=leg.legvander(locations, orders[0])
    elif locations.shape[1]==2:
        vandermonde=leg.legvander2d(locations[:,0], locations[:,1], orders)
    elif locations.shape[1]==3:
        vandermonde=leg.legvander3d(locations[:,0],locations[:,1],locations[:,2], orders)
    elif locations.shape[1]==4:
        vandermonde=legvander4d(locations,orders)
    elif locations.shape[1]==5:
        vandermonde=legvander5d(locations,orders)
    else:
        raise NotImplementedError # there's a bad startup joke about this being good enough for the paper.

    # preconditioner = np.diag((0.94) ** (2* (np.arange(vandermonde.shape[0]))))
    # vandermonde=np.dot(preconditioner, vandermonde)

    U,S,Vh=np.linalg.svd(vandermonde)
    numTake=0
    filtS=S
    if threshold is None:
        Eps= np.finfo(functionVals.dtype).eps
        Neps = np.prod(cmn.numpyze(orders)) * Eps * S[0] #truncation due to ill-conditioning
        Nt = max(np.argmax(Vh, axis=0)) #"Automatic" determination of threshold due to Runge's phenomenon
        threshold=min(Neps, Nt)
    while numTake<=0:
        filter=S>threshold
        numTake=filter.sum()
        if numTake>0:
            filtU=U[:,:numTake]; filtS=S[:numTake]; filtVh=Vh[:numTake, :]
        else:
            if threshold>1e-13:
                threshold=threshold/2
                warnings.warn('cutting threshold for eigenvalues to '+str(threshold))
            else:
                warnings('seems all eigenvalues are zero (<1e-13), setting to zero and breaking')
                filtS=np.zeros_like(S)
    truncVander=np.dot(filtU,np.dot(np.diag(filtS),filtVh))

    ret, _, _, _=npl.lstsq(truncVander, functionVals, rcond=None)
    return np.reshape(ret, np.array(orders).flatten()+1)

class CulledLegenderAnalyzer(PolynomialAnalyzer):
    def __init__(self,pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None):
        super().__init__(culledLegForwardTransform, legReconstruct, legReconstructDerivative, pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None)

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

class chebonDirectAnalyzer(PolynomialAnalyzer):
    def __init__(self,pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None):
        super().__init__(chebForwardTransform, chebReconstruct, chebReconstructDerivative, pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None)

class OptionNotSupportedError(Exception):
    pass

def run2danalysis(data,objHeaders=None,saveFigsPrepend=None,freqsToKeep=5, displayFigs=True, isMaxObj=None, ordersToRun=None):
    """

    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=mP.lowDimMeanPlane(data) # create the mean plane

    if saveFigsPrepend is not None:
        mps=saveFigsPrepend+'_meanPlane.png'
        trs=saveFigsPrepend+'_tradeRatios.png'
        spts=saveFigsPrepend+'_spectralPower_legendre.png'
        rts=saveFigsPrepend+'_reverseTransform.png'
    else:
        mps=None
        trs=None
        spts=None
        rts=None

    aC.runShowSaveClose(mp.draw2dMeanPlane,mps,displayFig=displayFigs)
    aC.runShowSaveClose(ft.partial(tMI.plotLogTradeRatios,mp,objHeaders),trs,displayFig=displayFigs)

    if ordersToRun is None:
        ordersToRun=min(freqsToKeep, 2*np.sqrt(data.shape[0])) # see: https://en.wikipedia.org/wiki/Runge%27s_phenomenon#Mitigations_to_the_problem

    fa=LegendreSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep,ordersToRun)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')

    aC.runShowSaveClose(ft.partial(aC.spectral1dPowerPlot_nonFFT,fa),spts,displayFig=displayFigs)

    aC.runShowSaveClose(ft.partial(aC.approximationPlot2d,mp,fa,objHeaders),rts,displayFig=displayFigs)
    # aC.runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)
    return (mp,fa)

def run3danalysis(data,objHeaders=None,saveFigsPrepend=None,freqsToKeep=5**2,displayFigs=True, ordersToRun=None):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=mP.lowDimMeanPlane(data) # create the mean plane
    aC.runShowSaveClose(mp.draw3dMeanPlane,saveFigsPrepend+'_meanPlane.png',displayFig=displayFigs)
    aC.runShowSaveClose(ft.partial(tMI.plotLogTradeRatios,mp,objHeaders),saveFigsPrepend+'_tradeRatios.png',displayFig=displayFigs)

    if ordersToRun is None:
        ordersToRun=min(freqsToKeep, 2*np.sqrt(np.sqrt(data.shape[0]))) # see: https://en.wikipedia.org/wiki/Runge%27s_phenomenon#Mitigations_to_the_problem
    fa=LegendreSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep, [ordersToRun,]*2)

    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')
    aC.runShowSaveClose(ft.partial(aC.spectral2dPowerImage_nonFFT,fa),saveFigsPrepend+'_spectralPower.png',displayFig=displayFigs)
    aC.runShowSaveClose(ft.partial(aC.spectral2dPowerPlot_nonFFT,fa),saveFigsPrepend+'_spectralPower3d.png',displayFig=displayFigs)
    aC.runShowSaveClose(ft.partial(aC.approximationPlot3d,mp,fa),saveFigsPrepend+'_reverseTransform.png',displayFig=displayFigs)
    aC.runShowSaveClose(ft.partial(aC.plot3dErr,mp.inputInPlane,mp.inputResidual),saveFigsPrepend+'_errorPlot.png',displayFig=displayFigs)
    aC.runShowSaveClose(fa.powerDeclineReport,saveFigsPrepend+'_powerDeclineReport.png',displayFig=displayFigs)
    # aC.runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

def runHighDimAnalysis(data, objHeaders=None, saveFigsPrepend=None,freqsToKeep=None,displayFigs=True, ordersToRun=None):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=mP.lowDimMeanPlane(data) # create the mean plane

    aC.runShowSaveClose(ft.partial(tMI.plotLogTradeRatios,mp,objHeaders),saveFigsPrepend+'_tradeRatios.png',displayFig=displayFigs)

    if freqsToKeep is None:
        freqsToKeep=2**data.shape[1]
    dim=data.shape[1]-1
    if ordersToRun is None:
        ordersToRun=min(freqsToKeep, 2*(data.shape[0])**(1/(2*dim)))
    fa=LegendreSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep,[ordersToRun,]*dim)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')

    aC.runShowSaveClose(fa.powerDeclineReport,saveFigsPrepend+'_powerDeclinePlot.png',displayFig=displayFigs)
    # aC.runShowSaveClose(ft.partial(tMI.plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

class PolynomialSummarizer():
    def __init__(self,numToTake,wavelenth=None):
        self.numTake=numToTake
        self.freqsTaken=[]
        self.freqSpectra=[]
        self.indcies=[]
        self.wavelength=wavelenth
        self.hasRun=False
        self.lostPower=None

    #TODO: Update
    def __findFreqs(self,freqs,spectrum):
        self.hasRun=True
        spectralPower=np.abs(spectrum)**2
        sortIndx=np.argsort(spectralPower.flatten())[::-1]
        toTake=sortIndx[:min(self.numTake,spectralPower.size)]
        notTaken=sortIndx[min(self.numTake,spectralPower.size):]
        self.indcies=toTake
        if len(freqs.shape)>1:
            ft=orderTuples(freqs).T
            self.freqsTaken=ft[toTake,:]
        else:
            self.freqsTaken=freqs[toTake]
        self.freqSpectra=spectrum.flatten()[toTake]
        self.lostPower=np.sum(np.abs(spectralPower.flatten()[notTaken])**2)
        return toTake

    def __call__(self,freqs,spectrum):
        return self.filtering(freqs,spectrum)

    #TODO: Update
    def filtering(self,freqs,spectrum):
        takeIndx=self.__findFreqs(freqs,spectrum)
        outSpectrum=np.zeros_like(spectrum)
        multiTakeIndx=np.unravel_index(takeIndx,spectrum.shape)
        outSpectrum[multiTakeIndx]=spectrum[multiTakeIndx]
        return outSpectrum

    def _toDataFrame(self):
        if not self.hasRun:
            raise cmn.InitializeRunFailError
        if len(self.freqsTaken.shape)>1:
            toUse=np.squeeze(self.freqsTaken[:,0]>0)
            ft=self.freqsTaken[toUse,:]
        else:
            toUse=self.freqsTaken>0
            ft=self.freqsTaken[toUse]
        fs=self.freqSpectra[toUse]
        if len(ft.shape)>1:
            d=dict()
            for i, ftarr in enumerate(ft.T):
                d['frequency, dim: '+str(i)]=ftarr
        else:
            d={'frequency: ': ft}
        d['spectral power']=np.abs(fs)**2
        d['spectral phase']=np.angle(fs, deg=True)
        if self.wavelength is not None:
            if not hasattr(self.wavelength,'len') or len(self.wavelength)==1:
                numHump=ft*self.wavelength
            else:
                numHump=ft*self.wavelength[np.newaxis,:]
            for i, humpArr in enumerate(numHump.T):
                d['number of humps dim '+str(i)]=humpArr
        return pd.DataFrame(d)

    def powerDeclineReport(self):
        # plt.fill(np.arange(len(self.freqSpectra)),np.abs(self.freqSpectra)**2)
        # plt.plot(np.arange(len(self.freqSpectra)),np.abs(self.freqSpectra)**2)
        plt.bar(np.arange(len(self.freqSpectra)),np.abs(self.freqSpectra)**2,color=cmn.globalBarPlotColor)
        if len(self.freqsTaken.shape)>1:
            plt.xticks(range(len(self.freqsTaken)),cmn.multiDimNumpyToPrettyStr(self.freqsTaken), rotation=75)
        else:
            plt.xticks(range(len(self.freqsTaken)),cmn.numpyToPrettyStr(self.freqsTaken), rotation=75)
        plt.ylabel('squared power of component')
        plt.xlabel('representative frequency')

    def report(self, tofile=None):
        if tofile is None:
            print(self._toDataFrame())
            print('captured power: '+str(np.sum(np.abs(self.freqSpectra)**2)))
            print('lost power: '+str(self.lostPower))
        else:
            self._toDataFrame().to_csv(tofile)
            with open(tofile,'a') as f:
                f.writelines(('captured power: '+str(np.sum(np.abs(self.freqSpectra)**2)), 'lost power: '+str(self.lostPower)))

class LegendreSummarizerAnalyzer(CulledLegenderAnalyzer):
    """
    mixin summarizer and analyzer. Recommended way to build and run.
    """
    def __init__(self,pointHeight,pointLocation,ordersToEval=None,freqsToKeep=5):
        super().__init__(pointHeight, pointLocation, ordersToEval=ordersToEval)
        self.summarizer=PolynomialSummarizer(freqsToKeep,wavelenth=np.ptp(self.pointLocation,axis=0))
        self.addSpectralFilter(self.summarizer)

    def report(self, tofile=None):
        try:
            self.summarizer.report(tofile)
        except(cmn.InitializeRunFailError):
            self.filteredSpectrum() # hack to force computation
            self.summarizer.report(tofile)

    def powerDeclineReport(self):
        try:
            self.summarizer.powerDeclineReport()
        except(cmn.InitializeRunFailError):
            self.filteredSpectrum() # hack to force computation
            self.summarizer.powerDeclineReport()

    @classmethod
    def fromMeanPlane(cls,meanPlane,freqsToKeep=5,ordersToEval=None):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return LegendreSummarizerAnalyzer(meanPlane.inputResidual, meanPlane.inputInPlane, ordersToEval=ordersToEval, freqsToKeep=freqsToKeep)

if __name__=="__main__":
    numsmpl=30

    # demo finding the mean plane in 2d
    # seedList=np.linspace(0,np.pi/2,numsmpl)
    seedList=np.sort(np.random.rand(numsmpl)*np.pi/2)
    dummyTest2d=np.vstack((np.sin(seedList),np.cos(seedList))).T

    # run2danalysis(dummyTest2d,saveFigsPrepend='testSave')
    # run2danalysis(dummyTest2d)

    # seedList=np.concatenate((np.linspace(0,0.15,16), np.linspace(0.3, 0.6, 32), np.linspace(0.85, 1, 16)))
    seedList=np.random.sample(64)
    # seedList=np.linspace(0,1,16)
    # seedList=2*seedList-1

    # testList=np.linspace(-1,1,128)
    testList=np.linspace(0,1,128)

    # y=np.sin(2*np.pi*seedList)
    y=np.sin(2*np.pi*seedList)+ 0.1*np.random.sample(len(seedList))
    fa=SlowLegendreAnalyzer(y, seedList, ordersToEval=2*np.sqrt(len(seedList)), normalizeMin=0, normalizeRange=1)
    # fa=SlowLegendreAnalyzer(y, seedList, ordersToEval=np.sqrt(len(seedList)))
    # fa=SlowLegendreAnalyzer(y, seedList)
    derivatives=np.array([fa.reconstructDerivative(x) for x in testList])


    plt.figure()
    plt.plot(testList,np.sin(2*np.pi*testList),'ro',testList,fa.reconstruction(testList),'b-',seedList,np.sin(2*np.pi*seedList),'ko')
    # plt.figure()
    # plt.plot(testList,2*np.pi*np.cos(2*np.pi*testList),'ro',testList,derivatives,'b-')
    plt.show()
    aC.spectral1dPowerPlot_nonFFT(fa)
    plt.show()

    #draw basis functions
    # for n in range(6):
    #     polycoeffs=polynomialCoeffs(n)
    #     lsp=np.linspace(-1,1,256)
    #     x=np.array([lsp**k for k in reversed(list(range(n+1)))])
    #     x=np.vander(lsp,n+1)
    #     interps=np.dot(polycoeffs,x)
    #     plt.plot(lsp,interps)
    # plt.show()
