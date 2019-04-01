import numpy as np
import numpy.linalg as npl
import functools as ft
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as leg
import warnings

import common as cmn
import meanPlane as mP
import tradeoffMatrixImage as tMI
import analyticsCommon as aC

#nice reference: http://mathfaculty.fullerton.edu/mathews/n2003/LegendrePolyMod.html
from polyfitting.polycommon import PolynomialAnalyzer, PolynomialSummarizer

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
    # aC.runShowSaveClose(ft.partial(tMI.plotLogTradeRatios,mp,objHeaders),trs,displayFig=displayFigs)

    if ordersToRun is None:
        ordersToRun=min(freqsToKeep, 2*np.sqrt(data.shape[0])) # see: https://en.wikipedia.org/wiki/Runge%27s_phenomenon#Mitigations_to_the_problem

    fa=LegendreSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep,ordersToRun)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')

    aC.runShowSaveClose(ft.partial(aC.spectral1dPowerPlot_nonFFT,fa),spts,displayFig=displayFigs)

    aC.runShowSaveClose(ft.partial(aC.approximationPlot2d,mp,fa,objHeaders),rts,displayFig=displayFigs)
    aC.runShowSaveClose(ft.partial(tMI.plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)
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
    aC.runShowSaveClose(ft.partial(tMI.plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

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
    aC.runShowSaveClose(ft.partial(tMI.plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

class LegendreSummarizerAnalyzer(CulledLegenderAnalyzer):
    """
    mixin summarizer and analyzer. Recommended way to build and run.
    """
    def __init__(self,pointHeight,pointLocation,ordersToEval=None,freqsToKeep=5):
        super().__init__(pointHeight, pointLocation, ordersToEval=ordersToEval)
        self.summarizer= PolynomialSummarizer(freqsToKeep, wavelenth=np.ptp(self.pointLocation, axis=0))
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
