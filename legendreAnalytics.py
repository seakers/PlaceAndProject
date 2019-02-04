import numpy as np
import scipy as sp
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import pca
import functools as ft
from meanPlane import *
import scipy.ndimage.filters as spndf

from common import *
from tradeoffMatrixImage import *

from numpy.polynomial import legendre as pl

class SlowLegendreAnalyzer():
    """
    https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Multidimensional_DFT
    https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform#2-D_NDFT
    """
    def __init__(self,pointHeight,pointLocation,ordersToEval=None):
        """
        initializes the FourierAnalyzer object
        :param pointHeight:
        :param pointLocation:
        """
        self.pointLocation=pointLocation
        self.pointHeight=pointHeight
        if ordersToEval is None:
            ordersToEval=np.arange(2,len(pointHeight))
        self.ordersToEval=ordersToEval
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

    @property
    def spectrum(self):
        """
        :return: the one-sided spectrum of the pointHeights and locations input when creating the analyzer
        """
        # instead product out the list of frequencies and then calculate
        """

        :return:
        """
        return self.filteredSpectrum()

    def filteredSpectrum(self):
        return filteredSpectrum(self.inputFilters,self.spectralFilters,self.ordersToEval, self.pointLocation, self.pointHeight)

    def trueSpectrum(self):
        freqProd=orderTuples(self.ordersToEval)
        if len(self.pointLocation.shape)==1:
            pointLoc=self.pointLocation[:,np.newaxis]
        else:
            pointLoc=self.pointLocation
        polyTerm=
        exponentTerm=-2*np.pi*1j*np.dot(pointLoc,freqProd)
        # return np.dot(self.pointHeight,np.exp(exponentTerm)).reshape(self.numFreqs)/self.pointHeight.size
        if len(self.ordersToEval.shape)>1:
            numFreqs=list(map(len, self.ordersToEval))
        else:
            numFreqs=len(self.ordersToEval)
        return np.dot(self.pointHeight,polyTerm.reshape(numFreqs)

        return forwardTransform(self.ordersToEval, self.pointLocation, self.pointHeight)

    def reconstruction(self, locations=None):
        """
        runs fourier series defined by this analysis.
        :param locations: locations to evaluate at. if None (default) evaluates on the input locations
        :return: value of the inverse transform at corresponding locations. if was done on input, returns heights for each point in the order of the fft input when creating the object
        """
        if locations is None:
            locations=self.pointLocation
        return reconstruction(self.ordersToEval,locations,self.filteredSpectrum(),self.pointHeight.size)

    def reconstructDerivative(self,locations=None):
        if locations is None:
            locations=self.pointLocation
        return reconstructDerivative(self.ordersToEval,locations, self.spectrum, self.pointHeight.size)

    def avgSqrdReconstructionError(self):
        return np.mean((self.reconstruction()-self.pointHeight)**2)

    def freqGrid(self):
        return orderTuples(self.ordersToEval)

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return SlowLegendreAnalyzer(meanPlane.inputResidual, meanPlane.inputInPlane)

def forwardTransform(freqs, locations, functionVals):
    freqProd=orderTuples(freqs)
    if len(locations.shape)==1:
        pointLoc=locations[:,np.newaxis]
    else:
        pointLoc=locations
    polyTerm=
    exponentTerm=-2*np.pi*1j*np.dot(pointLoc,freqProd)
    # return np.dot(self.pointHeight,np.exp(exponentTerm)).reshape(self.numFreqs)/self.pointHeight.size
    if len(freqs.shape)>1:
        numFreqs=list(map(len, freqs))
    else:
        numFreqs=len(freqs)
    return np.dot(functionVals,np.exp(exponentTerm)).reshape(numFreqs)

def orderTuples(orders):
    freqProd=np.array(orders)
    if len(freqProd.shape)>1:
        freqProd=np.array(list(map(lambda arr: arr.flatten(), np.meshgrid(*freqProd))))
    else:
        freqProd.resize((1,len(freqProd)))
    return freqProd


def stdNumPerSide(dataShape):
    nthRoot=int(np.ceil(dataShape[0]**(1/dataShape[1])))
    return (nthRoot,)*dataShape[1]

def filteredSpectrum(inputFilters, spectralFilters,frequencies,locations,heights,forwardTransform=forwardTransform):
    filtLoc=locations
    filtHeight=heights
    for filt in inputFilters:
        filtLoc, filtHeight=filt(filtLoc,filtHeight)
    ret=forwardTransform(frequencies, filtLoc, filtHeight)
    for filt in spectralFilters:
        ret=filt(frequencies,ret)
    return ret

def reconstruction(freqs, locations, spectrum,numPts):

    freqProd=np.array(freqs)
    if len(freqProd.shape)>1:
        freqProd=np.array(list(map(lambda arr: arr.flatten(), np.meshgrid(*freqProd))))
    else:
        freqProd.resize((1,len(freqProd)))
    spectrum=spectrum.flatten()
    spectrum=spectrum[np.newaxis,:]
    if len(locations.shape)==1:
        pointLoc=locations[:,np.newaxis]
    else:
        pointLoc=locations
    exponentTerm=2*np.pi*1j*np.dot(freqProd.T,pointLoc.T)
    return np.squeeze(np.dot(spectrum,np.exp(exponentTerm)))/numPts

def reconstructDerivative(freqs, locations, spectrum, numPts):
    """

    :param freqs:
    :param locations:
    :param spectrum:
    :param numPts:
    :return: derivative of iFFT of the spectrum as an array with shape (Npts, Ncomponent) or (Ncomponent,) if 1-d
    """
    if not len(locations.shape) == 0 and locations.shape[1] != 1:
        assert locations.shape[0]==1 and locations.shape[1]>1
        # TODO make consistent behavior. should have 1st index index over array of locations, 2nd index index is over components of output, 3rd is over input components
        accum=[]
        for compIndx in range(locations.size):
            broadcastArr=np.ones(len(spectrum.shape))
            broadcastArr[compIndx]=freqs.shape[1]
            freqMultiplier=2*np.pi*1j*np.reshape(np.squeeze(freqs[compIndx,:]),broadcastArr)
            accum.append(reconstruction(freqs, locations, freqMultiplier*spectrum, numPts))
        return np.array(accum).T
    else:
        return reconstruction(freqs,locations,2*np.pi*1j*freqs*spectrum,numPts) # should be scalar

def orderLocations1d(pointLocations):
    """
    returns an indexing array to place points in the proper locations
    :param pointLocations:
    :return:
    """
    return np.argsort(pointLocations)

def orderLocations(xy):
    """
    reorders elements of an array-of-arrays such that the elements are in a monotonic order.


    :param xy: an nxd array of locations in 2d to sort
    :return:
    """
    raise NotImplementedError

class OptionNotSupportedError(Exception):
    pass

def spectral1dPowerPlot(analyzerObj):
    spectralPower=np.abs(np.fft.fftshift(analyzerObj.spectrum))**2
    plt.plot(np.fft.fftshift(analyzerObj.fftFreqs),spectralPower,'k.-')
    # axis_font={'size':'28'}
    # plt.xlabel('frequency',**axis_font)
    # plt.ylabel('square power',**axis_font)
    plt.xlabel('frequency')
    plt.ylabel('square power')
def spectral1dPhasePlot(fourierAnalyzerObj):
    spectralPhase=np.angle(np.fft.fftshift(fourierAnalyzerObj.spectrum))
    # print(spectralPhase)
    plt.plot(np.fft.fftshift(fourierAnalyzerObj.fftFreqs),spectralPhase,'k.-')
    # axis_font={'size':'28'}
    # plt.xlabel('frequency',**axis_font)
    # plt.ylabel('phase (radians)',**axis_font)
    plt.xlabel('frequency')
    plt.ylabel('phase (radians)')

def spectral2dPowerPlot(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    freqProd=np.meshgrid(*fourierAnalyzerObj.fftFreqs, indexing='ij')
    ax=prep3dAxes()
    ax.plot_surface(freqProd[0],freqProd[1],spectralPower)

def spectral2dPowerImage(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    # plt.imshow(np.fft.fftshift(spectralPower),cmap=globalCmap,interpolation='nearest')
    plt.imshow(np.fft.fftshift(spectralPower),cmap=globalCmap,interpolation='nearest')
    plt.colorbar()
    shiftedFFTF=np.fft.fftshift(fourierAnalyzerObj.fftFreqs,axes=1)
    plt.xticks(np.arange(len(shiftedFFTF[0])), numpyToPrettyStr(shiftedFFTF[0]), rotation=60)
    plt.yticks(np.arange(len(shiftedFFTF[1])), numpyToPrettyStr(shiftedFFTF[1]))
    # plt.xticks(np.arange(len(fourierAnalyzerObj.fftFreqs[0])), fourierAnalyzerObj.fftFreqs[0], rotation=60)
    # plt.yticks(np.arange(len(fourierAnalyzerObj.fftFreqs[1])), fourierAnalyzerObj.fftFreqs[1])

def approximationPlot2d(meanPlane, analyzer,objLabels=None):
    dummyTest2d=meanPlane.paretoSamples
    mp=meanPlane
    # plt.plot(mp._centeredSamples[:,0],mp._centeredSamples[:,1])
    plt.plot(dummyTest2d[:,0],dummyTest2d[:,1],'k.',label='Pareto Points')
    plt.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],'kx',label='ProjectedLocations')
    spectralCurveInPlane=np.linspace(mp.inputInPlane.min(),mp.inputInPlane.max(),10*mp.inputInPlane.size)
    planeCurve=np.dot(spectralCurveInPlane[:,np.newaxis],np.squeeze(mp.basisVects)[np.newaxis,:])+mp.meanPoint[np.newaxis,:]
    plt.plot(planeCurve[:,0],planeCurve[:,1],'k--',label='mean plane')

    filteredCorrection=analyzer.reconstruction()
    spectralCurveOutOfPlane=analyzer.reconstruction(spectralCurveInPlane)
    spectralCurve=planeCurve+np.dot(spectralCurveOutOfPlane[:,np.newaxis],mp.normalVect[np.newaxis,:])
    plt.plot(spectralCurve[:,0],spectralCurve[:,1],'k-',label='reconstructed curve')

    # corrected=mp.inputProjections+np.dot(filteredCorrection[:,np.newaxis],mp.normalVect[np.newaxis,:])
    # plt.plot(corrected[:,0],corrected[:,1],'.',label='spectral representation of inputs')

    plt.axis('equal')
    if objLabels is not None:
        plt.xlabel(objLabels[0])
        plt.ylabel(objLabels[1])
    plt.legend(loc='best')

def plot3dErr(locations, values):
    pltVals=interpolateErr(locations,values)
    plt.imshow(pltVals, cmap=globalCmap, origin='lower',extent=(np.min(locations[:,0]),np.max(locations[:,0]), np.min(locations[:,1]),np.max(locations[:,1])))
    plt.colorbar()
    plt.plot(locations[:,0],locations[:,1], 'k.')

def runShowSaveClose(toPlot, saveName=None, displayFig=True):
    plt.figure()
    toPlot()
    if saveName is not None:
        plt.savefig(saveName,bbox_inches='tight')
    plt.show()
    if not displayFig:
        plt.close('all')

def run2danalysis(data,objHeaders=None,saveFigsPrepend=None,freqsToKeep=2, displayFigs=True, isMaxObj=None):
    """

    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane

    runShowSaveClose(mp.draw2dMeanPlane,saveFigsPrepend+'_meanPlane.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotLogTradeRatios,mp,objHeaders),saveFigsPrepend+'_tradeRatios.png',displayFig=displayFigs)

    fa=ChebySummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')

    runShowSaveClose(ft.partial(spectral1dPowerPlot,fa),saveFigsPrepend+'_spectralPower.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(spectral1dPhasePlot,fa),saveFigsPrepend+'_spectralPhase.png',displayFig=displayFigs)

    plt.figure()
    spectralPower=np.abs(np.fft.fftshift(fa.trueSpectrum()))**2
    plt.plot(np.fft.fftshift(fa.fftFreqs),spectralPower,'k.-')
    # axis_font={'size':'28'}
    # plt.xlabel('frequency',**axis_font)
    # plt.ylabel('square power',**axis_font)
    plt.xlabel('frequency')
    plt.ylabel('square power')
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_trueSpectralPower.png',bbox_inches='tight')
    plt.show()
    if not displayFigs:
        plt.close('all')

    runShowSaveClose(ft.partial(approximationPlot2d,mp,fa,objHeaders),saveFigsPrepend+'_reverseTransform.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)
    return (mp,fa)

def run3danalysis(data,objHeaders=None,saveFigsPrepend=None,freqsToKeep=3**2,displayFigs=True):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane
    runShowSaveClose(mp.draw3dMeanPlane,saveFigsPrepend+'_meanPlane.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotLogTradeRatios,mp,objHeaders),saveFigsPrepend+'_tradeRatios.png',displayFig=displayFigs)

    fa=FourierSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')
    runShowSaveClose(ft.partial(spectral2dPowerImage,fa),saveFigsPrepend+'_spectralPower.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(spectral2dPowerPlot,fa),saveFigsPrepend+'_spectralPower3d.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(approximationPlot3d,mp,fa),saveFigsPrepend+'_reverseTransform.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plot3dErr,mp.inputInPlane,mp.inputResidual),saveFigsPrepend+'_errorPlot.png',displayFig=displayFigs)
    runShowSaveClose(fa.powerDeclineReport,saveFigsPrepend+'_powerDeclineReport.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

def approximationPlot3d(mp,fa):
    grid_x,grid_y=np.meshgrid(np.linspace(np.min(mp.inputInPlane[:,0]),np.max(mp.inputInPlane[:,0])), np.linspace(np.min(mp.inputInPlane[:,1]),np.max(mp.inputInPlane[:,1])))
    points=np.vstack((grid_x.flatten(),grid_y.flatten())).T
    recons=np.real(fa.reconstruction(locations=points)).reshape(grid_x.shape)
    plt.imshow(recons, cmap=globalCmap,origin='lower',extent=(np.min(mp.inputInPlane[:,0]),np.max(mp.inputInPlane[:,0]), np.min(mp.inputInPlane[:,1]),np.max(mp.inputInPlane[:,1])))
    plt.plot(mp.inputInPlane[:,0],mp.inputInPlane[:,1],'k.')
    plt.colorbar()

def runHighDimAnalysis(data, objHeaders=None, saveFigsPrepend=None,freqsToKeep=None,displayFigs=True):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane

    runShowSaveClose(ft.partial(plotLogTradeRatios,mp,objHeaders),saveFigsPrepend+'_tradeRatios.png',displayFig=displayFigs)

    if freqsToKeep is None:
        freqsToKeep=2**data.shape[1]
    fa=FourierSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')

    runShowSaveClose(fa.powerDeclineReport,saveFigsPrepend+'_powerDeclinePlot.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

class FourierSummarizer():
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
        toTake=sortIndx[:min(2*self.numTake,spectralPower.size)] # assume symmetry works
        notTaken=sortIndx[min(2*self.numTake,spectralPower.size):]
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
            raise InitializeRunFailError
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
        plt.bar(np.arange(len(self.freqSpectra)),np.abs(self.freqSpectra)**2,color=globalBarPlotColor)
        if len(self.freqsTaken.shape)>1:
            plt.xticks(range(len(self.freqsTaken)),multiDimNumpyToPrettyStr(self.freqsTaken), rotation=75)
        else:
            plt.xticks(range(len(self.freqsTaken)),numpyToPrettyStr(self.freqsTaken), rotation=75)
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

class ChebySummarizerAnalyzer(SlowLegendreAnalyzer):
    # class FourierSummarizerAnalyzer(FourierAnalyzer): # somehow phase is off or somethign in the really easy problems
    def __init__(self,pointHeight,pointLocation,frequenciesToEval=None,freqsToKeep=5):
        super(SlowLegendreAnalyzer, self).__init__(pointHeight, pointLocation, frequenciesToEval)
        self.summarizer=FourierSummarizer(freqsToKeep,wavelenth=np.ptp(self.pointLocation,axis=0))
        self.addSpectralFilter(self.summarizer)

    def report(self, tofile=None):
        try:
            self.summarizer.report(tofile)
        except(InitializeRunFailError):
            self.filteredSpectrum() # hack to force computation
            self.summarizer.report(tofile)

    def powerDeclineReport(self):
        try:
            self.summarizer.powerDeclineReport()
        except(InitializeRunFailError):
            self.filteredSpectrum() # hack to force computation
            self.summarizer.powerDeclineReport()

    @classmethod
    def fromMeanPlane(cls,meanPlane,freqsToKeep=5):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return SlowLegendreAnalyzer(meanPlane.inputResidual, meanPlane.inputInPlane, freqsToKeep=freqsToKeep)

if __name__=="__main__":
    numsmpl=30

    # demo finding the mean plane in 2d
    # seedList=np.linspace(0,np.pi/2,numsmpl)
    # seedList=np.sort(np.random.rand(numsmpl)*np.pi/2)
    # dummyTest2d=np.vstack((np.sin(seedList),np.cos(seedList))).T

    # run2danalysis(dummyTest2d,saveFigsPrepend='testSave')
    # run2danalysis(dummyTest2d)

    seedList=np.linspace(0,1,64)
    y=np.sin(2*np.pi*seedList)
    fa=SlowLegendreAnalyzer(y, seedList)
    derivatives=np.array([fa.reconstructDerivative(x) for x in seedList])
    plt.figure()
    plt.plot(seedList,y,seedList,fa.reconstruction(seedList))
    plt.figure()
    plt.plot(seedList,2*np.pi*np.cos(2*np.pi*seedList),seedList,derivatives)
    plt.show()
# TODO: Report basis vector directions for interpretign direction in fourier analysis
# TODO: make sure normal vector points in one particular direction