import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import pca
import functools as ft
from meanPlane import *

from common import *

def incToEven(i):
    return i+(i%2)
def decToEven(i):
    return i-(i%2)

class SlowFourierAnalyzer():
    """
    https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Multidimensional_DFT
    https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform#2-D_NDFT
    """
    def __init__(self,pointHeight,pointLocation,frequenciesToEval=None):
        """
        initializes the FourierAnalyzer object
        :param pointHeight:
        :param pointLocation:
        """
        self.pointLocation=pointLocation
        self.pointHeight=pointHeight
        if frequenciesToEval is None:
            if len(pointLocation.shape)>1:
                nthRoot=incToEven(np.ceil(pointHeight.size**(1/pointLocation.shape[1])))
                frequencyDivisions=(nthRoot,)*pointLocation.shape[1]
            ranges=np.ptp(self.pointLocation,axis=0)
            # __oneFreqRange=lambda range, n: np.concatenate((np.linspace(0,range,n//2+1),np.linspace(0,range,incToEven(n)/2+1)[::-1]))
            if len(self.pointLocation.shape)==1:
                frequenciesToEval=1/(ranges)*np.concatenate((np.arange(1,pointHeight.size//2+1),-np.arange(1,incToEven(pointHeight.size)/2+1)[::-1]))
                # frequenciesToEval=__oneFreqRange(ranges,pointHeight.size)
                # frequenciesToEval=1/(ranges)*np.arange(1,incToEven(pointHeight.size)/2)
            else:
                frequenciesToEval=np.dot(1/(ranges[:,np.newaxis]),np.concatenate((np.arange(1,pointHeight.size//2),-np.arange(1,incToEven(pointHeight.size)/2)[::-1]))[np.newaxis,:])
                # frequenciesToEval=np.array(list(map(__oneFreqRange,ranges,it.repeat(pointHeight.size))))
                # frequenciesToEval=np.dot(1/(ranges[:,np.newaxis]),np.arange(1,incToEven(pointHeight.size)/2)[np.newaxis,:])
        self.fftFreqs=numpyze(frequenciesToEval)
        self.realInput=False
        if len(pointLocation.shape)==1:
            self.numFreqs=len(frequenciesToEval)
        else:
            self.numFreqs=tuple(map(len,frequenciesToEval))
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

    # @property
    # def __freqSumMat(self):
    #     numel=self.pointHeight.size
    #     powChngMat,freqChngMat=np.meshgrid(np.arange(0,numel),np.linspace(0,np.max(self.pointLocation))) # TODO: generalize for higher dimensions
    #     np.exp(powChngMat*freqChngMat*1j*self.pointLocation)

    @property
    def spectrum(self):
        """
        :return: the one-sided spectrum of the pointHeights and locations input when creating the analyzer
        """
        # instead product out the list of frequencies and then calculate
        return fowardTransform(self.fftFreqs,self.pointLocation, self.pointHeight)

    def filteredSpectrum(self):
        """

        :return:
        """
        filtLoc=self.pointLocation
        filtHeight=self.pointHeight
        for filt in self.inputFilters:
            filtLoc, filtHeight=filt(filtLoc,filtHeight)
        ret=fowardTransform(self.fftFreqs, filtLoc, filtHeight)
        for filt in self.spectralFilters:
            ret=filt(ret)
        return ret

    def reconstruction(self, locations=None):
        """
        runs fourier series defined by this analysis.
        :param locations: locations to evaluate at. if None (default) evaluates on the input locations
        :return: value of the inverse transform at corresponding locations. if was done on input, returns heights for each point in the order of the fft input when creating the object
        """
        if locations is None:
            locations=self.pointLocation
        return reconstruction(self.fftFreqs,locations,self.filteredSpectrum(),self.pointHeight.size)

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return SlowFourierAnalyzer(meanPlane.inputResidual,meanPlane.inputInPlane)

def fowardTransform(freqs, locations,functionVals):
    freqProd=np.array(freqs)
    if len(freqProd.shape)>1:
        freqProd=np.array(list(map(lambda arr: arr.flatten(), np.meshgrid(*freqProd))))
    else:
        freqProd.resize((1,len(freqProd)))
    if len(locations.shape)==1:
        pointLoc=locations[:,np.newaxis]
    else:
        pointLoc=locations
    exponentTerm=-2*np.pi*1j*np.dot(pointLoc,freqProd)
    # return np.dot(self.pointHeight,np.exp(exponentTerm)).reshape(self.numFreqs)/self.pointHeight.size
    numFreqs=list(map(len, freqs))
    return np.dot(functionVals,np.exp(exponentTerm)).reshape(numFreqs)

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

# def gaussBlur():
    #need to verify this works in arbitrary dimension.
    # scipy.ndimage.filters.gaussian_filter(input, sigma, truncate=)

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

class FourierAnalyzer():
    """
    resource on learning what the multidimensional transform is and does: https://see.stanford.edu/materials/lsoftaee261/chap8.pdf
    python nonuniform FFT: https://pypi.python.org/pypi/pynufft/0.3.2.8
    Some other libraries: http://dsp.stackexchange.com/questions/16590/non-uniform-fft-with-fftw
    """
    def __init__(self,pointHeight,pointLocation):
        """
        initializes the FourierAnalyzer object
        :param pointHeight:
        :param pointLocation:
        """
        orderingArray=orderLocations1d(pointLocation)
        self.pointHeight=pointHeight[orderingArray]
        self.pointLocation=pointLocation[orderingArray]
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
        return np.fft.fft(self.pointHeight)

    def filteredSpectrum(self):
        """

        :return:
        """
        ret=self.spectrum
        for filt in self.spectralFilters:
            ret=filt(ret)
        return ret

    @property
    def fftFreqs(self):
        """returns the frequencies at which the spectrum is evaluated"""
        return np.fft.fftfreq(len(self.pointHeight),d=np.mean(np.diff(self.pointLocation)))

    def reconstruction(self,locations=None):
        """
        runs fourier series defined by this analysis.
        :param locations: locations to evaluate at. if None (default) evaluates on the input locations
        :return: value of the inverse transform at corresponding locations. if was done on input, returns heights for each point in the order of the fft input when creating the object
        """
        if locations is None:
            return np.fft.ifft(np.squeeze(self.spectrum), self.pointHeight.size)
        else:
            if locations is None:
                locations=self.pointLocation
            return reconstruction(self.fftFreqs,locations,self.filteredSpectrum(),self.pointHeight.size)

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return FourierAnalyzer(meanPlane.inputResidual,meanPlane.inputInPlane)

def spectral1dPowerPlot(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    print(spectralPower)
    plt.plot(fourierAnalyzerObj.fftFreqs,spectralPower,'.-')
    plt.xlabel('frequency')
    plt.ylabel('square power')

def spectral1dPhasePlot(fourierAnalyzerObj):
    spectralPhase=np.angle(fourierAnalyzerObj.spectrum)
    print(spectralPhase)
    plt.plot(fourierAnalyzerObj.fftFreqs,spectralPhase,'.-')
    plt.xlabel('frequency')
    plt.ylabel('phase (radians)')

def spectral2dPowerPlot(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    freqProd=np.meshgrid(*fourierAnalyzerObj.fftFreqs, indexing='ij')
    ax=prep3dAxes()
    ax.plot_surface(freqProd[0],freqProd[1],spectralPower)

def spectral2dPowerImage(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    plt.imshow(spectralPower,cmap='Greys',interpolation='nearest')
    plt.colorbar()
    # plt.xticks(fourierAnalyzerObj.fftFreqs[0])
    # plt.yticks(fourierAnalyzerObj.fftFreqs[1])

def approximationPlot2d(meanPlane, analyzer):
    dummyTest2d=meanPlane.paretoSamples
    mp=meanPlane
    # plt.plot(mp._centeredSamples[:,0],mp._centeredSamples[:,1])
    plt.plot(dummyTest2d[:,0],dummyTest2d[:,1],'.',label='Pareto Points')
    plt.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],'.',label='ProjectedLocations')
    spectralCurveInPlane=np.linspace(mp.inputInPlane.min(),mp.inputInPlane.max(),10*mp.inputInPlane.size)
    planeCurve=np.dot(spectralCurveInPlane[:,np.newaxis],np.squeeze(mp.basisVects)[np.newaxis,:])+mp.meanPoint[np.newaxis,:]
    plt.plot(planeCurve[:,0],planeCurve[:,1])

    filteredCorrection=analyzer.reconstruction()
    spectralCurveOutOfPlane=analyzer.reconstruction(spectralCurveInPlane)
    spectralCurve=planeCurve+np.dot(spectralCurveOutOfPlane[:,np.newaxis],mp.normalVect[np.newaxis,:])
    plt.plot(spectralCurve[:,0],spectralCurve[:,1],label='reconstructed curve')

    # corrected=mp.inputProjections+np.dot(filteredCorrection[:,np.newaxis],mp.normalVect[np.newaxis,:])
    # plt.plot(corrected[:,0],corrected[:,1],'.',label='spectral representation of inputs')

    plt.axis('equal')
    plt.legend()

def run2danalysis(data,objHeaders=None,saveFigsPrepend=None):
    """

    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane

    # plt.figure()
    # mp.draw2dMeanPlane()
    # # mp.plot2dResidual()
    # plt.legend()
    # if saveFigsPrepend is not None:
    #     plt.savefig(saveFigsPrepend+'_meanPlane.png',bbox_inches='tight')
    # plt.show()

    plt.figure()
    plotLogTradeRatios(mp, objHeaders)
    plt.legend()
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_tradeRatios.png',bbox_inches='tight')
    plt.show()

    fa=SlowFourierAnalyzer.fromMeanPlane(mp)
    plt.figure()
    spectral1dPowerPlot(fa)
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_spectralPower.png',bbox_inches='tight')
    plt.show()

    plt.figure()
    approximationPlot2d(mp, fa)
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_reverseTransform.png',bbox_inches='tight')
    plt.show()
    return (mp,fa)

def run3danalysis(data,objHeaders=None,saveFigsPrepend=None):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane
    plt.figure()
    mp.draw3dMeanPlane()
    # mp.plot2dResidual()
    # plt.legend()
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_meanPlane.png',bbox_inches='tight')
    plt.show()

    plt.figure()
    plotLogTradeRatios(mp, objHeaders)
    plt.legend()
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_tradeRatios.png',bbox_inches='tight')
    plt.show()

    fa=SlowFourierAnalyzer.fromMeanPlane(mp)
    plt.figure()
    spectral2dPowerImage(fa)
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_spectralPower.png',bbox_inches='tight')
    plt.show()

    plt.figure()
    spectral2dPowerPlot(fa)
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_spectralPower3d.png',bbox_inches='tight')
    plt.show()

    # plt.figure()
    # approximationPlot3d(mp, fa) # not yet implemented
    # if saveFigsPrepend is not None:
    #    plt.savefig(saveFigsPrepend+'_reverseTransform.png',bbox_inches='tight')
    # plt.show()

def runHighDimAnalysis(data, objHeaders=None, saveFigsPrepend=None):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane

    plt.figure()
    plotLogTradeRatios(mp, objHeaders)
    plt.legend()
    if saveFigsPrepend is not None:
        plt.savefig(saveFigsPrepend+'_tradeRatios.png',bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    numsmpl=30

    # demo finding the mean plane in 2d
    seedList=np.linspace(0,np.pi/2,numsmpl)
    # seedList=np.sort(np.random.rand(numsmpl)*np.pi/2)
    dummyTest2d=np.vstack((np.sin(seedList),np.cos(seedList))).T

    # run2danalysis(dummyTest2d,saveFigsPrepend='testSave')
    run2danalysis(dummyTest2d)

