import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Common import analyticsCommon as aC, common as cmn


class PolynomialAnalyzer():
    """
    """
    def __init__(self,forwardTransform, reconstruct, reconstructDerivative, reconstructHessian, pointHeight,pointLocation,ordersToEval=None, normalizeMin=None, normalizeRange=None):
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
        self.reconHcall=reconstructHessian
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

    def reconstructHessian(self, locations=None):
        if locations is None:
            locations=self.normalizedPointLocation
        else:
            locations=self.normalize(locations)
        return self.reconHcall(self.ordersToEval, locations, self.filteredSpectrum(), self.pointHeight.size)

    def avgSqrdReconstructionError(self):
        return np.mean((self.reconstruction()-self.pointHeight)**2)

    def freqGrid(self):
        return orderTuples(self.fullOrders)

    @classmethod
    def fromMeanPlane(cls,meanPlane, ordersToEval=None, normalizeMin=None, normalizeRange=None):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return cls(meanPlane.inputResidual, meanPlane.inputInPlane, ordersToEval=ordersToEval, normalizeMin=normalizeMin, normalizeRange=normalizeRange) #dangerous. Assumes input arguments don't chagne in children classes


def orderTuples(orders):
    freqProd=np.array(orders)
    if len(freqProd.shape)>1:
        freqProd=np.array(list(map(lambda arr: arr.flatten(), np.meshgrid(*freqProd))))
    else:
        freqProd.resize((1,len(freqProd)))
    return freqProd


class OptionNotSupportedError(Exception):
    pass


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
        plt.xlabel('polynomial term and orders')

    def report(self, tofile=None):
        if tofile is None:
            print(self._toDataFrame())
            print('captured power: '+str(np.sum(np.abs(self.freqSpectra)**2)))
            print('lost power: '+str(self.lostPower))
        else:
            self._toDataFrame().to_csv(tofile)
            with open(tofile,'a') as f:
                f.writelines(('captured power: '+str(np.sum(np.abs(self.freqSpectra)**2)), 'lost power: '+str(self.lostPower)))