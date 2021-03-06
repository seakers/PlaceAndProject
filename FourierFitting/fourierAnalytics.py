from Common.analyticsCommon import filteredSpectrum
from plottingStuffAndDecisionMakers.tradeoffMatrixImage import *
import warnings
import Common.common as cmn
from Common.analyticsCommon import *
import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage.filters as spndf
import numpy.linalg as npl

import MeanPlanes.lowDimMeanPlane as lmp

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
                # does this go up to very high dimension properly?
                frequenciesToEval=np.array(list(map(lambda r, n: 1/r * np.concatenate((np.arange(1,n//2+1),-np.arange(1,incToEven(n)/2+1)[::-1])), ranges, frequencyDivisions)))
                # frequenciesToEval=np.dot(1/(ranges[:,np.newaxis]),np.concatenate((np.arange(1,pointHeight.size//2),-np.arange(1,incToEven(pointHeight.size)/2)[::-1]))[np.newaxis,:])
                # frequenciesToEval=np.array(list(map(__oneFreqRange,ranges,it.repeat(pointHeight.size))))
                # frequenciesToEval=np.dot(1/(ranges[:,np.newaxis]),np.arange(1,incToEven(pointHeight.size)/2)[np.newaxis,:])
        self.fftFreqs=numpyze(frequenciesToEval)
        self.periodHumps=np.concatenate((np.arange(1,pointHeight.size//2),-np.arange(1,incToEven(pointHeight.size)/2)[::-1]))
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
        """

        :return:
        """
        return self.filteredSpectrum()

    def filteredSpectrum(self):
        return filteredSpectrum(self.inputFilters,self.spectralFilters,self.fftFreqs, self.pointLocation, self.pointHeight, forwardTransform)

    def trueSpectrum(self):
        return forwardTransform(self.fftFreqs, self.pointLocation, self.pointHeight)

    def reconstruction(self, locations=None):
        """
        runs fourier series defined by this analysis.
        :param locations: locations to evaluate at. if None (default) evaluates on the input locations
        :return: value of the inverse transform at corresponding locations. if was done on input, returns heights for each point in the order of the fft input when creating the object
        """
        if locations is None:
            locations=self.pointLocation
        return reconstruction(self.fftFreqs,locations,self.filteredSpectrum(),self.pointHeight.size)

    def reconstructDerivative(self,locations=None):
        if locations is None:
            locations=self.pointLocation
        return reconstructDerivative(self.fftFreqs,locations, self.filteredSpectrum(), self.pointHeight.size)

    def avgSqrdReconstructionError(self):
        return np.mean((self.reconstruction()-self.pointHeight)**2)

    def freqGrid(self):
        return freqTuples(self.fftFreqs)

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return SlowFourierAnalyzer(meanPlane.inputResidual,meanPlane.inputInPlane)

def freqTuples(freqs):
    freqProd=np.array(freqs)
    if len(freqProd.shape)>1:
        freqProd=np.array(list(map(lambda arr: arr.flatten(), np.meshgrid(*freqProd))))
    else:
        freqProd.resize((1,len(freqProd)))
    return freqProd

def forwardTransform(freqs, locations, functionVals):
    freqProd=freqTuples(freqs)
    if len(locations.shape)==1:
        pointLoc=locations[:,np.newaxis]
    else:
        pointLoc=locations
    exponentTerm=-2*np.pi*1j*np.dot(pointLoc,freqProd) # no trasnpose?
    # return np.dot(self.pointHeight,np.exp(exponentTerm)).reshape(self.numFreqs)/self.pointHeight.size
    if len(freqs.shape)>1:
        numFreqs=list(map(len, freqs))
    else:
        numFreqs=len(freqs)
    return np.dot(functionVals,np.exp(exponentTerm)).reshape(numFreqs)
# def forwardTransform(freqs, locations, functionVals):
#     return truncatedVandermondeForwardTransform(freqs,locations,functionVals)

def truncatedVandermondeForwardTransform(freqs, locations, functionVals, threshold=None):
    freqProd=freqTuples(freqs)
    if len(locations.shape)==1:
        pointLoc=locations[:,np.newaxis]
    else:
        pointLoc=locations
    # TODO: noncomplex evaluation of vandermonde
    exponentTerm=2*np.pi*1j*np.dot(pointLoc,freqProd) # no transpose?
    pseudoVander=np.exp(exponentTerm)

    U,S,Vh=np.linalg.svd(pseudoVander)
    numTake=0
    filtS=S
    if threshold is None:
        Eps= np.finfo(functionVals.dtype).eps
        Neps = np.prod(cmn.numpyze(freqs.shape)) * Eps * S[0] #truncation due to ill-conditioning
        # Nt = max(np.argmax(Vh, axis=0)) #"Automatic" determination of threshold due to Runge's phenomenon
        # threshold=min(Neps, Nt)
        threshold=Neps
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
    if len(freqs.shape)>1:
        numFreqs=list(map(len, freqs))
    else:
        numFreqs=len(freqs)
    return np.reshape(ret, numFreqs)

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
            broadcastArr=np.ones(len(spectrum.shape),dtype=int)
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

class FourierAnalyzer():
    """
    resource on learning what the multidimensional transform is and does: https://see.stanford.edu/materials/lsoftaee261/chap8.pdf
    python nonuniform FFT: https://pypi.python.org/pypi/pynufft/0.3.2.8
    Some other libraries: http://dsp.stackexchange.com/questions/16590/non-uniform-fft-with-fftw
    """
    def __init__(self,pointHeight,pointLocation,frequenciesToEval=None):
        """

        initializes the FourierAnalyzer object
        :param pointHeight:
        :param pointLocation:
        """
        if frequenciesToEval is not None:
            raise OptionNotSupportedError('cannot select frequencies and still run FFT')
        # orderingArray=orderLocations1d(pointLocation)
        # self.pointHeight=pointHeight[orderingArray]
        # self.pointLocation=pointLocation[orderingArray]
        self.pointHeight=pointHeight
        self.pointLocation=pointLocation
        self.spectralFilters=[]
        self.inputFilters=[]
        if len(pointLocation.shape)>1:
            nthRoot=np.ceil(pointHeight.size**(1/pointLocation.shape[1]))
            frequencyDivisions=(nthRoot,)*pointLocation.shape[1]
        ranges=np.ptp(self.pointLocation,axis=0)

    def addSpectralFilter(self,filter):
        self.spectralFilters.append(filter)
    def removeSpectralFilter(self, filter):
        self.spectralFilters.remove(filter)
    def addInputFilter(self,filter):
        self.inputFilters.append(filter)
    def removeInputFilter(self,filter):
        self.inputFilters.remove(filter)

    def __interpAndFFT(self, freqs,locations, functionVals):
        errMat=interpolateErr(locations, functionVals)
        return np.fft.fftn(errMat)

    def __interpAndFFT1d(self, freqs,locations, functionVals):
        interpolator=sp.interpolate.interp1d(locations, functionVals, kind='nearest')
        interpLocs=np.linspace(np.min(locations), np.max(locations), len(locations))
        interpedVals=interpolator(interpLocs)
        return np.fft.fft(interpedVals)[1:]

    @property
    def spectrum(self):
        """
        :return: the one-sided spectrum of the pointHeights and locations input when creating the analyzer
        """
        return self.filteredSpectrum()

    def trueSpectrum(self):
        if len(self.pointLocation.shape)>1:
            return self.__interpAndFFT(self.fftFreqs,self.pointLocation,self.pointHeight)
        else:
            return self.__interpAndFFT1d(self.fftFreqs,self.pointLocation,self.pointHeight)

    def filteredSpectrum(self):
        """
        :return:
        """
        if len(self.pointLocation.shape)>1:
            return filteredSpectrum(self.inputFilters,self.spectralFilters,self.fftFreqs,self.pointLocation,self.pointHeight, self.__interpAndFFT)
        else:
            return filteredSpectrum(self.inputFilters,self.spectralFilters,self.fftFreqs,self.pointLocation,self.pointHeight, self.__interpAndFFT1d)

    @property
    def fftFreqs(self):
        """returns the frequencies at which the spectrum is evaluated"""
        if len(self.pointLocation.shape)>1:
            nps=stdNumPerSide(self.pointLocation.shape)
            spacing=list(map(lambda lo, hi, n: (hi-lo)/(n-1), np.min(self.pointLocation, axis=0), np.max(self.pointLocation, axis=0), nps))
            return np.array([np.fft.fftfreq(n,d=space)[1:] for n,space in zip(nps,spacing)])
        else:
            return np.fft.fftfreq(len(self.pointHeight),d=np.ptp(self.pointLocation)/len(self.pointHeight))[1:]

    def reconstruction(self,locations=None):
        """
        runs fourier series defined by this analysis.
        :param locations: locations to evaluate at. if None (default) evaluates on the input locations
        :return: value of the inverse transform at corresponding locations. if was done on input, returns heights for each point in the order of the fft input when creating the object
        """
        if locations is None:
            locations=self.pointLocation
        # interpolator=sp.interpolate.interp1d()
        # return np.fft.ifft(self.spectrum)
        return reconstruction(self.fftFreqs,locations,self.filteredSpectrum(),self.pointHeight.size)

    def avgSqrdReconstructionError(self):
        return np.mean((self.reconstruction()-self.pointHeight)**2)

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return FourierAnalyzer(meanPlane.inputResidual,meanPlane.inputInPlane)

def run2danalysis(data,objHeaders=None,saveFigsPrepend=None,freqsToKeep=2, displayFigs=True, isMaxObj=None):
    """

    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lmp.lowDimMeanPlane(data) # create the mean plane

    runShowSaveClose(mp.draw2dMeanPlane,saveFigsPrepend+'_meanPlane.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotLogTradeRatios,mp,objHeaders),saveFigsPrepend+'_tradeRatios.png',displayFig=displayFigs)

    fa=FourierSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
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
    mp=lmp.lowDimMeanPlane(data) # create the mean plane
    runShowSaveClose(mp.draw3dMeanPlane,noneSafeConcat(saveFigsPrepend,'_meanPlane.png'),displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotLogTradeRatios,mp,objHeaders),noneSafeConcat(saveFigsPrepend,'_tradeRatios.png'),displayFig=displayFigs)

    fa=FourierSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')
    runShowSaveClose(ft.partial(spectral2dPowerImage,fa),noneSafeConcat(saveFigsPrepend,'_spectralPower.png'),displayFig=displayFigs)
    runShowSaveClose(ft.partial(spectral2dPowerPlot,fa),noneSafeConcat(saveFigsPrepend,'_spectralPower3d.png'),displayFig=displayFigs)
    runShowSaveClose(ft.partial(approximationPlot3d,mp,fa),noneSafeConcat(saveFigsPrepend,'_reverseTransform.png'),displayFig=displayFigs)
    runShowSaveClose(ft.partial(plot3dErr,mp.inputInPlane,mp.inputResidual),noneSafeConcat(saveFigsPrepend,'_errorPlot.png'),displayFig=displayFigs)
    runShowSaveClose(fa.powerDeclineReport,noneSafeConcat(saveFigsPrepend,'_powerDeclineReport.png'),displayFig=displayFigs)
    runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),noneSafeConcat(saveFigsPrepend,'_tradeoffPlot.png'),displayFig=displayFigs)

def runHighDimAnalysis(data, objHeaders=None, saveFigsPrepend=None,freqsToKeep=None,displayFigs=True):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lmp.lowDimMeanPlane(data) # create the mean plane

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

def spectralGaussBlur(freqs,spectrum,bandwidth=1):
    # kernel=np.exp(-(freqs/bandwidth)**2/2)/np.sqrt(2*np.pi)/bandwidth
    kernel=np.exp(-(freqs/bandwidth)**2/2)
    return spectrum*kernel

def gaussBlur(input, sigma=10):
    #need to verify this works in arbitrary dimension. No errros thrown atleast
    # problem: fails on unevenly spaced data
    spndf.gaussian_filter(input, sigma)

class FourierSummarizer():
    def __init__(self,numToTake,wavelenth=None):
        self.numTake=numToTake
        self.freqsTaken=[]
        self.freqSpectra=[]
        self.indcies=[]
        self.wavelength=wavelenth
        self.hasRun=False
        self.lostPower=None

    def __findFreqs(self,freqs,spectrum):
        self.hasRun=True
        spectralPower=np.abs(spectrum)**2
        sortIndx=np.argsort(spectralPower.flatten())[::-1]
        toTake=sortIndx[:min(2*self.numTake,spectralPower.size)] # assume symmetry works
        notTaken=sortIndx[min(2*self.numTake,spectralPower.size):]
        self.indcies=toTake
        if len(freqs.shape)>1:
            ft=freqTuples(freqs).T
            self.freqsTaken=ft[toTake,:]
        else:
            self.freqsTaken=freqs[toTake]
        self.freqSpectra=spectrum.flatten()[toTake]
        self.lostPower=np.sum(np.abs(spectralPower.flatten()[notTaken])**2)
        return toTake

    def __call__(self,freqs,spectrum):
        return self.filtering(freqs,spectrum)

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

class FourierSummarizerAnalyzer(SlowFourierAnalyzer):
    def __init__(self,pointHeight,pointLocation,frequenciesToEval=None,freqsToKeep=15):
        super(FourierSummarizerAnalyzer, self).__init__(pointHeight,pointLocation,frequenciesToEval)
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
        return FourierSummarizerAnalyzer(meanPlane.inputResidual,meanPlane.inputInPlane,freqsToKeep=freqsToKeep)

if __name__=="__main__":
    numsmpl=30

    # demo finding the mean plane in 2d
    seedList=np.linspace(0,np.pi/2,numsmpl)
    seedList=np.sort(np.random.rand(numsmpl)*np.pi/2)
    dummyTest2d=np.vstack((np.sin(seedList),np.cos(seedList))).T

    run2danalysis(dummyTest2d,saveFigsPrepend='testSave')
    run2danalysis(dummyTest2d)

    seedList=np.linspace(0,1,64)
    y=np.sin(2*np.pi*seedList)
    fa=SlowFourierAnalyzer(y,seedList)
    derivatives=np.array([fa.reconstructDerivative(x) for x in seedList])
    plt.figure()
    plt.plot(seedList,y,seedList,fa.reconstruction(seedList))
    plt.figure()
    plt.plot(seedList,2*np.pi*np.cos(2*np.pi*seedList),seedList,derivatives)
    plt.show()
# TODO: Report basis vector directions for interpretign direction in fourier analysis
# TODO: make sure normal vector points in one particular direction
# TODO: Possible bug, should we be normalizing the coefficients in the forward transform and not in the reconstruction?
