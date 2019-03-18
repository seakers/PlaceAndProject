from tradeoffMatrixImage import *
from analyticsCommon import *

from RBFN.RBFN import RBFN
from RBFN.recalcRBFN import RBFNwithParamTune
from RBFN.kmeansRBFN import kmeansRBFN

class rbfAnalyzer():
    def __init__(self,pointHeight,pointLocation,numHiddenNodes=None):
        """
        initializes the FourierAnalyzer object
        :param pointHeight:
        :param pointLocation:
        """
        self.pointLocation=pointLocation
        self.pointHeight=pointHeight
        if numHiddenNodes is None:
            numHiddenNodes=pointHeight.size
        self.ordersToEval=numHiddenNodes
        self.spectralFilters=[]
        self.inputFilters=[]
        if len(pointLocation.shape)>1:
            dists=sp.spatial.distance.cdist(pointLocation, pointLocation)
            # shortestDists=np.min(dists, axis=0) # add infinity to diag to get this to work.
            shortestDists=np.mean(dists)/2
        else:
            shortestDists=np.abs(np.diff(np.sort(pointLocation)))
        # self.rbfn=RBFN(numHiddenNodes, sigma=np.mean(shortestDists))
        # self.rbfn=RBFN(numHiddenNodes/5, sigma=np.mean(shortestDists))
        # self.rbfn.fit(pointLocation, pointHeight)
        # self.rbfn=kmeansRBFN(int(numHiddenNodes/5), sigma=np.mean(shortestDists))
        # self.rbfn.fit(pointLocation, pointHeight)
        self.rbfn=RBFNwithParamTune(constantTerm=False)
        self.rbfn.fit(pointLocation, pointHeight, kfolds=5)
        self.centers=self.rbfn.centers
        # self.sigmas=self.rbfn.sigmas

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

    def __forward(self, u, X,Y):
        return self.rbfn.weights

    def filteredSpectrum(self):
        return filteredSpectrum(self.inputFilters,self.spectralFilters,self.centers, self.pointLocation, self.pointHeight, self.__forward)

    def trueSpectrum(self):
        return self.rbfn.weights

    def reconstruction(self, locations=None):
        """
        runs fourier series defined by this analysis.
        :param locations: locations to evaluate at. if None (default) evaluates on the input locations
        :return: value of the inverse transform at corresponding locations. if was done on input, returns heights for each point in the order of the fft input when creating the object
        """
        if locations is None:
            locations=self.pointLocation
        return self.rbfn.predict(locations)

    def reconstructDerivative(self,locations=None):
        if locations is None:
            locations=self.pointLocation
        return self.rbfn.derivative(locations)

    def avgSqrdReconstructionError(self):
        return np.mean((self.reconstruction()-self.pointHeight)**2)

    def freqGrid(self):
        raise NotImplementedError

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return rbfAnalyzer(meanPlane.inputResidual, meanPlane.inputInPlane)

def orderTuples(orders):
    freqProd=np.array(orders)
    if len(freqProd.shape)>1:
        freqProd=np.array(list(map(lambda arr: arr.flatten(), np.meshgrid(*freqProd))))
    else:
        freqProd.resize((1,len(freqProd)))
    return freqProd

class rbfSummarizer():
    def __init__(self,numToTake):
        self.numTake=numToTake
        self.freqSpectra=[]
        self.droppedSpectra=[]
        self.indcies=[]
        self.hasRun=False
        self.lostPower=None
        self.centersTaken=[]
        self.sigmasTaken=[]

    def __findFreqs(self, centers, spectrum):
        self.hasRun=True
        spectralPower=np.abs(spectrum)**2
        sortIndx=np.argsort(spectralPower.flatten())[::-1]
        toTake=sortIndx[:min(self.numTake,spectralPower.size)]
        notTaken=sortIndx[min(self.numTake,spectralPower.size):]
        self.indcies=toTake
        if len(centers.shape)==1:
            self.centersTaken=centers[toTake]
        else:
            self.centersTaken=centers[toTake,:]
        self.freqSpectra=spectrum.flatten()[toTake]
        self.droppedSpectra=spectrum.flatten()[notTaken]
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
        if len(self.centersTaken.shape) == 1:
            toUse=self.centersTaken > 0
            ft=self.centersTaken[toUse]
        else:
            toUse=np.logical_not(np.all(self.centersTaken == 0, axis=1))
            ft=self.centersTaken[toUse,:]
        fs=self.freqSpectra[toUse]
        d=dict()
        for i, ftarr in enumerate(ft.T):
            d['center, coordinate: '+str(i)]=ftarr
        d['spectral power']=np.abs(fs)**2
        return pd.DataFrame(d)

    def powerDeclineReport(self):
        plt.bar(np.arange(len(self.freqSpectra)),np.abs(self.freqSpectra)**2,color=globalBarPlotColor)
        if len(self.centersTaken.shape) == 1:
            xtickLbl=list(( 'center: '+c for c,o in zip(numpyToPrettyStr(self.centersTaken), numpyToPrettyStr(self.sigmasTaken))))
        else:
            xtickLbl=list(( 'center: '+c for c,o in zip(multiDimNumpyToPrettyStr(self.centersTaken), numpyToPrettyStr(self.sigmasTaken))))
        plt.xticks(range(len(self.centersTaken)),xtickLbl, rotation=75)
        plt.ylabel('squared power of component')
        plt.xlabel('representative frequency')

    def truncatedPowerDeclineReport(self):
        plt.bar(np.arange(len(self.freqSpectra)),np.abs(self.freqSpectra)**2,color=globalBarPlotColor)
        xtickLbl=list(( 'center: '+c for c,o in zip(multiDimNumpyToPrettyStr(self.centersTaken), numpyToPrettyStr(self.sigmasTaken))))
        plt.xticks(range(len(self.centersTaken)),xtickLbl, rotation=75)
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

class rbfSummarizerAnalyzer(rbfAnalyzer):
    def __init__(self,pointHeight,pointLocation,frequenciesToEval=None,freqsToKeep=5):
        super(rbfSummarizerAnalyzer, self).__init__(pointHeight, pointLocation, frequenciesToEval)
        self.summarizer=rbfSummarizer(freqsToKeep)
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
        return rbfSummarizerAnalyzer(meanPlane.inputResidual, meanPlane.inputInPlane, freqsToKeep=freqsToKeep)

def run2danalysis(data,objHeaders=None,saveFigsPrepend=None,freqsToKeep=2, displayFigs=True, isMaxObj=None):
    """

    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane

    if saveFigsPrepend is not None:
        mps=saveFigsPrepend+'_meanPlane.png'
        trs=saveFigsPrepend+'_tradeRatios.png'
        spts=saveFigsPrepend+'_spectralPower_legendre.png'
        rts=saveFigsPrepend+'_reverseTransform.png'
        pdr=saveFigsPrepend+'_powerDeclineReport.png'
    else:
        mps=None
        trs=None
        spts=None
        rts=None
        pdr=None

    runShowSaveClose(mp.draw2dMeanPlane,mps,displayFig=displayFigs)

    fa=rbfSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')

    # maybe use a scatterplot?
    # runShowSaveClose(ft.partial(spectral1dPowerPlot_nonFFT,fa),spts,displayFig=displayFigs)
    runShowSaveClose(fa.powerDeclineReport,pdr,displayFig=displayFigs)

    runShowSaveClose(ft.partial(approximationPlot2d,mp,fa,objHeaders),saveFigsPrepend+'_reverseTransform.png',displayFig=displayFigs)
    # runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)
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

    fa=rbfSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')
    # runShowSaveClose(ft.partial(spectral2dPowerImage,fa),saveFigsPrepend+'_spectralPower.png',displayFig=displayFigs)
    # runShowSaveClose(ft.partial(spectral2dPowerPlot,fa),saveFigsPrepend+'_spectralPower3d.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(approximationPlot3d,mp,fa),saveFigsPrepend+'_reverseTransform.png',displayFig=displayFigs)
    runShowSaveClose(ft.partial(plot3dErr,mp.inputInPlane,mp.inputResidual),saveFigsPrepend+'_errorPlot.png',displayFig=displayFigs)
    runShowSaveClose(fa.powerDeclineReport,saveFigsPrepend+'_powerDeclineReport.png',displayFig=displayFigs)
    # runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

def runHighDimAnalysis(data, objHeaders=None, saveFigsPrepend=None,freqsToKeep=None,displayFigs=True):
    """
    standard set of plots generated for 2-objective problems
    :param data: designs to plot. each row is a design and each column is an objective
    :param saveFigsPrepend: a prepend name for saving figures generated. None (default) prevents automatic saving.
    """
    if objHeaders is None:
        objHeaders=list(map(lambda n: 'obj: '+str(n),range(data.shape[1])))
    mp=lowDimMeanPlane(data) # create the mean plane

    # runShowSaveClose(ft.partial(plotLogTradeRatios,mp,objHeaders),saveFigsPrepend+'_tradeRatios.png',displayFig=displayFigs)

    if freqsToKeep is None:
        freqsToKeep=2**data.shape[1]
    fa=rbfSummarizerAnalyzer.fromMeanPlane(mp,freqsToKeep)
    if displayFigs:
        fa.report()
    if saveFigsPrepend is not None:
        fa.report(saveFigsPrepend+'_report.csv')

    runShowSaveClose(fa.powerDeclineReport,saveFigsPrepend+'_powerDeclinePlot.png',displayFig=displayFigs)
    # runShowSaveClose(ft.partial(plotTradeRatios,mp,fa,objHeaders),saveFigsPrepend+'_tradeoffPlot.png',displayFig=displayFigs)

if __name__=="__main__":
    numsmpl=30

    # demo finding the mean plane in 2d
    # seedList=np.linspace(0,np.pi/2,numsmpl)
    # seedList=np.sort(np.random.rand(numsmpl)*np.pi/2)
    # dummyTest2d=np.vstack((np.sin(seedList),np.cos(seedList))).T

    # run2danalysis(dummyTest2d,saveFigsPrepend='testSave')
    # run2danalysis(dummyTest2d)

    seedList=np.linspace(0,1,3)
    y=np.sin(2*np.pi*seedList)
    fa=rbfAnalyzer(y, seedList)
    plt.plot(seedList,y,seedList,fa.reconstruction(seedList))
    plt.show()
    derivatives=np.array([fa.reconstructDerivative(x) for x in seedList])
    plt.show()
