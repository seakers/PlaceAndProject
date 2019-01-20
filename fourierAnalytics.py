import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import pca

from common import *

class InvalidValueError(Exception):
    pass

class MeanPlane():
    def __init__(self, paretoSamples):
        self.meanPoint=np.mean(paretoSamples,axis=0) # the mean of the samples. a point on the plane
        self.paretoSamples=paretoSamples
        self._centeredSamples=paretoSamples-self.meanPoint
        self.embedDim=paretoSamples.shape[1]
        self.__U, self.__S, self._V=np.linalg.svd(self._centeredSamples)
        if np.any(self.__S==0):
            raise InvalidValueError

    @property
    def normalVect(self):
        """
        :return: the normalized normal vector to the plane
        """
        return self._V[-1,:]

    @property
    def basisVects(self):
        return self._V[:-1,:]

    @property
    def projectionToPlaneMat(self): # I do believe this is the same as basisVects actually
        # projection=np.hstack((np.vstack((np.eye(self.embedDim-1),np.zeros(self.embedDim-1))),np.zeros((self.embedDim,1))))
        # return np.dot(self._V,projection)
        return self.basisVects

    @property
    def projectionMat(self):
        return np.dot(self.projectionToPlaneMat.T,self.projectionToPlaneMat)

    @property
    def inputProjections(self):
        """
        :return: the locations of the input points in the mean plane but in the orginal coordinate system with the mean point added. Give a set of points in the plane
        """
        return np.dot(self._centeredSamples,self.projectionMat)+self.meanPoint[np.newaxis,:]

    @property
    def inputInPlane(self):
        """
        :return: the locations of the input points in the imean plane but in the original coordinate system. Basically, if think of the mean plane as defining locations, these are where the projections land in the plane when looking at the plane
        """
        return np.squeeze(np.dot(self._centeredSamples,self.projectionToPlaneMat.T))

    @property
    def inputResidual(self):
        """
        :return: the displacement of the point to the plane along the direction self.normalVect
        """
        return np.squeeze(np.dot(self._centeredSamples,self.normalVect[:,np.newaxis]))

class lowDimMeanPlane(MeanPlane):
    """
    additional methods and properties enabled by having a mean plane in 2d or 3d. really a convienence for plotting.
    """
    def draw2dMeanPlane(self):
        dummyTest2d=self.paretoSamples
        # plt.plot(self._centeredSamples[:,0],self._centeredSamples[:,1])
        plt.plot(dummyTest2d[:,0],dummyTest2d[:,1],'.',label='Pareto Surface')
        # plt.plot(self.inputProjections[:,0],self.inputProjections[:,1],label='Projections')
        planeSampleX=np.linspace(0,1,5)
        planeSampleY=(np.dot(self.normalVect,self.meanPoint)-self.normalVect[0]*planeSampleX)/self.normalVect[1]
        plt.plot(planeSampleX,planeSampleY, label='plane (from normal vector)')

    def plot2dResidual(self):
        plt.plot(self.inputResidual)

    def draw3dMeanPlane(self):
        dummyTest3d = self.paretoSamples
        ax = Axes3D(plt.gcf())
        # ax.scatter(dummyTest3d[:, 0], dummyTest3d[:, 1], dummyTest3d[:, 2], '.', label='sample points')
        ax.plot(dummyTest3d[:, 0], dummyTest3d[:, 1], dummyTest3d[:, 2], '.', label='sample points')
        # ax.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],label='Projections')
        minVal = np.min(self.inputProjections[:, 0:2], axis=0)
        maxVal = np.max(self.inputProjections[:, 0:2], axis=0)
        evalPointsX, evalPointsY = np.meshgrid((minVal[0], maxVal[0]),(minVal[1], maxVal[1]))
        print(minVal)
        print(maxVal)
        assert not np.isclose(self.normalVect[2], 0)
        evalPointsZ = (np.squeeze(np.dot(self.normalVect, self.meanPoint)) - self.normalVect[0] * evalPointsX -
                       self.normalVect[1] * evalPointsY) / self.normalVect[2]
        print(evalPointsZ)
        ax.plot_surface(evalPointsX, evalPointsY, evalPointsZ,color='r',label='mean plane')

    def plot3dResidual(self):
        """
        plots a graphic of the residual at any location on the plane
        :return:
        """
        raise NotImplementedError

class SlowFourierAnalyzer():
    """
    https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Multidimensional_DFT
    https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform#2-D_NDFT
    """
    def __init__(self,pointHeight,pointLocation):
        """
        initializes the FourierAnalyzer object
        :param pointHeight:
        :param pointLocation:
        """
        self.pointLocation=pointLocation
        self.pointHeight=pointHeight

    @property
    def __freqSumMat(self):
        numel=self.pointHeight.size
        powChngMat,freqChngMat=np.meshgrid(np.arange(0,numel),np.linspace(0,np.max(self.pointLocation))) # TODO: generalize for higher dimensions
        np.exp(powChngMat*freqChngMat*1j*self.pointLocation)

    @property
    def spectrum(self):
        """
        :return: the one-sided spectrum of the pointHeights and locations input when creating the analyzer
        """
        return np.dot(self.__freqSumMat,self.pointHeight.T)

    @property
    def fftFreqs(self):
        """returns the frequencies at which the spectrum is evaluated"""
        return np.fft.rfftfreq(len(self.pointHeight))

    def reconstruction(self):
        return np.fft.irfft(np.squeeze(self.spectrum), self.pointHeight.size)

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return FourierAnalyzer(meanPlane.inputResidual,meanPlane.inputInPlane)

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

    @property
    def spectrum(self):
        """
        :return: the one-sided spectrum of the pointHeights and locations input when creating the analyzer
        """
        return np.fft.rfft(self.pointHeight)

    @property
    def fftFreqs(self):
        """returns the frequencies at which the spectrum is evaluated"""
        return np.fft.rfftfreq(len(self.pointHeight))

    def reconstruction(self):
        return np.fft.irfft(np.squeeze(self.spectrum), self.pointHeight.size)

    @classmethod
    def fromMeanPlane(cls,meanPlane):
        """returns a FourierAnalyzer which analyzes the residuals as defined by locations in the inputProjections"""
        return FourierAnalyzer(meanPlane.inputResidual,meanPlane.inputInPlane)

class FourierSummarizer():
    def __init__(self, FourierAnalyzerObj):
        raise NotImplementedError

def spectralPowerPlot(self):
    spectralPower=np.abs(self.spectrum)**2
    print(spectralPower)
    plt.plot(self.fftFreqs,spectralPower)
    plt.xlabel('frequency')
    plt.ylabel('square power')

def spectral2dPlot(meanPlane, analyzer):
    dummyTest2d=meanPlane.paretoSamples
    mp=meanPlane
    # plt.plot(mp._centeredSamples[:,0],mp._centeredSamples[:,1])
    plt.plot(dummyTest2d[:,0],dummyTest2d[:,1],'.',label='Pareto Surface')
    plt.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],'.',label='Projections')
    # for point in zip(mp.inputProjections,dummyTest2d):
    #     plt.plot((point[0][0],point[1][0]), (point[0][1],point[1][1]))
    plt.plot(np.linspace(0,1,5),(np.dot(mp.normalVect,mp.meanPoint)-mp.normalVect[0]*np.linspace(0,1,5))/mp.normalVect[1])
    filteredCorrection=analyzer.reconstruction()
    corrected=mp.inputProjections+np.dot(filteredCorrection[:,np.newaxis],mp.normalVect[np.newaxis,:])
    plt.plot(corrected[:,0],corrected[:,1],'.',label='corrected after spectral representation')
    plt.legend()

def quick2dscatter(points):
    """a quick plot made for debugging"""
    plt.plot(points[:,0],points[:,1])
    plt.show()

if __name__=="__main__":
    numsmpl=300

    # demo finding the mean plane in 2d
    seedList=np.linspace(0,np.pi/2,numsmpl)
    seedList=np.sort(np.random.rand(numsmpl)*np.pi/2)
    dummyTest2d=np.vstack((np.sin(seedList),np.cos(seedList)+0.1*np.sin(seedList*10))).T

    mp=lowDimMeanPlane(dummyTest2d) # create the mean plane
    plt.figure()
    mp.draw2dMeanPlane()
    plt.legend()
    plt.show()

    # demo spectral analysis in 2d
    fa=lowDimFourierAnalyzer.fromMeanPlane(mp) # create the fourier analysis

    plt.figure()
    fa.spectralPowerPlot()
    plt.show()

    plt.figure()
    spectral2dPlot(mp,fa)
    plt.show()

    # demo finding the mean plane in 3d
    # dummyTest3d = concaveHypersphere(numsmpl)
    # mp = lowDimMeanPlane(dummyTest3d)
    # plt.figure()
    # mp.draw3dMeanPlane()
    # plt.show()
