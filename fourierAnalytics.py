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
        :return: the vectors as represented in the basis of self.basisVects
        """
        return np.dot(self._centeredSamples,self.projectionMat)+self.meanPoint[np.newaxis,:]

    @property
    def inputResidual(self):
        """
        :return: the displacement of the point to the plane along the direction self.normalVect
        """
        return np.squeeze(np.dot(self._centeredSamples,self.normalVect[:,np.newaxis]))

# class FourierErrAnalytic():
#     def __init__(self,values):
#         self.values=values
#         fourierObj=np.rfft
def draw2dMeanPlane(meanPlane):
    dummyTest=meanPlane.paretoSamples
    mp=meanPlane
    # plt.plot(mp._centeredSamples[:,0],mp._centeredSamples[:,1])
    plt.plot(dummyTest[:,0],dummyTest[:,1],label='Pareto Surface')
    plt.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],label='Projections')
    # for point in zip(mp.inputProjections,dummyTest):
    #     plt.plot((point[0][0],point[1][0]), (point[0][1],point[1][1]))
    plt.plot(np.linspace(0,1,5),(np.dot(mp.normalVect,mp.meanPoint)-mp.normalVect[0]*np.linspace(0,1,5))/mp.normalVect[1])
    plt.legend()
    plt.axis('equal')
    plt.show()

def spectralPowerPlot(spectrum, fftFreqs):
    spectralPower=spectrum.real**2+spectrum.imag**2
    print(spectralPower)
    plt.figure()
    plt.plot(spectralPower)
    plt.xlabel('frequency')
    plt.ylabel('square power')
    plt.show()

def spectral2dPlot(meanPlane, spectrum, fftFreqs):
    dummyTest=meanPlane.paretoSamples
    mp=meanPlane
    # plt.plot(mp._centeredSamples[:,0],mp._centeredSamples[:,1])
    plt.plot(dummyTest[:,0],dummyTest[:,1],label='Pareto Surface')
    plt.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],label='Projections')
    # for point in zip(mp.inputProjections,dummyTest):
    #     plt.plot((point[0][0],point[1][0]), (point[0][1],point[1][1]))
    plt.plot(np.linspace(0,1,5),(np.dot(mp.normalVect,mp.meanPoint)-mp.normalVect[0]*np.linspace(0,1,5))/mp.normalVect[1])
    filteredCorrection=np.fft.irfft(np.squeeze(spectrum), mp.inputProjections.shape[0])
    corrected=mp.inputProjections+np.dot(filteredCorrection[:,np.newaxis],mp.normalVect[np.newaxis,:])
    plt.plot(corrected[:,0],corrected[:,1],label='corrected after spectral representation')
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__=="__main__":
    # demo finding the mean plane in 2d
    numsmpl=300
    dummyTest=np.vstack((np.sin(np.linspace(0,np.pi/2,numsmpl)),np.cos(np.linspace(0,np.pi/2,numsmpl))+0.5)).T
    mp=MeanPlane(dummyTest)
    draw2dMeanPlane(mp)

    # demo spectral analysis in 2d
    spectrum=np.fft.rfft(mp.inputResidual)
    fftFreqs=np.fft.rfftfreq(len(mp.inputResidual))
    spectralPowerPlot(spectrum,fftFreqs)

    spectral2dPlot(mp, spectrum, fftFreqs)


