import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import pca
import functools as ft
import scipy.optimize as spo

from common import *

# TODO: Constrain to be postitive. Use Quadprog from pip

class MeanPlaneError(Exception):
    pass

class MeanPlane():
    def __init__(self, paretoSamples):
        self.meanPoint=np.mean(paretoSamples,axis=0) # the mean of the samples. a point on the plane
        self.paretoSamples=paretoSamples
        self._centeredSamples=paretoSamples-self.meanPoint
        self.embedDim=paretoSamples.shape[1]
        self._U, self._S, self._V=np.linalg.svd(self._centeredSamples)

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
        #WARNING: I think this is faulty
        """
        return np.dot(self._centeredSamples,self.projectionMat)+self.meanPoint[np.newaxis,:]

    @property
    def inputInPlane(self):
        """
        :return: the locations of the input points in the imean plane but in the original coordinate system. Basically, if think of the mean plane as defining locations, these are where the projections land in the plane when looking at the plane
        """
        return np.squeeze(np.dot(self._centeredSamples,self.projectionToPlaneMat.T))

    def projectToPlaneCoor(self,locations):
        return np.squeeze(np.dot(locations-self.meanPoint[np.newaxis,:],self.projectionToPlaneMat.T))

    @property
    def inputResidual(self):
        """
        :return: the displacement of the point to the plane along the direction self.normalVect
        """
        return np.squeeze(np.dot(self._centeredSamples,self.normalVect[:,np.newaxis]))

    @property
    def tradeRatios(self):
        """
        :return: returns
        """
        return np.tile(self.normalVect[:,np.newaxis],(1,len(self.normalVect)))/np.tile(self.normalVect[np.newaxis,:],(len(self.normalVect),1))

class NotPointingToOriginError(MeanPlaneError):
    pass

class DegeneratePlaneError(MeanPlaneError):
    pass

class OptimFailError(MeanPlaneError):
    pass

class PositiveOrthantMeanPlane():
    def __init__(self, paretoSamples):
        self.meanPoint=np.mean(paretoSamples,axis=0) # the mean of the samples. a point on the plane
        self.paretoSamples=paretoSamples
        self._centeredSamples=paretoSamples-self.meanPoint
        self.embedDim=paretoSamples.shape[1]

        obj=lambda v: np.linalg.norm(np.dot(self._centeredSamples,v))**2
        divObj=lambda v: np.linalg.norm(np.dot(self._centeredSamples,v))**2/np.linalg.norm(v)**2
        cons={'type': 'ineq', 'fun': lambda v: np.linalg.norm(v)**2-1, 'jac': lambda v: v}
        consCOBLYA=[{'type': 'ineq','fun': lambda v: v[i], 'jac':  elemBasis(i,self.embedDim)} for i in range(self.embedDim)]+[cons,]
        positive=tuple((0,float('inf')) for cnt in range(self.embedDim))
        # self.normVectRes=spo.minimize(obj, np.ones(self.embedDim),constraints=consCOBLYA, method='COBYLA')
        # self.normVectRes=spo.minimize(obj, np.ones(self.embedDim),constraints=cons, method='SLSQP', bounds=positive)
        self.normVectRes=spo.minimize(divObj, np.ones(self.embedDim), method='SLSQP', bounds=positive)
        if not self.normVectRes.success:
            raise OptimFailError()

    @property
    def normalVect(self):
        """
        :return: the normalized normal vector to the plane
        """
        return self.normVectRes.x/np.linalg.norm(self.normVectRes.x)

    @property
    def basisVects(self):
        assert not np.isclose(self.normalVect[-1],0)
        proj=lambda u,v: np.dot(u,v)/np.dot(u,u) * u
        ret=np.vstack((self.normalVect[np.newaxis,:],np.eye(self.embedDim-1,self.embedDim)))
        for dimOut in range(self.embedDim):
            temp=ret[dimOut,:]
            for j in range(dimOut):
                temp=temp-np.dot(temp,ret[j,:])
            ret[dimOut,:]=temp/np.linalg.norm(temp)
        return ret[1:,:] #TODO, simply used Gram-schmidt on the original basis. There has to be a better option.

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
        #WARNING: I think this is faulty
        """
        return np.dot(self._centeredSamples,self.projectionMat)+self.meanPoint[np.newaxis,:]

    @property
    def inputInPlane(self):
        """
        :return: the locations of the input points in the imean plane but in the original coordinate system. Basically, if think of the mean plane as defining locations, these are where the projections land in the plane when looking at the plane
        """
        return np.squeeze(np.dot(self._centeredSamples,self.projectionToPlaneMat.T))

    def projectToPlaneCoor(self,locations):
        return np.squeeze(np.dot(locations-self.meanPoint[np.newaxis,:],self.projectionToPlaneMat.T))

    @property
    def inputResidual(self):
        """
        :return: the displacement of the point to the plane along the direction self.normalVect
        """
        return np.squeeze(np.dot(self._centeredSamples,self.normalVect[:,np.newaxis]))

    @property
    def tradeRatios(self):
        """
        :return: returns
        """
        return np.tile(self.normalVect[:,np.newaxis],(1,len(self.normalVect)))/np.tile(self.normalVect[np.newaxis,:],(len(self.normalVect),1))

class ParetoMeanPlane(MeanPlane):
    def __init__(self,paretoSamples):
        super(ParetoMeanPlane,self).__init__(paretoSamples)
        if np.all(self.normalVect<0):
            self._V*=-1 # default to pointing out--positive
            self._U*=-1
        elif np.any(self.normalVect<0): # if not all negative or positive
            raise NotPointingToOriginError(self.normalVect)
            # pass
        if np.any(self._S==0):
            raise DegeneratePlaneError

class DimTooHighError(Exception):
    pass

class lowDimMeanPlane(MeanPlane):
# class lowDimMeanPlane(PositiveOrthantMeanPlane):
    """
    additional methods and properties enabled by having a mean plane in 2d or 3d. really a convienence for plotting.
    """
    def draw(self):
        lookup=(self.draw2dMeanPlane, self.draw3dMeanPlane)
        sysDim=self.paretoSamples.shape[1]
        if sysDim>3:
            raise DimTooHighError
        else:
            return lookup[sysDim-2]()

    def draw2dMeanPlane(self):
        dummyTest2d=self.paretoSamples
        # plt.plot(self._centeredSamples[:,0],self._centeredSamples[:,1])
        plt.plot(dummyTest2d[:,0],dummyTest2d[:,1],'.',label='Pareto Surface')
        # plt.plot(self.inputProjections[:,0],self.inputProjections[:,1],label='Projections')
        planeSampleX=np.linspace(0,1,5)
        planeSampleY=(np.dot(self.normalVect,self.meanPoint)-self.normalVect[0]*planeSampleX)/self.normalVect[1]
        plt.plot(planeSampleX,planeSampleY, label='plane (from normal vector)')
        plt.legend()

    def plot2dResidual(self):
        plt.plot(self.inputInPlane,self.inputResidual,'.-')

    def draw3dMeanPlane(self):
        dummyTest3d = self.paretoSamples
        ax = Axes3D(plt.gcf())
        # ax.scatter(dummyTest3d[:, 0], dummyTest3d[:, 1], dummyTest3d[:, 2], '.', label='sample points')
        ax.plot(dummyTest3d[:, 0], dummyTest3d[:, 1], dummyTest3d[:, 2], 'k.', label='sample points')
        # ax.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],label='Projections')
        minVal = np.min(self.inputProjections[:, 0:2], axis=0)
        maxVal = np.max(self.inputProjections[:, 0:2], axis=0)
        evalPointsX, evalPointsY = np.meshgrid((minVal[0], maxVal[0]),(minVal[1], maxVal[1]))
        # print(minVal)
        # print(maxVal)
        assert not np.isclose(self.normalVect[2], 0)
        evalPointsZ = (np.squeeze(np.dot(self.normalVect, self.meanPoint)) - self.normalVect[0] * evalPointsX -
                       self.normalVect[1] * evalPointsY) / self.normalVect[2]
        # print(evalPointsZ)
        ax.plot_surface(evalPointsX, evalPointsY, evalPointsZ,color=globalPlaneColor,label='mean plane')

    def plot3dResidual(self):
        """
        plots a graphic of the residual at any location on the plane
        :return:
        """
        raise NotImplementedError

