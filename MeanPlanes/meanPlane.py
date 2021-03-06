from Common.common import *

# TODO: Constrain to be postitive. Use Quadprog from pip

class MeanPlaneError(Exception):
    pass

class MeanPlane():
    def __init__(self, paretoSamples):
        self.meanPoint=np.mean(paretoSamples,axis=0) # the mean of the samples. a point on the plane
        self.paretoSamples=paretoSamples
        self._centeredSamples=paretoSamples-self.meanPoint
        self.embedDim=paretoSamples.shape[1]
        self._U, self._S, self._Vt=np.linalg.svd(self._centeredSamples)

    @property
    def normalVect(self):
        """
        :return: the normalized normal vector to the plane
        """
        return self._Vt[-1, :]

    @property
    def basisVects(self):
        return self._Vt[:-1, :]

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

    def pointsInOriginalCoors(self,coorsInMeanPlaneSubspace):
        if len(coorsInMeanPlaneSubspace.shape) == 1:
            return np.dot(coorsInMeanPlaneSubspace[:,np.newaxis],np.squeeze(self.basisVects)[np.newaxis,:])+self.meanPoint[np.newaxis,:]
        else:
            return np.dot(coorsInMeanPlaneSubspace,self.basisVects)+self.meanPoint[np.newaxis,:]

class NotPointingToOriginError(MeanPlaneError):
    pass

class DegeneratePlaneError(MeanPlaneError):
    pass

class OptimFailError(MeanPlaneError):
    pass

class DimTooHighError(Exception):
    pass

