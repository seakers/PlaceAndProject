import scipy.optimize as spo
from MeanPlanes.meanPlane import *

class PositiveOrthantMP():
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
