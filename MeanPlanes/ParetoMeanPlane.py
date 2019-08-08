from MeanPlanes.meanPlane import *

class ParetoMeanPlane(MeanPlane):
    def __init__(self,paretoSamples):
        super(ParetoMeanPlane,self).__init__(paretoSamples)
        if np.all(self.normalVect<0):
            self._Vt*=-1 # default to pointing out--positive
            self._U*=-1
        elif np.any(self.normalVect<0): # if not all negative or positive
            raise NotPointingToOriginError(self.normalVect)
            # pass
        if np.any(self._S==0):
            raise DegeneratePlaneError
