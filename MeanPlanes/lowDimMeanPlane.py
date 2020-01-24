import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Common.common import globalPlaneColor
from MeanPlanes.meanPlane import MeanPlane, DimTooHighError
from MeanPlanes.MeanVectMP import MeanVectMP


class lowDimMeanPlane(MeanPlane):
# class lowDimMeanPlane(MeanVectMP):
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