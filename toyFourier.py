import numpy as np
import scipy as sp
import pandas as pd
from pandas.plotting import parallel_coordinates as paraCoor
import itertools as it
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import pca
import functools as ft

from meanPlane import *
from common import *
import fourierAnalytics as fA
import RBFN.rbfAnalytics as rA
from polyfitting import legender as lA

def unbalanced(mC=fA):
    x=np.concatenate((np.linspace(0,0.5,32),0.5*np.ones(32)))
    y=np.concatenate((np.ones(32),    np.linspace(1,0,32)))
    data=np.vstack((x,y)).T
    mp,fa=mC.run2danalysis(data,saveFigsPrepend='unevenBoxDemo',freqsToKeep=100)
    plt.figure()
    plt.plot(mp.inputInPlane,mp.inputResidual)
    plt.xlabel('projected inputs in the plane')
    plt.ylabel('residual to the input location from the plane')
    plt.savefig('singleBoxyDemo_residualPlot.png')
    plt.show()
    plt.figure()
    mp.draw2dMeanPlane()
    plt.axis('equal')
    plt.savefig('singleBoxDemo_meanPlane.png')
    plt.show()

def fourierTesting():
    # fa = SlowFourierAnalyzer(np.sin(np.linspace(0,2*np.pi,10)),np.linspace(0,10,10)) # what the FFT sees:
    # x=np.linspace(0,2*np.pi,10)
    # x=np.arange(0,10)/10
    x=np.linspace(0,1,100)
    # x=np.linspace(0,1,9)
    # x=np.sort(np.random.rand(10))
    y=np.sin(2*np.pi*x)
    nyqFreq=len(x)//2
    # fa = SlowFourierAnalyzer(y,x,frequenciesToEval=np.concatenate((np.arange(nyqFreq)/nyqFreq,-np.arange(nyqFreq)/nyqFreq))) # what needs to agree.
    faref = fA.FourierAnalyzer(y,x)
    fa=fA.SlowFourierAnalyzer(y,x)
    # fa=SlowFourierAnalyzer.fromMeanPlane(mp)

    print(fa.fftFreqs)
    print(faref.fftFreqs)
    print(fa.spectrum)
    print(faref.spectrum)

    plt.figure()
    plt.hold('on')
    fA.spectral1dPowerPlot(fa)
    fA.spectral1dPowerPlot(faref)
    plt.show()

    plt.figure()
    plt.hold('on')
    fA.spectral1dPhasePlot(fa)
    fA.spectral1dPhasePlot(faref)
    plt.show()

    plt.figure()
    plt.hold('on')
    smplX=np.sort(np.concatenate((np.linspace(0,1,64),x)))
    smplY=np.sin(2*np.pi*smplX)
    plt.plot(smplX,smplY,label='true')
    plt.plot(smplX,fa.reconstruction(smplX),label='slow')
    plt.plot(smplX,faref.reconstruction(smplX),label='fast')
    plt.plot(x,y,'.',label='true')
    plt.plot(fa.pointLocation,fa.reconstruction(),'.',label='slow')
    plt.plot(x,faref.reconstruction(),'.',label='fast')
    plt.legend()
    plt.show()

def circlePfrontDemo(mC=fA):
    # seedList=np.linspace(0,np.pi/2,128)
    seedList=np.sort(np.random.rand(64)*np.pi/2)
    np.savetxt('quarterCircleSeed.csv',seedList)
    x=np.cos(seedList)
    y=np.sin(seedList)
    data=np.vstack((x,y)).T
    mp,fa=mC.run2danalysis(data,saveFigsPrepend='circleDemo')

    plt.figure()
    plt.plot(mp.inputInPlane,mp.inputResidual,'.')
    plt.xlabel('projected inputs in the plane')
    plt.ylabel('residual to the input location from the plane')
    plt.savefig('circleDemo_residualPlot.png')
    plt.show()

    plt.figure()
    mp.draw2dMeanPlane()
    plt.axis('equal')
    plt.savefig('circleDemo_meanPlane.png')
    plt.show()

def singleEvenBoxDemo(mC=fA):
    x=np.concatenate((np.linspace(0,0.5,32),0.5*np.ones(32)))*2
    y=np.concatenate((np.ones(32),    np.linspace(1,0,32)))
    data=np.vstack((x,y)).T
    mp,fa=mC.run2danalysis(data,saveFigsPrepend='singleBoxDemo')
    plt.figure()
    plt.plot(mp.inputInPlane,mp.inputResidual)
    plt.xlabel('projected inputs in the plane')
    plt.ylabel('residual to the input location from the plane')
    plt.savefig('singleBoxyDemo_residualPlot.png')
    plt.show()
    plt.figure()
    mp.draw2dMeanPlane()
    plt.axis('equal')
    plt.savefig('singleBoxDemo_meanPlane.png')
    plt.show()

def wavyPfrontDemo(mC, label=''):
    x=np.concatenate((np.linspace(0,0.25,32),0.25*np.ones(32),np.linspace(0.25,0.5,32),0.5*np.ones(32),np.linspace(0.5,0.75,32),0.75*np.ones(32),np.linspace(0.75,1,32),np.ones(32)))
    y=np.concatenate((np.ones(32),np.linspace(1,0.75,32),0.75*np.ones(32),np.linspace(0.75,0.5,32), 0.5*np.ones(32), np.linspace(0.5,0.25,32),0.25*np.ones(32),np.linspace(0.25,0,32)))
    data=np.vstack((x,y)).T
    mp,fa=mC.run2danalysis(data,saveFigsPrepend='./output/boxyDemo'+label,freqsToKeep=1000)
    plt.figure()
    plt.plot(mp.inputInPlane,mp.inputResidual)
    plt.xlabel('projected inputs in the plane')
    plt.ylabel('residual to the input location from the plane')
    plt.savefig(label+'boxyDemo_residualPlot.png')
    plt.show()
    plt.figure()
    mp.draw2dMeanPlane()
    plt.axis('equal')
    plt.savefig(label+'boxyDemo_meanPlane.png')
    plt.show()

def dim3hypersphereTesting(mC=fA, label=''):
    # demo finding the mean plane in 3d
    numsmpl=30**2
    dummyTest3d = concaveHypersphere(numsmpl)
    np.savetxt('concaveHyperspherePoints.csv',dummyTest3d)
    mC.run3danalysis(dummyTest3d,saveFigsPrepend='./output/3dhypersphere'+label)
    # xx,yy=map(lambda ar: ar.flatten(), np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128)))
    # zz=np.sin(2*np.pi*xx)*np.cos(2*np.pi*yy)

    # fa=SlowFourierAnalyzer(zz,np.vstack((xx,yy)).T)
    # fa=SlowFourierAnalyzer.fromMeanPlane(mp)

def positiveCovarDemo():
    numsmpl=900
    dummyTest3d=pathologicalRing(numsmpl)
    # draw3dSurface(dummyTest3d)
    # plt.gca().set_aspect('equal')
    # plt.savefig('positiveCovarPoints.png')

    # plt.plot(dummyTest3d[:,0],dummyTest3d[:,1],'.')
    # plt.gca().set_aspect('equal')
    # plt.savefig('positiveCovarPointsIn2d.png')

    pointsAsDF=pd.DataFrame(dummyTest3d,columns=('x','y','z'))
    plottingDF=pointsAsDF
    plottingDF['inBand']=np.logical_and(0.4<dummyTest3d[:,0],dummyTest3d[:,0]<0.6)
    # paraCoor(pointsAsDF,'inBand')
    # plt.show()

    covar=np.cov(dummyTest3d.T)
    sprnco=sp.stats.spearmanr(dummyTest3d)
    print('covariance')
    print(covar)
    print('spearman''s rank coefficient')
    print(sprnco)

#     run3danalysis(dummyTest3d)
    import meanPlane as mp
    testmp=mp.lowDimMeanPlane(dummyTest3d)
    print('normal vector')
    print(testmp.normalVect)
    print('basis vectors')
    print(testmp.basisVects)
    # testmp.draw3dMeanPlane()
    ax = Axes3D(plt.gcf())
    # ax.scatter(dummyTest3d[:, 0], dummyTest3d[:, 1], dummyTest3d[:, 2], '.', label='sample points')
    ax.plot(dummyTest3d[:, 0], dummyTest3d[:, 1], dummyTest3d[:, 2], 'k.', label='sample points')
    # ax.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],label='Projections')
    plt.show()

if __name__=='__main__':
    # fourierTesting()
    # wavyPfrontDemo(fA, 'fourier');  wavyPfrontDemo(lA, 'legpoly');
    wavyPfrontDemo(rA, 'rbfn')
    # circlePfrontDemo()
    # dim3hypersphereTesting(fA, 'fourier'); dim3hypersphereTesting(lA, 'legpoly');
    # dim3hypersphereTesting(rA,'rbfn')
    # unbalanced()
    # positiveCovarDemo()
