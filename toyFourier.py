import numpy as np
import scipy as sp
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import pca
import functools as ft

from meanPlane import *
from common import *
from fourierAnalytics import *

def unbalanced():
    x=np.concatenate((np.linspace(0,0.5,32),0.5*np.ones(32)))
    y=np.concatenate((np.ones(32),    np.linspace(1,0,32)))
    data=np.vstack((x,y)).T
    mp,fa=run2danalysis(data,saveFigsPrepend='unevenBoxDemo',freqsToKeep=100)
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
    faref = FourierAnalyzer(y,x)
    fa=SlowFourierAnalyzer(y,x)
    # fa=SlowFourierAnalyzer.fromMeanPlane(mp)

    print(fa.fftFreqs)
    print(faref.fftFreqs)
    print(fa.spectrum)
    print(faref.spectrum)

    plt.figure()
    plt.hold('on')
    spectral1dPowerPlot(fa)
    spectral1dPowerPlot(faref)
    plt.show()

    plt.figure()
    plt.hold('on')
    spectral1dPhasePlot(fa)
    spectral1dPhasePlot(faref)
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

def circlePfrontDemo():
    # seedList=np.linspace(0,np.pi/2,128)
    seedList=np.sort(np.random.rand(64)*np.pi/2)
    np.savetxt('quarterCircleSeed.csv',seedList)
    x=np.cos(seedList)
    y=np.sin(seedList)
    data=np.vstack((x,y)).T
    mp,fa=run2danalysis(data,saveFigsPrepend='circleDemo')

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

def singleEvenBoxDemo():
    x=np.concatenate((np.linspace(0,0.5,32),0.5*np.ones(32)))*2
    y=np.concatenate((np.ones(32),    np.linspace(1,0,32)))
    data=np.vstack((x,y)).T
    mp,fa=run2danalysis(data,saveFigsPrepend='singleBoxDemo')
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

def wavyPfrontDemo():
    x=np.concatenate((np.linspace(0,0.25,32),0.25*np.ones(32),np.linspace(0.25,0.5,32),0.5*np.ones(32),np.linspace(0.5,0.75,32),0.75*np.ones(32),np.linspace(0.75,1,32),np.ones(32)))
    y=np.concatenate((np.ones(32),np.linspace(1,0.75,32),0.75*np.ones(32),np.linspace(0.75,0.5,32), 0.5*np.ones(32), np.linspace(0.5,0.25,32),0.25*np.ones(32),np.linspace(0.25,0,32)))
    data=np.vstack((x,y)).T
    mp,fa=run2danalysis(data,saveFigsPrepend='boxyDemo',freqsToKeep=1000)
    plt.figure()
    plt.plot(mp.inputInPlane,mp.inputResidual)
    plt.xlabel('projected inputs in the plane')
    plt.ylabel('residual to the input location from the plane')
    plt.savefig('boxyDemo_residualPlot.png')
    plt.show()
    plt.figure()
    mp.draw2dMeanPlane()
    plt.axis('equal')
    plt.savefig('boxyDemo_meanPlane.png')
    plt.show()

def dim3hypersphereTesting():
    # demo finding the mean plane in 3d
    numsmpl=30**2
    dummyTest3d = concaveHypersphere(numsmpl)
    np.savetxt('concaveHyperspherePoints.csv',dummyTest3d)
    run3danalysis(dummyTest3d,saveFigsPrepend='3dhypersphere')
    # xx,yy=map(lambda ar: ar.flatten(), np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128)))
    # zz=np.sin(2*np.pi*xx)*np.cos(2*np.pi*yy)

    # fa=SlowFourierAnalyzer(zz,np.vstack((xx,yy)).T)
    # fa=SlowFourierAnalyzer.fromMeanPlane(mp)

if __name__=='__main__':
    # fourierTesting()
    # wavyPfrontDemo()
    # circlePfrontDemo()
    dim3hypersphereTesting()
    # unbalanced()
