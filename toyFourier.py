import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import pca
import functools as ft

from meanPlane import *
from common import *
from fourierAnalytics import *

def fourierTesting():
    # fa = SlowFourierAnalyzer(np.sin(np.linspace(0,2*np.pi,10)),np.linspace(0,10,10)) # what the FFT sees:
    # x=np.linspace(0,2*np.pi,10)
    # x=np.arange(0,10)/10
    x=np.linspace(0,1,9)
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

def dim3hypersphereTesting():


if __name__=='__main__':
    # demo finding the mean plane in 3d
    dummyTest3d = concaveHypersphere(numsmpl)
    run3danalysis(dummyTest3d)
    mp=MeanPlane(dummyTest3d)

    xx,yy=map(lambda ar: ar.flatten(), np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128)))
    zz=np.sin(2*np.pi*xx)*np.cos(2*np.pi*yy)

    fa=SlowFourierAnalyzer(zz,np.vstack((xx,yy)).T)
    fa=SlowFourierAnalyzer.fromMeanPlane(mp)
    plt.figure()
    spectral2dPowerPlot(fa)
    plt.show()

    plt.figure()
    spectral2dPowerImage(fa)
    plt.show()
