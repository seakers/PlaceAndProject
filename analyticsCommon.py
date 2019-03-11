import scipy as sp

from common import *

def filteredSpectrum(inputFilters, spectralFilters,frequencies,locations,heights,forwardTransform):
    """
    :param inputFilters: queue of filters to run on locations and heights before computation of forward transform
    :param spectralFilters: queue of filters to run on spectra after forward transform
    :param frequencies: passed to forward transform as 1st argument
    :param locations: locations (X) for forward transform. filtered by inputFilters
    :param heights: heights (Y) for forward transform. filtered by inputFilters
    :param forwardTransform: function to go from real space (X,Y) to spectral space
    :return: filtered frequencies
    """
    filtLoc=locations
    filtHeight=heights
    for filt in inputFilters:
        filtLoc, filtHeight=filt(filtLoc,filtHeight)
    ret=forwardTransform(frequencies, filtLoc, filtHeight)
    for filt in spectralFilters:
        ret=filt(frequencies,ret)
    return ret

def spectral1dPowerPlot(fourierAnalyzerObj):
    spectralPower=np.abs(np.fft.fftshift(fourierAnalyzerObj.spectrum))**2
    plt.plot(np.fft.fftshift(fourierAnalyzerObj.fftFreqs),spectralPower,'k.-')
    # axis_font={'size':'28'}
    # plt.xlabel('frequency',**axis_font)
    # plt.ylabel('square power',**axis_font)
    plt.xlabel('frequency')
    plt.ylabel('square power')
def spectral1dPhasePlot(fourierAnalyzerObj):
    spectralPhase=np.angle(np.fft.fftshift(fourierAnalyzerObj.spectrum))
    # print(spectralPhase)
    plt.plot(np.fft.fftshift(fourierAnalyzerObj.fftFreqs),spectralPhase,'k.-')
    # axis_font={'size':'28'}
    # plt.xlabel('frequency',**axis_font)
    # plt.ylabel('phase (radians)',**axis_font)
    plt.xlabel('frequency')
    plt.ylabel('phase (radians)')

def spectral2dPowerPlot_nonFFT(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    freqProd=np.meshgrid(*fourierAnalyzerObj.fullOrders, indexing='ij')
    ax=prep3dAxes()
    ax.plot_surface(freqProd[0],freqProd[1],spectralPower)

def spectral2dPowerImage_nonFFT(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    plt.imshow(spectralPower,cmap=globalCmap,interpolation='nearest')
    plt.colorbar()
    orders=fourierAnalyzerObj.fullOrders
    plt.xticks(orders[0], numpyToPrettyStr(orders[0]), rotation=60)
    plt.yticks(orders[1], numpyToPrettyStr(orders[1]))

def spectral2dPowerPlot(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    freqProd=np.meshgrid(*fourierAnalyzerObj.fftFreqs, indexing='ij')
    ax=prep3dAxes()
    ax.plot_surface(freqProd[0],freqProd[1],spectralPower)

def spectral1dPowerPlot_nonFFT(analyzerObj):
    spectralPower=np.abs(analyzerObj.spectrum)**2
    plt.plot(analyzerObj.fullOrders,spectralPower,'k.-')
    # axis_font={'size':'28'}
    # plt.xlabel('frequency',**axis_font)
    # plt.ylabel('square power',**axis_font)
    plt.xlabel('frequency')
    plt.ylabel('square power')

def spectral2dPowerImage(fourierAnalyzerObj):
    spectralPower=np.abs(fourierAnalyzerObj.spectrum)**2
    # plt.imshow(np.fft.fftshift(spectralPower),cmap=globalCmap,interpolation='nearest')
    plt.imshow(np.fft.fftshift(spectralPower),cmap=globalCmap,interpolation='nearest')
    plt.colorbar()
    shiftedFFTF=np.fft.fftshift(fourierAnalyzerObj.fftFreqs,axes=1)
    plt.xticks(np.arange(len(shiftedFFTF[0])), numpyToPrettyStr(shiftedFFTF[0]), rotation=60)
    plt.yticks(np.arange(len(shiftedFFTF[1])), numpyToPrettyStr(shiftedFFTF[1]))
    # plt.xticks(np.arange(len(fourierAnalyzerObj.fftFreqs[0])), fourierAnalyzerObj.fftFreqs[0], rotation=60)
    # plt.yticks(np.arange(len(fourierAnalyzerObj.fftFreqs[1])), fourierAnalyzerObj.fftFreqs[1])

def reconstructInOriginalSpace(meanPlane, analyzer, Z=None):
    if Z is None:
        Z=analyzer.pointLocation
    spectralCurveOutOfPlane=analyzer.reconstruction(Z)
    meanCurve=meanPlane.pointsInOriginalCoors(Z)
    spectralCurve=meanCurve+np.dot(spectralCurveOutOfPlane[:,np.newaxis],meanPlane.normalVect[np.newaxis,:])
    return spectralCurve

def approximationPlot2d(meanPlane, analyzer,objLabels=None,planeLocations=None):
    dummyTest2d=meanPlane.paretoSamples
    mp=meanPlane
    # plt.plot(mp._centeredSamples[:,0],mp._centeredSamples[:,1])
    plt.plot(dummyTest2d[:,0],dummyTest2d[:,1],'k.',label='Pareto Points')
    plt.plot(mp.inputProjections[:,0],mp.inputProjections[:,1],'kx',label='ProjectedLocations')

    if planeLocations is None:
        spectralCurveInPlane=np.linspace(mp.inputInPlane.min(),mp.inputInPlane.max(),10*mp.inputInPlane.size)
    else:
        spectralCurveInPlane=planeLocations
    planeCurve=meanPlane.pointsInOriginalCoors(spectralCurveInPlane)
    spectralCurve=reconstructInOriginalSpace(meanPlane, analyzer, spectralCurveInPlane)

    plt.plot(planeCurve[:,0],planeCurve[:,1],'k--',label='mean plane')
    plt.plot(spectralCurve[:,0],spectralCurve[:,1],'k-',label='reconstructed curve')
    plt.axis('equal')
    if objLabels is not None:
        plt.xlabel(objLabels[0])
        plt.ylabel(objLabels[1])
    plt.legend(loc='best')

def approximationPlot3d(mp,fa):
    grid_x,grid_y=np.meshgrid(np.linspace(np.min(mp.inputInPlane[:,0]),np.max(mp.inputInPlane[:,0])), np.linspace(np.min(mp.inputInPlane[:,1]),np.max(mp.inputInPlane[:,1])))
    points=np.vstack((grid_x.flatten(),grid_y.flatten())).T
    recons=np.real(fa.reconstruction(locations=points)).reshape(grid_x.shape)

    plt.imshow(recons, cmap=globalCmap,origin='lower',extent=(np.min(mp.inputInPlane[:,0]),np.max(mp.inputInPlane[:,0]), np.min(mp.inputInPlane[:,1]),np.max(mp.inputInPlane[:,1])))
    plt.plot(mp.inputInPlane[:,0],mp.inputInPlane[:,1],'k.')
    plt.colorbar()

def stdNumPerSide(dataShape):
    nthRoot=int(np.ceil(dataShape[0]**(1/dataShape[1])))
    return (nthRoot,)*dataShape[1]

def interpolateErr(locations, values, numPerSide=None):
    if numPerSide is None:
        numPerSide=stdNumPerSide(locations.shape)
    gridLocs=list(map(lambda lo, hi, n: np.linspace(lo, hi, n), np.min(locations, axis=0), np.max(locations, axis=0), numPerSide))
    freqProd=tuple(np.meshgrid(*gridLocs))
    return sp.interpolate.griddata(locations, values, freqProd,method='nearest')
    # grid_x,grid_y=np.meshgrid(np.linspace(np.min(locations[:,0]),np.max(locations[:,0])), np.linspace(np.min(locations[:,1]),np.max(locations[:,1])))
    # return sp.interpolate.griddata(locations, values, (grid_x,grid_y),method='nearest')

def plot3dErr(locations, values):
    pltVals=interpolateErr(locations,values)
    plt.imshow(pltVals, cmap=globalCmap, origin='lower',extent=(np.min(locations[:,0]),np.max(locations[:,0]), np.min(locations[:,1]),np.max(locations[:,1])))
    plt.colorbar()
    plt.plot(locations[:,0],locations[:,1], 'k.')

def runShowSaveClose(toPlot, saveName=None, displayFig=True):
    plt.figure()
    toPlot()
    if saveName is not None:
        plt.savefig(saveName,bbox_inches='tight')
    plt.show()
    if not displayFig:
        plt.close('all')

