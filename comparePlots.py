import numpy as np
import meanPlane as mP
import matplotlib.pyplot as plt
import itertools as it

import common as cmn
import analyticsCommon as aC

def reconstructionErr(refY, meanPlane, analyzer):
    return np.squeeze(np.mean(np.linalg.norm(refY-aC.reconstructInOriginalSpace(meanPlane, analyzer, meanPlane.projectToPlaneCoor(refY))))) # hack. assumes refY are training points

def createModels(data, analyzerClasses, numTermsToUse=None, analyzerClassNames=None):
    # TODO: cache
    if analyzerClassNames is None:
        analyzerClassNames=['analyzer'+str(n) for n, c in enumerate(analyzerClasses)]

    if not hasattr(numTermsToUse, '__len__'):
        numTermsToUse=range(1, numTermsToUse+1)

    print('creating common mean plane')
    commonMeanPlane=mP.lowDimMeanPlane(data)
    print('beginning to build all models')
    models=[[c.fromMeanPlane(commonMeanPlane, n) for n in numTermsToUse] for c in analyzerClasses]
    return commonMeanPlane, models, numTermsToUse, analyzerClassNames

#slow, ugly, ugly hack. Better to modify the analyzers to give incremental output than just making a bunch of analyzers. TODO
def paramAccuracyPlot(data, analyzerClasses, numTermsToUse=4, analyzerClassNames=None, holdoutData=None, saveFig=None, displayFig=True):
    if holdoutData is None:
        holdoutData=data

    commonMeanPlane, models, numTermsToUse, analyzerClassNames = createModels(data, analyzerClasses, numTermsToUse, analyzerClassNames)

    for i,modelClass in enumerate(models):
        print('reconstructing for model: '+ str(i))
        errs=np.array([reconstructionErr(holdoutData, commonMeanPlane, m) for m in modelClass])
        plt.plot(numTermsToUse, errs)
    plt.legend(analyzerClassNames)
    plt.xlabel('number of significant terms')
    plt.ylabel('reconstruction error')
    if saveFig is not None:
        plt.savefig(saveFig, bbox_inches='tight')
    plt.show()
    if not displayFig:
        plt.close('all')

def multipleReconstructionPlotFromData(data, analyzerClasses, numTermsToUse=4, analyzerClassNames=None, holdoutData=None, saveFig=None, displayFig=True, objLabels=None):
    if holdoutData is None:
        holdoutData=data
    commonMeanPlane, models, numTermsToUse, analyzerClassNames = createModels(data, analyzerClasses, numTermsToUse, analyzerClassNames)
    modelsSingle=it.chain.from_iterable(models)
    classNamesSingle=map(lambda cn, n: "".join((cn,' ',str(n),' terms')), it.cycle(analyzerClassNames), it.chain.from_iterable(it.repeat(numTermsToUse, len(analyzerClassNames))))
    multipleReconstructionPlot(commonMeanPlane, modelsSingle, placesToReconstruct=holdoutData, saveFig=saveFig, displayFig=displayFig, objLabels=objLabels, modelNames=classNamesSingle)

def multipleReconstructionPlot(meanPlanes, analyzers, placesToReconstruct=None, saveFig=None, displayFig=True, objLabels=None, modelNames=None):
    if not hasattr(meanPlanes, '__iter__'): # assume is a single, common meanplane
        meanPlane=meanPlanes
        meanPlanes=it.repeat(meanPlane)
        Z=meanPlane.projectToPlaneCoor(placesToReconstruct)
        getZ=lambda mp: Z
    else:
        getZ=lambda mp: meanPlane.projectToPlaneCoor(placesToReconstruct)

    meanPlanesCpy, tmp=it.tee(meanPlanes)
    n=next(tmp).embedDim
    if n==2:
        def allPlots():
            for meanPlane, analyzer in zip(meanPlanesCpy,analyzers):
                inZ=getZ(meanPlane)
                aC.approximationPlot2d(meanPlane, analyzer, objLabels, inZ)
            if modelNames is not None:
                plt.legend(modelNames)
        aC.runShowSaveClose(allPlots, saveName=saveFig,displayFig=displayFig)
    # elif n==3:
    #     def allPlots():
    #         for meanPlane, analyzer in zip(meanPlanesCpy,analyzers):
    #             aC.approximationPlot3d(meanPlane, analyzer)
    #         if modelNames is not None:
    #             plt.legend(modelNames)
    #     aC.runShowSaveClose(allPlots, saveName=saveFig,displayFig=displayFig)
    else:
        pass
