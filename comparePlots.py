import numpy as np
import meanPlane as mP
import matplotlib.pyplot as plt

import common as cmn
import analyticsCommon as aC

def reconstructionErr(refY, meanPlane, analyzer):
    return np.squeeze(np.mean(np.linalg.norm(refY-aC.reconstructInOriginalSpace(meanPlane, analyzer, meanPlane.projectToPlaneCoor(refY))))) # hack. assumes refY are training points

#slow, ugly, ugly hack. Better to modify the analyzers to give incremental output than just making a bunch of analyzers. TODO
def paramAccuracyPlot(data, analyzerClasses, maxTerms=4, analyzerClassNames=None, holdoutData=None, saveFig=None, displayFig=True):
    if analyzerClassNames is None:
        analyzerClassNames=['analyzer'+str(n) for n, c in enumerate(analyzerClasses)]
    if holdoutData is None:
        holdoutData=data
    commonMeanPlane=mP.lowDimMeanPlane(data)
    models=[[c.fromMeanPlane(commonMeanPlane, n) for n in range(1,maxTerms)] for c in analyzerClasses]
    for i,modelClass in enumerate(models):
        errs=np.array([reconstructionErr(data, commonMeanPlane, m) for m in modelClass])
        plt.plot(range(1,maxTerms), errs)
    plt.legend(analyzerClassNames)
    plt.xlabel('number of significant terms')
    plt.ylabel('reconstruction error')
    if saveFig is not None:
        plt.savefig(saveFig, bbox_inches='tight')
    plt.show()
    if not displayFig:
        plt.close('all')
