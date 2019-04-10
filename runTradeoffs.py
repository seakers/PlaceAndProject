from FourierFitting import fourierAnalytics as fA
from RBFN import rbfAnalytics as rA
from polyfitting import legender as lA
import itertools as it
import numpy as np
import pandas as pd
import os
import warnings as w

from MeanPlanes import meanPlane as mp
from plottingStuffAndDecisionMakers import comparePlots as cP


def anyCompetion(sample):
    ranks=np.argsort(sample, axis=0)
    ret=np.zeros((sample.shape[1],sample.shape[1]), dtype=bool)
    for i,j in it.combinations(range(sample.shape[0]),2):
        ret[i,j]=np.any(ranks[:,i]!=ranks[:,j])
        ret[j,i]=ret[i,j]
    return ret

def covarCompete(sample):
    cov=np.dot(sample.T,sample)
    return cov<0

def printCooperating(competeMatrix,objNames=None):
    if objNames is None:
        objNames=['obj: '+str(i) for i in range(competeMatrix.shape[0])]
    coopObj=np.nonzero(np.logical_not(competeMatrix))
    skipList=set()
    for i,j in zip(coopObj[0], coopObj[1]):
        if (j,i) not in skipList:
            skipList.add((j,i))
            print(objNames[i],objNames[j])


def fixHeader(filename):
    """
    strips out excess commas in the header line. not sure why they are there.
    :param filename:
    :return:
    """
    with open(filename) as f:
        allLines=f.readlines()
        f.close()
    header=allLines[0].split(',')
    notIsEmpty=lambda s: s and s.rstrip()
    clean=[','.join(s.strip() for s in header if notIsEmpty(s))+'\n',]
    clean+=allLines[1:]
    with open(filename,'w') as f:
        f.writelines(clean)

def dummyPass(*args, **kwargs):
    pass

def runProblemAnalysis(probName,metricsFile,preferenceFile, module, filePrepend="./output/"):
    dataRead=pd.read_csv(metricsFile)
    headers=dataRead.columns.values

    # read preference file and get whether will need to multiply by -1 to make minimization
    with open(preferenceFile) as f:
        f.readline()
        prefVect=f.readline()
    minMaxDict={'min':1, 'max':-1}
    correctedHeaders=headers

    isMax=np.array(list(map(lambda e: e.strip()=='max',prefVect.split(','))))
    multiplier=-2*isMax+1
    sameDirVals=dataRead.values*multiplier[np.newaxis,:]
    appendDict=['','neg. ']
    correctedHeaders=[appendDict[int(isM)]+s for isM,s in zip(isMax,correctedHeaders)]

    normalizedData=(sameDirVals-sameDirVals.min(axis=0)[np.newaxis,:])/np.ptp(sameDirVals,axis=0)[np.newaxis,:]
    correctedHeaders=['FoR '+s for s in correctedHeaders] # "FoR" stands for "Fraction of Range" and corresponds to the normalization of the dataset by min-max.

    meanCoop=covarCompete(normalizedData)
    if np.any(meanCoop):
        w.warn('detected average cooperation in problem: '+probName)
        printCooperating(meanCoop, headers)
    runAnalysisDict=[module.run2danalysis,module.run3danalysis,module.runHighDimAnalysis]
    # runAnalysisDict=[module.run2danalysis,module.run3danalysis,dummyPass]
    try:
        runAnalysisDict[min(len(headers)-2,2)](normalizedData,correctedHeaders,filePrepend+probName,displayFigs=False)
    except mp.NotPointingToOriginError as valErr:
        # print('not pointing to origin: valErr')
        w.warn('not pointing to origin:'+str(valErr))
    # runAnalysisDict[min(len(headers)-2,2)](normalizedData,headers,None) # for testing

def runComparisons(probName,metricsFile,preferenceFile, filePrepend=None):
    dataRead=pd.read_csv(metricsFile)
    headers=dataRead.columns.values

    if filePrepend is not None:
        cps=filePrepend+probName+"_comparePlot.png"
        mps=filePrepend+probName+"_multiplePlot.png"
    else:
        cps=None
        mps=None

    # read preference file and get whether will need to multiply by -1 to make minimization
    with open(preferenceFile) as f:
        f.readline()
        prefVect=f.readline()
    minMaxDict={'min':1, 'max':-1}
    correctedHeaders=headers

    isMax=np.array(list(map(lambda e: e.strip()=='max',prefVect.split(','))))
    multiplier=-2*isMax+1
    sameDirVals=dataRead.values*multiplier[np.newaxis,:]
    appendDict=['','neg. ']
    correctedHeaders=[appendDict[int(isM)]+s for isM,s in zip(isMax,correctedHeaders)]

    normalizedData=(sameDirVals-sameDirVals.min(axis=0)[np.newaxis,:])/np.ptp(sameDirVals,axis=0)[np.newaxis,:] # set to positive orthant [0,1]**d
    correctedHeaders=['FoR '+s for s in correctedHeaders] # "FoR" stands for "Fraction of Range" and corresponds to the normalization of the dataset by min-max.

    meanCoop=covarCompete(normalizedData)
    if np.any(meanCoop):
        w.warn('detected average cooperation in problem: '+probName)
        printCooperating(meanCoop, headers)
    knownClasses=(fA.FourierSummarizerAnalyzer, lA.LegendreSummarizerAnalyzer, rA.rbfSummarizerAnalyzer)
    cP.paramAccuracyPlot(normalizedData,knownClasses,
                         analyzerClassNames=('fourier series','legendre polynomials','exponential RBF NN'), saveFig=cps)
    cP.multipleReconstructionPlotFromData(normalizedData, knownClasses, numTermsToUse=4, analyzerClassNames=None, holdoutData=None, saveFig=mps, displayFig=True, objLabels=None)

if __name__=="__main__":
    metricsFiles=list(filter(lambda f: f[-8:]=='_met.csv' and 'walker' not in f, os.listdir('./cityplotData')))
    # metricsFiles=['continuous6obj_met.csv',]
    # metricsFiles=['EOSSdownSel3_met.csv',]
    # metricsFiles=['EOSSdownSel_met.csv','GNC_scenario_9_met.csv']
    # metricsFiles=['EOSSdownSel_met.csv',]
    # metricsFiles=['GNC_scenario_1_met.csv',]
    for metricsFile in metricsFiles:
        pathedMetricFile=os.path.join('./cityplotData',metricsFile)
        if os.path.isfile(pathedMetricFile):
            pathParts=os.path.split(pathedMetricFile)
            probName=pathParts[-1][:-8]
            print('analyzing: '+probName)
            # fixHeader(pathedMetricFile)
            # runProblemAnalysis(probName,pathedMetricFile,r'./cityplotData/'+probName+'_pref.csv', fA, filePrepend=r'./output/fourier_')
            runProblemAnalysis(probName,pathedMetricFile,r'./cityplotData/'+probName+'_pref.csv', lA, filePrepend=r'./output/legendre_')
            # runProblemAnalysis(probName,pathedMetricFile,r'./cityplotData/'+probName+'_pref.csv', rA, filePrepend=r'./output/rbf_')
            # runComparisons(probName, pathedMetricFile, r'./cityplotData/'+probName+'_pref.csv', filePrepend=r'./output/')
        else:
            print('skipped file: '+pathedMetricFile)
