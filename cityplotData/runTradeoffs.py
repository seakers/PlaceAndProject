from common import *
from fourierAnalytics import *
import numpy as np
import pandas as pd
import os
import functools as ft
import operator as op
import warnings as w

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

def runProblemAnalysis(probName,metricsFile,preferenceFile):
    dataRead=pd.read_csv(metricsFile)
    headers=dataRead.columns.values

    # read preference file and get whether will need to multiply by -1 to make minimization
    with open(preferenceFile) as f:
        f.readline()
        prefVect=f.readline()
    minMaxDict={'min':1, 'max':-1}
    multiplier=np.array(list(map(lambda e: minMaxDict[e.strip()],prefVect.split(','))))

    sameDirVals=dataRead.values*multiplier[np.newaxis,:]
    normalizedData=(sameDirVals-sameDirVals.min(axis=0)[np.newaxis,:])/np.ptp(sameDirVals,axis=0)[np.newaxis,:]

    meanCoop=covarCompete(normalizedData)
    if np.any(meanCoop):
        w.warn('detected average cooperation in problem: '+probName)
        printCooperating(meanCoop, headers)
    runAnalysisDict=[run2danalysis,run3danalysis,runHighDimAnalysis]
    runAnalysisDict[min(len(headers)-2,2)](normalizedData,headers,probName)
    # runAnalysisDict[min(len(headers)-2,2)](normalizedData,headers,None) # for testing

if __name__=="__main__":
    metricsFiles=list(filter(lambda f: f[-8:]=='_met.csv', os.listdir()))
    # metricsFiles=['continuous6obj_met.csv',]
    # metricsFiles=['EOSSdownSel_met.csv',]
    for metricsFile in metricsFiles:
        if os.path.isfile(metricsFile):
            pathParts=os.path.split(metricsFile)
            probName=pathParts[-1][:-8]
            print('analyzing: '+probName)
            # fixHeader(metricsFile)
            runProblemAnalysis(probName,metricsFile,probName+'_pref.csv')
        else:
            print('skipped file: '+metricsFile)
