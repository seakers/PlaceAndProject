import numpy as np
import matplotlib.pyplot as plt
from blist import sortedlist
import itertools as it
from functools import wraps
import bisect as bi
import collections as col
from functools import partial
from common import *
import scipy.stats as sps

def preprocSampleAndAxes(functionToPreproc):
    """ decorator function for preprocessing and posprocessing a function which picks subarrays for sampleToFollow, obj1, obj2
    :sampleToFollowIndx: samples to follow
    :obj1: objectives to consider as the first of the objective indexes
    :obj2: objectives to consider as the second of the objective indexes
    :return:
    """
    @wraps(functionToPreproc)
    def preprocedDefaults(self, *args, sampleToFollowIndx=None, obj1=None, obj2=None, **kwargs):
        if sampleToFollowIndx is None:
            return np.squeeze(preprocedDefaults(self, *args, sampleToFollowIndx=np.arange(0, self.nsmpl), obj1=obj1, obj2=obj2,**kwargs))
        if obj1 is None:
            return np.squeeze(preprocedDefaults(self, *args, sampleToFollowIndx=sampleToFollowIndx, obj1=np.arange(0, self.nobj), obj2=obj2, **kwargs))
        if obj2 is None:
            return np.squeeze(preprocedDefaults(self, *args, sampleToFollowIndx=sampleToFollowIndx, obj1=obj1, obj2=np.arange(0, self.nobj), **kwargs))
        sampleToFollowIndx=numpyze(sampleToFollowIndx)
        obj1=numpyze(obj1)
        obj2=numpyze(obj2)
        return functionToPreproc(self, sampleToFollowIndx, obj1, obj2)
    return preprocedDefaults

def lenArrOrScalar(arrOrScalar):
    """  returns the length of the arrOrScalar. Or 1 if it's a scalar
    :param arrOrScalar: input to find length of
    :return: length og the arrOrScalar
    """
    if hasattr(arrOrScalar,'__len__'):
        return len(arrOrScalar)
    else:
        return 1

def conditionalCovar(paretoSamples):
    Nsmpl,Nobj=paretoSamples.shape

    # reduce to the 2 objective case
    if Nobj>2:
        accum=np.zeros((Nsmpl, Nobj,Nobj))
        for pair in it.combinations(range(Nobj),2):
            accum[:, pair[0],pair[1]]=conditionalCovar(paretoSamples[:,pair])
            accum[:, pair[1],pair[0]]=accum[:,pair[0],pair[1]]
        return accum
    if Nobj<2:
        raise ValueError('should have >=2 objectives in tradeoff index')

    means=np.mean(paretoSamples, axis=0,keepdims=True)
    iTerms=np.prod(paretoSamples-means, axis=1)
    sampleCov=np.sum(iTerms)/(Nsmpl-1)
    return Nsmpl/(Nsmpl-1) * iTerms + sampleCov

def fastTradeIndex(paretoSamples):
    Nsmpl,Nobj=paretoSamples.shape

    # reduce to the 2 objective case
    if Nobj>2:
        accum=np.zeros((Nsmpl, Nobj,Nobj))
        for pair in it.combinations(range(Nobj),2):
            accum[:, pair[0],pair[1]]=fastTradeIndex(paretoSamples[:,pair])
            accum[:, pair[1],pair[0]]=accum[:,pair[0],pair[1]]
        return accum
    if Nobj<2:
        raise ValueError('should have >=2 objectives in tradeoff index')

    # sort the elements so that the left is defined as 0-n, right corresponds, and if have an equality, will have ordering such that same on either side
    # e[i] is an i->e map. the ith position is the eth element

    Lsort=np.lexsort(paretoSamples[:,(1,0)].T) # dunno if faster or slower than argsort. might just be doing sequential mergesorts
    # i[e] is an e->i map where e is the element and i is the position in the sorted list
    # notice that the unindexed list is equivalent to array[1:len]-- a map from the domain to the output of the map
    LinvSort=-np.ones(Lsort.shape,dtype=np.intc)
    LinvSort[Lsort]=np.arange(Nsmpl, dtype=np.intc)

    Rsort=np.lexsort(paretoSamples[Lsort,:].T) #paretoSamples[:,(0,1)]
    # # by first rearranging right, will ignore equalities. Reverse ordering to count equalities.
    # Lsort=np.argsort(paretoSamples[:,0],kind='mergesort')
    # rightInLeftOrder=paretoSamples[Lsort,1]
    # # problem, if left side have equal elements and the right swaps, get wrong pattern
    # Rsort=np.argsort(rightInLeftOrder, kind='mergesort')
    # # solution, isolate the left side equality cases, rarrange locally by the values of corresopnding elments on the right
    # Lvals=paretoSamples[Lsort,0]
    # sameAsNext=np.isclose(Lvals[1:len(Lvals)], Lvals[0:(len(Lvals)-1)])
    # sameLowIndx=np.nonzero(sameAsNext)[0]
    # continuityKey=np.cumsum(np.logical_not(sameAsNext))
    # for trash,lEqual in it.groupby(sameLowIndx, key=lambda i: continuityKey[i]): # lEqual is indicies such that i,i+1 are equal on the left, in sorted
    #     fullLeq=list(lEqual);
    #     fullLeq.append(fullLeq[-1]+1)
    #     fullLeq=np.array(fullLeq)
    #     rarrLarg=np.argsort(Rsort[fullLeq])
    #     Lsort[fullLeq]=Lsort[fullLeq[rarrLarg]]
    # # swapping around Lsort doesn't do anything. I think I actually need to toy with invRightSort and rightSort...if swap out values of equal left values, nothing actually changes...


    #from now on, left is 0-n elements
    # work out mapping on left side due to edges
    RinvSort=-np.ones(Rsort.shape,dtype=np.intc)
    RinvSort[Rsort]=np.arange(Nsmpl, dtype=np.intc)

    #Lsort is the number of elements above on the left (entering from left)
    crossings=-np.ones(Nsmpl)

    funnySlashDirLeftLogical=np.arange(Nsmpl)<=RinvSort
    funnySlashDirLeft=np.nonzero(funnySlashDirLeftLogical)[0]
    crossings[funnySlashDirLeftLogical]=__computeFromFunnyCrossing(Rsort, RinvSort, funnySlashDirLeft, Nsmpl)

    # now swap roles of right and left to get the remaining crossings
    slashDir=np.sort(RinvSort[~funnySlashDirLeftLogical])
    crossings[Rsort[slashDir]]=__computeFromFunnyCrossing(RinvSort, Rsort, slashDir, Nsmpl)

    return crossings[LinvSort]/(Nsmpl-1)
def __computeFromFunnyCrossing(Rsort, RinvSort, filter, Nsmpl):
    leftInvReduce=len(filter)*np.ones(Nsmpl) # gives the e=lp->lrp=lre where lre is the element in reduced space
    leftInvReduce[filter]=np.arange(len(filter))
    leftPosFiltered=np.arange(Nsmpl)[filter] # gives the filtered position on the left in origianl coordinates
    rightPosFiltered=RinvSort[filter] # gives the filtered position on the right in origianl coordinates
    funnySlashDirRight=np.sort(rightPosFiltered)
    rightInvReduce=len(funnySlashDirRight)*np.ones(Nsmpl) # gives the rp -> rre map where rp is the right position and rre is the position on the right in reduced space
    rightInvReduce[funnySlashDirRight]=np.arange(len(funnySlashDirRight))
    rightPosInRed=rightInvReduce[rightPosFiltered] # position of filtered elements in rrp coordinates
    funnySlashCrosses=np.fmax(np.arange(len(rightPosInRed))-rightPosInRed,0) # the number of crossings where a \ crosses a less steep \
    reducedCrossings=rightPosFiltered-leftPosFiltered+2*funnySlashCrosses # the number of elments which cross the given element, in lre space
    return reducedCrossings

def tradeoffIndexInPaper(paretoSamples):
    """
    return of the tradeoff index defined in mehmet's paper "Introduction of a Tradeoff Index for Tradespace Exploration"
    as defined in the pseudocode in Fig. 5. Return[:, k,l] is \Lambda_kl
    """
    Nsol,Nobj=paretoSamples.shape

    indx=np.array(np.zeros((Nsol, Nobj, Nobj)))
    #     obj1count=np.tile(np.reshape(paretoSamples, (paretoSamples.shape[0], 1, paretoSamples.shape[1], 1)), (1,paretoSamples.shape[0],1,paretoSamples.shape[1]))
    #     obj2count=np.tile(np.reshape(paretoSamples, (paretoSamples.shape[0], 1, 1, paretoSamples.shape[1])), (1,paretoSamples.shape[0],paretoSamples.shape[1],1))
    #     return np.squeeze(np.sum(((obj1count-obj2count)*())))
    for k in range(Nobj-1):
        for l in range(k+1, Nobj):
            for i in range(Nsol):
                # indx[i,k,l]=0
                # for j in range(Nsol):
                #     indx[i,k,l]+=(paretoSamples[i,k]-paretoSamples[j,k])*(paretoSamples[i,l]-paretoSamples[j,l])<0
                # indx[i,k,l]/=(Nsol-1)
                indx[i, k,l]=np.sum((paretoSamples[i,k]-paretoSamples[:,k])*(paretoSamples[i,l]-paretoSamples[:,l])<0)/(Nsol-1)

    # make symmetric
    for i in range(Nsol):
        indx[i,:,:]+=indx[i,:,:].T

    return indx

breakSort=col.namedtuple('breakSort', 'argArrs invArgSort sortVals')
def sortAndBreak(toSort):
    """ sorts the 1d array to sort and breaks. breaks into subarrays where each subarray has equal valued elements
    :param toSort:
    :return: argArrs= an array of arrays in which each top-level (1st index) array corresponds to a value in sortVals and the 2nd level (2nd index) arrays are lists of indixes in the original toSort which have a given value. The top level array sorts toSort.
             invArgSort= a numpy array from toSort which maps (a[i] <-> i=>a) where i is a position in toSort and a is the index of the sub-array (2nd level array) in argArrs which contains the position i
             sortVals= the unique sorted values in toSort
    """
    argSort=np.argsort(toSort,kind='mergesort')
    sortVals=[]
    invArgSort=np.arange(len(argSort))
    prevElem=toSort[argSort[0]]
    sortVals.append(prevElem)
    ret=[]
    curVal=[argSort[0],]
    invArgSort[curVal]=0
    ret.append(curVal)
    for i in range(1, len(argSort)):
        thisElem=toSort[argSort[i]]
        if thisElem==prevElem:
            curVal.append(argSort[i])
        else:
            curVal=[argSort[i],]
            sortVals.append(thisElem)
            ret.append(curVal)
        invArgSort[argSort[i]]=len(ret)-1
        prevElem=thisElem
    return breakSort(ret, np.array(invArgSort), np.array(sortVals))

class TradeoffAnalyzer():
    # instance variables
    # paretoSample
    # sortSample -- indicies which sort ^ according to each axis
    # countBelow -- position of each element in ^
    def __init__(self, paretoSample):
        self.paretoSample=paretoSample
        self.sortSample=np.argsort(paretoSample, kind='mergesort', axis=0) # a[i] is a i->e map where i is an index location and e is the element in that position
        # better idea. branch this stuff and prune it. just preprocess the right side ordering so equal elements appear in the same order as on the left
#        self.invSortSample=-np.ones(self.sortSample.shape,dtype=np.uintc) # i[e] is an e->i map where e is the element and i is the position in the sorted list
#                                                           # notice that the unindexed list is equivalent to array[1:len]-- a map from the domain to the output of the map
#        self.invSortSample[self.sortSample,np.arange(self.nobj)]=np.tile(np.arange(self.nsmpl, dtype=np.uintc), (self.nobj,1)).T
#        minExecDict={(li, ri) : self.__buildBaseTable(li, ri) for li, ri in it.combinations(range(self.nobj),2)}
        objSorts=[None,]*self.nobj
        for i in range(self.nobj):
            objSorts[i]=sortAndBreak(paretoSample[:,i])
        minExecDict={(li, ri) : self.__buildBaseTable(objSorts[li], objSorts[ri]) for li, ri in it.combinations(range(self.nobj),2)}
        def matrixize(li, ri):
            if li<ri:
                return minExecDict[(li,ri)]
            elif li==ri:
                return np.zeros(self.nsmpl)
            else: # li>ri
                return minExecDict[(ri,li)]
        baseTable=np.vstack(tuple(matrixize(li, ri) for li, ri in it.product(range(self.nobj),repeat=2))).T
        self.baseTable=baseTable.reshape((self.nsmpl, self.nobj, self.nobj))

        # self.countBelow=-np.ones(self.sortSample.shape)
        # sorted=self.paretoSample[self.sortSample,np.arange(self.nobj)]
        # for obj in range(self.nobj):
        #     diffTransIndxs=np.insert(np.where(np.diff(sorted[:,obj]))[0]+1,0,(0))
        #     nReps=np.append(np.diff(diffTransIndxs), self.nsmpl-diffTransIndxs[-1])
        #     self.countBelow[self.sortSample[:,obj],obj]=np.repeat(diffTransIndxs, nReps)


    def __buildBaseTable(self, sortStructLeft, sortStructRight):
        encounterList=sortedlist()
        numCrossList=-np.ones(self.nsmpl)
        for i in range(len(sortStructLeft.argArrs)):
            lCommon=sortStructLeft.argArrs[i]
            for j in range(len(lCommon)):
                rightAbove=sortStructRight.invArgSort[lCommon[j]]
                numSeenLess=bi.bisect_left(encounterList,rightAbove)
                numSeenGreater=len(encounterList)-bi.bisect_right(encounterList, rightAbove)
                encountered=sortStructRight.argArrs[rightAbove]
                for encounter, encI in zip(encountered, range(len(encountered))):
                    numCrossList[encounter]=rightAbove-numSeenLess+numSeenGreater
                encounterList.add(rightAbove)
        return numCrossList

#     def __buildBaseTable(self, axisLeftIndx, axisRightIndx):
#         leftSort=self.sortSample[:,axisLeftIndx]
#         rightSort=self.sortSample[:,axisRightIndx]
#         invLeftSort=self.invSortSample[:,axisLeftIndx]
#         invRightSort=self.invSortSample[:,axisRightIndx]
#         leftToRightPosition=lambda leftPosition: invRightSort[leftSort[leftPosition]] # returns the position in the right list corresponding to the input position in the left list. Reflexsive
#         encounterList=sortedlist()
#         numCrossList=-np.ones(self.nsmpl)
#         for i in range(self.nsmpl):
#             encountered=leftToRightPosition(i)
#             numLess=bi.bisect_left(encounterList,encountered)
#             numGreater=len(encounterList)-bi.bisect_right(encounterList,encountered) # ignoring equality case for now. Possible solution for equality: keep a buffer. write by flushing buffer by encountering a different value. when do compensate
#             numCrossList[encountered]=encountered-numLess+numGreater # possible indexing bug. encountered is a position. May need element.
#             encounterList.add(encountered)
#         return numCrossList[invLeftSort]

    @property
    def nobj(self):
        return self.paretoSample.shape[1]

    @property
    def nsmpl(self):
        return self.paretoSample.shape[0]

    @preprocSampleAndAxes
    def indicatorCovar(self, sampleToFollowIndx, obj1, obj2):
        return self.baseTable[np.ix_(sampleToFollowIndx,obj1,obj2)]/(self.nsmpl-1)

    def indicatorCovarLowerBnd(self, sampleToFollowIndx=None, obj1=None, obj2=None):
        """
        fast return of the tradeoff index defined in mehmet's paper "Introduction of a Tradeoff Index for Tradespace Exploration"

        NOTE: if any parameter is None then the return will return all permuatations of the input values.
        the return is then a multidimensional numpy array indexed in the order (sampleToFollowIndex, obj1, obj2) with the singleton
        dimensions removed (i.e. those which were given a scalar input)
        :param sampleToFollowIndx: the index (i in the paper) of the sample we wish to follow
        :param obj1: the 1st objective to consider in the tradeoff analysis (left). either an array-like of indicies or an index
        :param obj2: the 2nd objective to consider in the tradeoff analysis (right). either an array-like of indicies or an index
        :return: the tradeoffs as a multidimensional numpy array
        """
        # input processing
        sampleToFollowIndx, obj1, obj2= self.__recursivePreProc(self.indicatorCovarLowerBnd(), sampleToFollowIndx, obj1, obj2)
        obj1count=np.tile(np.reshape(self.countBelow[sampleToFollowIndx, obj1], (len(sampleToFollowIndx), len(obj1), 1)),(1,1,len(obj2)))
        obj2count=np.tile(np.reshape(self.countBelow[sampleToFollowIndx, obj2], (len(sampleToFollowIndx), 1, len(obj2))),(1,len(obj1),1))
        return np.abs(obj2count[np.ix_(sampleToFollowIndx, obj1,obj2)]-obj1count[np.ix_(sampleToFollowIndx, obj1,obj2)])/(self.nsmpl-1) # the joys of numpy. never what youd initally expect

def _basicTesting(testPts):
    # draw3dSurface(testPts)
    print('test points')
    print(testPts)
    # tradeoffs=TradeoffAnalyzer(testPts).indicatorCovar(sampleToFollowIndx=1,obj2=2)
    tradeoffs=fastTradeIndex(testPts[:,(0,2)])
    refTradeoffs=tradeoffIndexInPaper(testPts)[:,0,2]
    maxErr=np.max(np.abs(tradeoffs-refTradeoffs))
    print('worst error')
    print(maxErr)

    if maxErr>1e-10:
        print('difference')
        print(tradeoffs-refTradeoffs)

        print("\n tradeoffs")
        print(tradeoffs)

        print("\n tradeoffs for referenece")
        print(refTradeoffs)

def _timingTesting():
    # test the faster and original indicators.
    # numTest=10**np.linspace(1,6,64)
    numTest=np.ceil(np.linspace(1,5012, 64))
    # numTest=np.ceil(np.linspace(1,1000, 2))
    totalNumTest=numTest[-1]
    testPts=concaveHypersphere(totalNumTest)
    timesMehmet=-np.ones(numTest.shape)
    timesFaster=-np.ones(numTest.shape)

    import gc
    import timeit as tt
    import pandas as pd

    # fasterCall=lambda ps: TradeoffAnalyzer(ps).indicatorCovar()
    fasterCall=fastTradeIndex
    nRepsForCPUerr=100
    # nRepsForCPUerr=1
    for i,n in enumerate(numTest):
        print(n)
        thisTest=testPts[0:n, :]
        fromPaper=partial(tradeoffIndexInPaper,thisTest)
        newMethod=partial(fasterCall,thisTest)
        # gc.collect() # not useful if saving results. Clear GC to prevent re-collect in large runs
        timesMehmet[i]=tt.timeit(fromPaper, number=nRepsForCPUerr)/nRepsForCPUerr
        print('old'+str(timesMehmet[i]))
        timesFaster[i]=tt.timeit(newMethod, number=nRepsForCPUerr)/nRepsForCPUerr
        print('new'+str(timesFaster[i]))

    asDF=pd.DataFrame(np.vstack((timesFaster, timesMehmet)).T, index=numTest, columns=("new method","old method"))
    asDF.to_hdf('tradeoffRunTestResults.hdf','testResults')

    asDF.plot()
    plt.xlabel('number of points in frontier')
    plt.ylabel('runtime (s)')
    plt.show()
    return (testPts, asDF)

if __name__=="__main__":
    # testPts, res=_timingTesting()
    testPts=concaveHypersphere(8)
    _basicTesting(testPts)
