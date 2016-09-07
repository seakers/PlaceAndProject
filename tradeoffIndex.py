import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #it's tempting. don't delete.
from numbers import Number
from blist import sortedlist
import itertools as it
from functools import wraps
import bisect as bi

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

def numpyze(input):
    """
    Attempts to turn the input into a numpy array. Of note, if input is scalar, will turn into a singleton array
    This is different from numpy.asarray which will make the wrapped scalar relatively shallow
    :param input:
    :return:
    """
    if isinstance(input, np.ndarray):
        return input
    if isinstance(input, Number):
        return np.array((input,))
    if hasattr(input, (list, tuple)):
        return np.array(input)
    if hasattr(input, '__iter__'):
        return np.fromiter(input)
    raise TypeError('cannot cast '+str(input)+' into a numpy array')

def tradeoffIndexInPaper(paretoSamples):
    """
    return of the tradeoff index defined in mehmet's paper "Introduction of a Tradeoff Index for Tradespace Exploration"
    as defined in the pseudocode in Fig. 5. Return[:, k,l] is \Lambda_kl
    """
    Nsol,Nobj=paretoSamples.shape

    indx=np.array(np.zeros((Nsol, Nobj, Nobj)))
    for k in range(Nobj-1):
        for l in range(k+1, Nobj):
            for i in range(Nsol):
                indx[i, k,l]=np.sum((paretoSamples[i,k]-paretoSamples[:,k])*(paretoSamples[i,l]-paretoSamples[:,l])<0)/(Nsol-1)
    for i in range(Nsol):
        indx[i,:,:]+=indx[i,:,:].T
    return indx

# def vectorizedTradeoffIndexInPaper(paretoSamples):
#     """
#
#     :param paretoSamples:
#     :return:
#     """
#     obj1count=np.tile(np.reshape(paretoSamples, (paretoSamples.shape[0], 1, paretoSamples.shape[1], 1)), (1,paretoSamples.shape[0],1,paretoSamples.shape[1]))
#     obj2count=np.tile(np.reshape(paretoSamples, (paretoSamples.shape[0], 1, 1, paretoSamples.shape[1])), (1,paretoSamples.shape[0],paretoSamples.shape[1],1))
#     return np.squeeze(np.sum(((obj1count-obj2count)*())))


class TradeoffAnalyzer():
    # instance variables
    # paretoSample
    # sortSample -- indicies which sort ^ according to each axis
    # countBelow -- position of each element in ^
    def __init__(self, paretoSample):
        self.paretoSample=paretoSample
        self.sortSample=np.argsort(paretoSample, kind='mergesort', axis=0) # a[i] is a i->e map where i is an index location and e is the element in that position
        self.invSortSample=-np.ones(self.sortSample.shape,dtype=np.uintc) # i[e] is an e->i map where e is the element and i is the position in the sorted list
                                                           # notice that the unindexed list is equivalent to array[1:len]-- a map from the domain to the output of the map
        self.invSortSample[self.sortSample,np.arange(self.nobj)]=np.tile(np.arange(self.nsmpl, dtype=np.uintc), (self.nobj,1)).T
        minExecDict={(li, ri) : self.__buildBaseTable(li, ri) for li, ri in it.combinations(range(self.nobj),2)}
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

    def __buildBaseTable(self, axisLeftIndx, axisRightIndx):
        leftSort=self.sortSample[:,axisLeftIndx]
        rightSort=self.sortSample[:,axisRightIndx]
        invLeftSort=self.invSortSample[:,axisLeftIndx]
        invRightSort=self.invSortSample[:,axisRightIndx]
        leftToRightPosition=lambda leftPosition: invRightSort[leftSort[leftPosition]] # returns the position in the right list corresponding to the input position in the left list. Reflexsive
        encounterList=sortedlist()
        numCrossList=-np.ones(self.nsmpl)
        for i in range(self.nsmpl):
            encountered=leftToRightPosition(i)
            numLess=bi.bisect_left(encounterList,encountered)
            numGreater=len(encounterList)-bi.bisect_right(encounterList,encountered) # ignoring equality case for now.
            numCrossList[encountered]=encountered-numLess+numGreater
            encounterList.add(encountered)
        return numCrossList[invLeftSort]

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

def concaveHypersphere(numTest):
    # rands=np.random.rand(2, numTest)
    # rands=np.tile(np.arange(numTest)/(numTest+1), (2,1))
    rands=np.vstack((np.ones(numTest)/2, np.arange(numTest)/(numTest+1)))
    z=rands[0,:]
    altRad=np.sqrt(1-z**2)
    return np.vstack((altRad*np.cos(np.pi/2*rands[1,:]), altRad*np.sin(np.pi/2*rands[1,:]), z)).T

def draw3dSurface(points):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2], '.')
    ax.legend()
    plt.show()

if __name__=="__main__":
    # test the faster and original indicators.
    numTest=4
    testPts=concaveHypersphere(numTest)
    # draw3dSurface(testPts)
    print('test points')
    print(testPts)
    tradeoffs=TradeoffAnalyzer(testPts).indicatorCovar(sampleToFollowIndx=1,obj2=2)
    refTradeoffs=tradeoffIndexInPaper(testPts)[1,2,:]
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
