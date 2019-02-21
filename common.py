import numpy as np
import matplotlib.pyplot as plt
from numbers import Number
from mpl_toolkits.mplot3d import Axes3D #it's tempting. don't delete.

# on caching:
# https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
# https://www.fullstackpython.com/caching.html
# http://book.pythontips.com/en/latest/function_caching.html

# globalCmap='Greys'
# globalCmap='plasma'
globalCmap='YlOrBr'
# globalBarPlotColor='#888888'
globalBarPlotColor='#cc6600'
# globalPlaneColor=[.5,.5,.5]
globalPlaneColor=[204./255.,102./255.,0]

def incToEven(i):
    return i+(i%2)
def decToEven(i):
    return i-(i%2)

def concaveHypersphere(numTest):
    """returns a set of random points on the hypersphere in the all positive quadrant"""
    np.random.seed(2130935987)
    rands=np.random.rand(2, numTest)
    # rands=np.tile(np.arange(numTest)/(numTest+1), (2,1))
    # rands=np.vstack((np.ones(numTest)/2, np.arange(numTest)/(numTest+1)))
    z=rands[0,:]
    altRad=np.sqrt(1-z**2)
    return np.vstack((altRad*np.cos(np.pi/2*rands[1,:]), altRad*np.sin(np.pi/2*rands[1,:]), z)).T

def pathologicalRing(numsmpl,llim=40*np.pi/180,ulim=50*np.pi/180):
    # np.random.seed(2130935987)
    np.random.seed(121310)
    rndmRollTh=np.random.rand(numsmpl)*(ulim-llim)+llim
    z=np.random.rand(numsmpl)
    altRad=np.sqrt(1-z**2)
    return np.vstack((altRad*np.cos(rndmRollTh), altRad*np.sin(rndmRollTh), z)).T

def prep3dAxes():
    fig=plt.gcf()
    ax=fig.add_subplot(111,projection='3d')
    return ax

def quick2dscatter(points):
    """a quick plot made for debugging"""
    plt.plot(points[:,0],points[:,1])
    plt.show()

def draw3dSurface(points):
    ax=prep3dAxes()
    ax.plot(points[:,0], points[:,1], points[:,2], '.')
    ax.legend()
    # plt.show()

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
    if isinstance(input, (list, tuple)):
        return np.array(input)
    if hasattr(input, '__iter__'):
        return np.fromiter(input)
    raise TypeError('cannot cast '+str(input)+' into a numpy array')

def noneSafeConcat(*args):
    if any(e is None for e in args):
        return None
    else:
        return "".join(args)
def numpyToPrettyStr(numpyArr):
    tickFormat=lambda x: "%.4f" % x
    return list(map(tickFormat, numpyArr))

def multiDimNumpyToPrettyStr(numpyArr):
    formatStrBuilder="( "+", ".join(("%.3f",)*numpyArr.shape[1])+")"
    tickFormat=lambda x: formatStrBuilder % tuple(x)
    return list(map(tickFormat, numpyArr))

class InitializeRunFailError(Exception):
    pass

elemBasis=lambda i,dim: np.array([0,]*(i)+[1,]+[0,]*(dim-i-1))
