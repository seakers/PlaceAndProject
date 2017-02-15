import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spst

from common import *
from meanPlane import *
from fourierAnalytics import *

def plotMeanPlaneTradeRatios(mp, objLabels,preconditioner=None):
    """
    plots ratios of components of the plane. each box represents the value of the objectives trading between each other when restricted to the plane
    :param mp: a mean plane object to plot
    :return:
    """

    if preconditioner is None:
        tr=mp.tradeRatios
    else:
        tr=preconditioner(mp.tradeRatios)

    # reorder elements
    reorderArr=np.argsort(np.mean(tr,axis=0))
    trr=tr[:,reorderArr]
    trr=trr[reorderArr,:]
    objLabels_reorder=list(map(lambda i: objLabels[i], range(len(objLabels))))
    plt.imshow(trr,cmap='Greys',interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(objLabels_reorder)),objLabels_reorder)
    plt.yticks(range(len(objLabels_reorder)),objLabels_reorder)

def plotTradeRatios(mp, objLabels,preconditioner=None,pixPerSide=200):
    if preconditioner is None:
        pmtr=mp.tradeRatios
    else:
        tr=preconditioner(mp.tradeRatios)

    samples=mp.inputInPlane
    U,S,V=np.linalg.svd(samples)
    locs2d=np.dot(U[:-1,:].T,mp.inputInPlane)[:2,:]
    ranges=np.ptp(locs2d,axis=0)
    locs2dNoised=locs2d+np.random.random(locs2d.shape)*ranges[np.newaxis,:]*1/1000
    rangesNoised=np.ptp(locs2dNoised,axis=0)

    gridLocs=(np.meshgrid[0:pixPerSide,0:pixPerSide]+0.5)*(rangesNoised[:,np.newaxis,np.newaxis])
    dists=spst.distance.cdist(gridLocs,locs2dNoised)
    matches=munkres(dists)

    reorderArr=np.argsort(np.mean(tr,axis=0))
    trr=tr[:,reorderArr]
    imageMat=trr[reorderArr,:]
    imageMat[]= # set to the values of the tradeoffs at teh elements of the samples.
    objLabels_reorder=list(map(lambda i: objLabels[i], range(len(objLabels))))

logAbs=lambda a: np.log10(np.abs(a))
plotLogTradeRatios=ft.partial(plotTradeRatios,preconditioner=logAbs)
