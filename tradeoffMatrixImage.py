import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spst
import munkres

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

def plotTradeRatios(mp, fa, objLabels,preconditioner=None,numToSample=None,pixPerSide=200):
    if preconditioner is None:
        tr=mp.tradeRatios
    else:
        tr=preconditioner(mp.tradeRatios)

    if numToSample is None:
        samples=mp.inputInPlane
    else:
        dataSize=mp.inputInPlane.shape[0]
        samples=mp.inputInPlane[np.random.randint(0,min(dataSize,numToSample)),:]
    U,S,V=np.linalg.svd(samples)
    locs2d=np.dot(U[:-1,:].T,samples)[:2,:]
    ranges=np.ptp(locs2d,axis=0)
    locs2dNoised=locs2d+np.random.random(locs2d.shape)*ranges[np.newaxis,:]*1/1000
    rangesNoised=np.ptp(locs2dNoised,axis=0)
    gradient=fa.reconstructDerivative(samples)

    gridLocs=(np.meshgrid[0:pixPerSide,0:pixPerSide]+0.5)*(rangesNoised[:,np.newaxis,np.newaxis])
    gridIndxs=np.meshgrid[0:pixPerSide, 0:pixPerSide]
    flatLocs=np.array([arr.flatten() for arr in gridLocs])
    flatIndxs=[arr.flatten for arr in gridIndxs]
    dists=spst.distance.cdist(flatLocs,locs2dNoised)

    accum=[([None,]*len(tr))*len(tr)]
    if dists.shape[0]<=dists.shape[1]:
        matches=munkres(dists)
        for i,j in it.combinations(range(len(tr))):
            imageMat=np.ones((pixPerSide,pixPerSide))*tr[i,j]
            imageMat[flatIndxs[:,0],flatIndxs[:,1]]=gradient[matches[:,0],i]/gradient[matches[:,1],j]# set to the values of the tradeoffs at teh elements of the samples.
            accum[i][j]=imageMat
            accum[j][i]=1/imageMat
    else:
        matches=munkres(dists.T)
        for i,j in it.combinations(range(len(tr))):
            imageMat=np.ones((pixPerSide,pixPerSide))*tr[i,j]
            imageMat[matches[:,0],matches[:,1]]=gradient[flatIndxs[:,0],i]/gradient[flatIndxs[:,1],j]# set to the values of the tradeoffs at teh elements of the samples.
            accum[i][j]=imageMat
            accum[j][i]=1/imageMat
    reorderArr=np.argsort(np.mean(tr,axis=0))
    objLabels_reorder=list(map(lambda i: objLabels[i], range(len(objLabels))))
    accumReorder=[[accum[l][k] for k in reorderArr] for l in reorderArr]
    toPlot=np.vstack(tuple(map(np.hstack, map(tuple,accumReorder))))

    plt.imshow(toPlot,cmap='Greys',interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(objLabels_reorder)),objLabels_reorder)
    plt.yticks(range(len(objLabels_reorder)),objLabels_reorder)

logAbs=lambda a: np.log10(np.abs(a))
plotLogTradeRatios=ft.partial(plotTradeRatios,preconditioner=logAbs)
