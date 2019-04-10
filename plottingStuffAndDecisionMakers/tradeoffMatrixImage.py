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
    plt.imshow(trr,cmap=globalCmap,interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(objLabels_reorder)),objLabels_reorder)
    plt.yticks(range(len(objLabels_reorder)),objLabels_reorder)

import scipy
def makeSquare(A):
    if A.shape[1]>A.shape[0]:
        return np.vstack((A,np.zeros((A.shape[1]-A.shape[0],A.shape[1]))))
    elif A.shape[0]>A.shape[1]:
        return makeSquare(A.T).T
    else:
        return A

def nullspace(A, eps=1e-13):
    A=makeSquare(A)
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def derivativeMat(meanPlane, fourierAnalyzer, loc):
    if len(loc.shape) == 0:
        singleLoc=np.array([loc,])
    else:
        singleLoc=loc
    singleLoc=singleLoc[np.newaxis,:]
    thisDerivative=fourierAnalyzer.reconstructDerivative(singleLoc)
    if np.isscalar(thisDerivative):
        thisDerivative=np.array([thisDerivative,])
    return meanPlane.projectionToPlaneMat.T+np.dot(meanPlane.normalVect[:,np.newaxis],thisDerivative[np.newaxis,:])

def tradeoffCalcSingleLoc(meanPlane, fourierAnalyzer,loc,obj1,obj2):
    locX=meanPlane.projectToPlaneCoor(loc)
    gradYwrtX=derivativeMat(meanPlane,fourierAnalyzer,locX)
    if gradYwrtX.shape[1]>1:
        constraintMat=gradYwrtX[np.setdiff1d(np.arange(gradYwrtX.shape[0]),np.array([obj1,obj2])),:]
        dirToTravel=nullspace(constraintMat) # i feel like this is probably something tame that oculd be saved and simplified over multiple computations.
        dirToTravel/=np.linalg.norm(dirToTravel)
    else:
        dirToTravel=np.array([1,])[:,np.newaxis]
    moveInY=np.real(np.dot(gradYwrtX,dirToTravel)) # should be real
    return -moveInY[obj2]/moveInY[obj1]

def tradeoffCalc(meanPlane, fourierAnalyzer,locations,obj1,obj2):
    accum=[]
    for loc in locations:
        accum.append(tradeoffCalcSingleLoc(meanPlane,fourierAnalyzer,loc,obj1,obj2))
    return np.array(accum)

def plotTradeRatios(mp, fa, objLabels,preconditioner=None,numToSample=75,pixPerSide=10):
    if preconditioner is None:
        tr=mp.tradeRatios
    else:
        tr=preconditioner(mp.tradeRatios)

    if numToSample is None:
        samples=mp.paretoSamples
    else:
        dataSize=mp.paretoSamples.shape[0]
        smplIndxsWithReplacement=np.random.randint(0,dataSize,size=min(dataSize,numToSample))
        samples=mp.paretoSamples[smplIndxsWithReplacement,:]
    U,S,V=np.linalg.svd(samples)
    locs2d=np.dot(samples,V)[:,:2]
    ranges=np.ptp(locs2d,axis=0)
    locs2dNoised=locs2d+np.random.random(locs2d.shape)*ranges[np.newaxis,:]*1/1000
    rangesNoised=np.ptp(locs2dNoised,axis=0)
    # gradient=fa.reconstructDerivative(locations=mp.projectToPlaneCoor(samples))

    gridLocs=(np.mgrid[0:pixPerSide,0:pixPerSide]+0.5)/pixPerSide*(rangesNoised[:,np.newaxis,np.newaxis])
    gridIndxs=np.mgrid[0:pixPerSide, 0:pixPerSide]
    flatLocs=np.array([arr.flatten() for arr in gridLocs]).T
    flatIndxs=np.array([arr.flatten() for arr in gridIndxs]).T
    dists=spst.distance.cdist(flatLocs-np.mean(flatLocs,axis=0)[np.newaxis,:],locs2dNoised-np.mean(locs2dNoised,axis=0)[np.newaxis,:])

    accum=[[None,]*len(tr) for cnt in range(len(tr))]
    for k in range(len(accum)):
        accum[k][k]=np.ones((pixPerSide,pixPerSide))
    print('beginning matching with '+str(dists.shape)+' sized matrix')
    if dists.shape[0]<=dists.shape[1]: # number of drawn pixels < number of elements in the sample
        matches=np.array(Munkres().compute(dists)) # sparse represetnation. [:,0] is the pixel index, [:,1] is matching sample index
        print('finished matching')
        for i,j in it.combinations(range(len(tr)),2):
            imageMat=np.ones((pixPerSide,pixPerSide))*tr[i,j]
            # imageMat[flatIndxs[:,0],flatIndxs[:,1]]=gradient[matches[:,0],i]/gradient[matches[:,1],j]# set to the values of the tradeoffs at teh elements of the samples.
            imageMat[flatIndxs[matches[:,0],0],flatIndxs[matches[:,0],1]]=np.squeeze(tradeoffCalc(mp,fa,samples[matches[:,1],:],i,j))
            accum[i][j]=imageMat
            accum[j][i]=1/imageMat
    else:
        matches=np.array(Munkres().compute(dists.T)) # [:,0] is the sample index, [:,1] is the pixel index

        # check matching
        # plt.plot(flatLocs[:,0],flatLocs[:,1],'.',samples[:,0],samples[:,1],'o')
        # plt.plot(np.vstack((samples[matches[:,0],0],flatLocs[matches[:,1],0])),np.vstack((samples[matches[:,0],1],flatLocs[matches[:,1],1])))
        # plt.show()

        print('finished matching')
        for i,j in it.combinations(range(len(tr)),2):
            imageMat=np.ones((pixPerSide,pixPerSide))*tr[i,j]
            # imageMat[matches[:,0],matches[:,1]]=gradient[flatIndxs[:,0],i]/gradient[flatIndxs[:,1],j]# set to the values of the tradeoffs at teh elements of the samples.
            imageMat[flatIndxs[matches[:,1],0],flatIndxs[matches[:,1],1]]=np.squeeze(tradeoffCalc(mp,fa,samples[matches[:,0],:],i,j))
            accum[i][j]=imageMat
            accum[j][i]=1/imageMat
    # reorderArr=np.argsort(np.mean(tr,axis=0))
    reorderArr=np.arange(len(tr))
    objLabels_reorder=list(map(lambda i: objLabels[i], range(len(objLabels))))
    accumReorder=[[accum[l][k] for k in reorderArr] for l in reorderArr]
    toPlot=np.vstack(tuple(map(np.hstack, map(tuple,accumReorder))))

    toPlot=np.log(np.abs(toPlot))

    plt.imshow(toPlot,cmap=globalCmap,interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(pixPerSide//2,pixPerSide*len(objLabels_reorder),pixPerSide),objLabels_reorder)
    plt.yticks(range(pixPerSide//2,pixPerSide*len(objLabels_reorder),pixPerSide),objLabels_reorder)

logAbs=lambda a: np.log10(np.abs(a))
plotLogTradeRatios=ft.partial(plotMeanPlaneTradeRatios,preconditioner=logAbs)
