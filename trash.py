def plotTradeRatios(mp, objLabels):
    """
    plots ratios of components of the plane. each box represents the value of the objectives trading between each other when restricted to the plane
    :param mp: a mean plane object to plot
    :return:
    """
    # reorder elements
    tr=mp.tradeRatios
    reorderArr=np.argsort(np.mean(tr,axis=0))
    trr=tr[:,reorderArr]
    trr=trr[reorderArr,:]
    objLabels_reorder=list(map(lambda i: objLabels[i], range(len(objLabels))))
    plt.imshow(trr,cmap='Grey')
    plt.colorbar()
    plt.xticks(objLabels_reorder)
    plt.yticks(objLabels_reorder)


# belongs in MeanPlane
    @property
    def tradeRatios(self):
        """
        :return: returns
        """
        return np.tile(self.normalVect[:,np.newaxis],(1,len(self.normalVect)))/np.tile(self.normalVect[np.newaxis,:],(len(self.normalVect,1)))
