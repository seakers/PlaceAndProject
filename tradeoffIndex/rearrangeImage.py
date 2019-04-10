import tradeoffIndex as tI
from Common.common import *
from tradeoffIndex import TradeoffAnalyzer
import matplotlib.colors as colors
import matplotlib.cm as mplcm

def indexJKLtoImageMatrix(tensorIn):
    """
    converts the array of [designs, axis1, axis2] to a 2d matrix which can be put through image plot by linearizing the
    1st axis as was done in the paper.
    :param tensorIn:
    :return:
    """
    nsmpl, ax1, ax2=tensorIn.shape
    boxSize=np.ceil(np.sqrt(nsmpl))

    output=np.ones((boxSize*(ax1-1), boxSize*(ax2-1)))
    for i in range(ax1):
        for j in range(i+1,ax2):
            for k in range(nsmpl):
                output[boxSize*i+ np.mod(k,boxSize), boxSize*(j-1)+ np.floor(k/boxSize)]=tensorIn[k,i,j]
    return output

def indexJKLtoImage(tensorIn):
    image=plt.imshow(indexJKLtoImageMatrix(tensorIn), interpolation='nearest',cmap='gray')
    plt.gca().set_axis_off()
    plt.colorbar(image)

def indexJKLtoSubplots(tensorIn):
    nsmpl, ax1, ax2=tensorIn.shape
    boxSize=np.ceil(np.sqrt(nsmpl))
    figHndl, axArr=plt.subplots(ax1-1, ax2-1, sharex=True, sharey=True)
    # numpy documentation, as always, SUCKS. See: http://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib/8391452#8391452
    # one directional colormapping
    # cmap=plt.get_cmap('gray')
    # cNorm=colors.Normalize(vmin=np.min(tensorIn,axis=None), vmax=np.max(tensorIn, axis=None))
    # scalarMap=mplcm.ScalarMappable(norm=cNorm, cmap=cmap)

    # diverging colormapping, symetrized
    dataMin=np.min(tensorIn,axis=None)
    dataMax=np.max(tensorIn,axis=None)
    negFloor=min((dataMin, -dataMax))
    posCeil=max((dataMax, -dataMin))
    cmap=plt.get_cmap('bwr')
    cNorm=colors.Normalize(vmin=negFloor, vmax=posCeil)
    scalarMap=mplcm.ScalarMappable(norm=cNorm, cmap=cmap)

    accum=[]
    for i in range(0, ax1):
        for j in range(i+1, ax2):
            imagePlot=np.concatenate((tensorIn[:,i,j].flatten(), np.zeros(boxSize**2-nsmpl))).reshape((boxSize,boxSize))
            # http://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
            mplsux=axArr[i,j-1].imshow(imagePlot, interpolation='none', cmap=cmap)
            # mplsux=axArr[i,j-1].imshow(scalarMap.to_rgba(imagePlot), interpolation='none', cmap=cmap)
            axArr[i,j-1].set_aspect('auto')
            axArr[i,j-1].set_adjustable('box-forced')
            axArr[i,j-1].set_axis_off()

    for i in range(0,ax1-1):
        for j in range(0,i):
            axArr[i,j].set_axis_off()

    # Matplotlib. Gauaranteed not to work, not to be intuitive and not to be correctly documented.
    # see: http://stackoverflow.com/questions/28801803/matplotlib-scalarmappable-why-need-to-set-array-if-norm-set
    # http://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    # figHndl.colorbar(mplsux,ax=axArr.ravel().tolist()) # fails to update to match colorbar choice
    # actually, see stackoverflow, questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    scalarMap._A=[]
    plt.colorbar(scalarMap, ax=axArr.ravel().tolist())


if __name__=="__main__":
    points=concaveHypersphere(100)
    # toPlot=TradeoffAnalyzer(points).indicatorCovar()
    # toOrder=tI.conditionalCovar(points)
    toOrder=TradeoffAnalyzer(points).indicatorCovar()
    toPlot=tI.conditionalCovar(points)

    for k in range(toPlot.shape[1]):
        for l in range(toPlot.shape[2]):
            toPlot[:,k,l]=toPlot[np.argsort(toOrder[:,k,l]),k,l]

    # indexJKLtoImage(toPlot)
    # plt.show()

    plt.figure()
    indexJKLtoSubplots(toPlot)
    plt.show()