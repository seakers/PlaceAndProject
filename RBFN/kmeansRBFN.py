from RBFN.RBFN import RBFN
import sklearn.cluster as sklc
import numpy as np


class kmeansRBFN(RBFN):
    def __init__(self, hidden_shape, sigma=1.0, constantTerm=False):
        super().__init__(hidden_shape, sigma=sigma, constantTerm=constantTerm)

    def _select_centers(self, X):
        if len(X.shape)>1:
            km=sklc.KMeans(n_clusters=self.hidden_shape)
            km.fit(X)
            return km.cluster_centers_
        else:
            return np.linspace(X.min(), X.max(), self.hidden_shape)
