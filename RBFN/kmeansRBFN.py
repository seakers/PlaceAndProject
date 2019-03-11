from RBFN.RBFN import RBFN
import sklearn.cluster as sklc


class kmeansRBFN(RBFN):
    def __init__(self, hidden_shape, numCenters, sigma=1.0):
        super().__init__(hidden_shape, sigma=sigma)
        self.numCenters=numCenters

    def _select_centers(self, X):
        km=sklc.kmeans(n_clusters=self.numCenters)
        km.fit(X)
        return km.cluster_centers_