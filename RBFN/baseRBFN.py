"""
based on:
File: baseRBFN.py
Author: Octavio Arriaga
Email: arriaga.camargo@email.com
Github: https://github.com/oarriaga
Description: Minimal implementation of a radial basis function network

but it's rather poor and rudimentary, so made lots of mods
"""

import scipy as sp
from Common.common import *

class RBFN(object):
    def __init__(self, hidden_shape, sigma=1.0, constantTerm=False):
        """ radial basis function network
        # Arguments
            hidden_shape: number of hidden radial basis functions. Also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None
        self.constant=0
        self.activeConstant=constantTerm

    def _vectorized_exponential(self, radial):
        return np.exp(-radial**2/self.sigma**2)

    # OBSOLETE: use vectorized exponential instead
    # def _kernel_function(self, center, data_point):
    #     return np.exp(-np.linalg.norm(center-data_point)**2/self.sigma)

    def _vectorized_interoplation_matrix(self,X):
        if len(X.shape)==1:
            locs=X[:,np.newaxis]
            if len(self.centers.shape)==1:
                cents=self.centers[:,np.newaxis]
            else:
                cents=self.centers
        else:
            locs=X
            cents=self.centers
        expMat = self._vectorized_exponential(sp.spatial.distance.cdist(locs, cents))
        if self.activeConstant:
            big=np.concatenate((np.ones((expMat.shape[0], 1)),expMat), axis=1)
        else:
            big=expMat
        return big

    # def _calculate_interpolation_matrix(self, X):
    #     """ Calculates interpolation matrix using a kernel_function
    #     # Arguments
    #         X: Training data
    #     # Input shape
    #         (num_data_samples, input_shape)
    #     # Returns
    #         G: Interpolation matrix
    #     OBSOLETE. Taken from original author. But this code sucks. use _vectorized_interpolation_matrix instead
    #     """
    #     G = np.zeros((X.shape[0], self.hidden_shape))
    #     for data_point_arg, data_point in enumerate(X):
    #         for center_arg, center in enumerate(self.centers):
    #             G[data_point_arg, center_arg] = self._kernel_function(
    #                     center, data_point)
    #     return G

    def _select_centers(self, X):
        random_args = np.sort(np.random.choice(X.shape[0], int(self.hidden_shape), replace=False)) #really bad idea to be honest
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._vectorized_interoplation_matrix(X)
        self.weights, _, _, _= np.linalg.lstsq(G,Y, rcond=None)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        tX=numpyze(X)
        G = self._vectorized_interoplation_matrix(tX)
        predictions = np.dot(G, self.weights)
        return predictions

    def derivative(self,X):
        """
        calculates derivative. ASSUMES NETWORK IS SINGLE-LAYER RBF SUM.
        :param X: locations to take derivative at (#points x dimensionality)
        :return: gradient vector in components of X. (#pointsInput x dimensionality)
        """
        tX=numpyze(X)
        G= self._vectorized_interoplation_matrix(tX)
        Gw=G*self.weights[np.newaxis,:]
        d=np.zeros((Gw.shape[0], self.centers.shape[1]))
        if len(tX.shape)==1:
            tX=tX.reshape((1,len(tX)))
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                d[i,j]=(tX[j,:]-self.centers[i,:])/self.sigma**2
        return 2*np.dot(Gw, d)

    def hessian(self,X):
        """
        calculates 2nd derivative. ASSUMES NETWORK IS SINGLE-LAYER RBF SUM.
        :param X: locations to take derivative at (#points x dimensionality)
        :return: gradient vector in components of X
        """
        raise NotImplementedError

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    model = RBFN(hidden_shape=10, sigma=1.)
    model.fit(x, y)
    y_pred = model.predict(x)
    
    plt.plot(x, y, 'b-', label='real')
    plt.plot(x, y_pred, 'r-', label='fit')
    plt.legend(loc='upper right')
    plt.title('Interpolation using a RBFN')
    plt.show()