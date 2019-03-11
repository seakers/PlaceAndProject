"""
based on:
File: RBFN.py
Author: Octavio Arriaga
Email: arriaga.camargo@email.com
Github: https://github.com/oarriaga
Description: Minimal implementation of a radial basis function network

but it's rather poor and rudimentary, so made lots of mods
"""

import numpy as np
import scipy as sp
import numpy.linalg as npl
from common import *

class RBFN(object):
    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _vectorized_exponential(self, radial):
        return np.exp(-radial/self.sigma)

    def _kernel_function(self, center, data_point):
        return np.exp(-np.linalg.norm(center-data_point)**2/self.sigma)

    def _vectorized_interoplation_matrix(self,X):
        if len(X.shape)==1:
            locs=X[:,np.newaxis]
            cents=self.centers[:,np.newaxis]
        else:
            locs=X
            cents=self.centers
        return self._vectorized_exponential(sp.spatial.distance.cdist(locs, cents))

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((X.shape[0], self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.sort(np.random.choice(X.shape[0], self.hidden_shape, replace=False)) #really bad idea to be honest
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
                d[i,j]=(tX[j,:]-self.centers[i,:])/self.sigma
        return 2*np.dot(Gw, d)

    def hessian(self,X):
        """
        calculates 2nd derivative. ASSUMES NETWORK IS SINGLE-LAYER RBF SUM.
        :param X: locations to take derivative at (#points x dimensionality)
        :return: gradient vector in components of X
        """
        raise NotImplementedError

class RBFNwithParamTune(RBFN):
    def __init__(self):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        super(RBFN, self).__init__(0, sigma=1.0)

    def fit(self, X, Y):
        super(RBFN, self).fit(X,Y)

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