import numpy as np
from scipy.interpolate import Rbf
x, y, z, d = np.random.rand(4, 50)
print(x)
print(y)
print(z)
print(d)
rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
xi = yi = zi = np.linspace(0, 1, 20)
di = rbfi(xi, yi, zi)   # interpolated values
di.shape