# coding: utf-8
import numpy as np

# In[]:
def F(x1, x2):
    #return (x1**3 - 3 * x1 * x2**2 ) / 200
    #return 1 / (1 + x1**2 * x2**2)
    return np.exp(-0.5 * (x1**2 + x2**2) / 2)

# In[]:
# TRAINING
from lolimot import LolimotRegressor

k = 250 # Grid führt zu k*k Samples
x1, x2 = np.meshgrid(np.linspace(-5, 5, k), np.linspace(-5, 5, k))

y = np.reshape(F(x1, x2), (k*k, ))
X = np.vstack((x1.flatten(), x2.flatten())).T

lolimot = LolimotRegressor(smoothing='proportional', model_complexity=100)

_ = lolimot.fit(X=X, y=y)

# In[]:
# PREDICTION

k = 50 # Grid führt zu k*k Samples
x1, x2 = np.meshgrid(np.linspace(-5, 5, k), np.linspace(-5, 5, k))

y = np.reshape(F(x1, x2), (k*k, ))
X_test = np.vstack((x1.flatten(), x2.flatten())).T

Z = np.reshape(lolimot.predict(X_test), (k,k))