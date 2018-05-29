#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import numpy.testing as npt
from lolimot import LolimotRegressor


class LolimotTest:
    
    def setUp(self):
        self.lolimot = LolimotRegressor()
        self.Theta = None
        
    def test_update_validity_functions(self):
        X = self.lolimot.X
        Xi = self.lolimot.Xi
        
        mu = np.ones((self.lolimot.M_, self.lolimot.k))
        for k in range(self.lolimot.k):
            for m in range(self.lolimot.M_):
                mu_m = [(X[k, n] - Xi[n, m])**2 / self.lolimot.Sigma[n, m]**2
                        for n in range(self.lolimot.N)]  
                mu[m, k] = np.exp(-0.5*np.sum(mu_m))
        
        mu_sum = np.sum(mu, axis=0)
        A = np.array(mu) / mu_sum
        
        self.lolimot._update_validity_functions() 
        npt.assert_allclose(A, self.lolimot.A)
        
    def test_get_theta(self):
        npt.assert_allclose(
                self.lolimot._get_theta((0, )).flatten(),
                self.Theta, atol=1e-14)


class LolimotTest1D(LolimotTest, unittest.TestCase):

    def setUp(self):
        self.lolimot = LolimotRegressor()
        self.lolimot.M_ = 1
        self.lolimot.N = 1
        self.lolimot.k = 2
        self.lolimot.X = np.array([[1], [3]])
        self.lolimot.y = np.array([1, 2])
        self.lolimot.Xi = np.array([[2]])
        self.lolimot.Sigma = np.ones_like(self.lolimot.Xi) * 0.5
        self.lolimot.A = np.ones((self.lolimot.M_, self.lolimot.k))
        
        # expected results
        self.Theta = [0.5, 0.5]


class LolimotTest2D(LolimotTest, unittest.TestCase):
    
    def setUp(self):
        self.lolimot = LolimotRegressor()
        self.lolimot.M_ = 3
        self.lolimot.N = 2
        self.lolimot.k = 3
        self.lolimot.X = np.array([[1, 1], [1, 3], [3, 3]])
        self.lolimot.y = np.array([1 ,2, 5])
        self.lolimot.Xi = np.array([[1, 3, 2], [1, 1, 3]])
        self.lolimot.Sigma = np.ones_like(self.lolimot.Xi) * 0.5
        self.lolimot.A = np.ones((self.lolimot.M_, self.lolimot.k))
        
        self.Theta = [-1.0, 1.5, 0.5]
        
        
if __name__ == "__main__":
    unittest.main(verbosity=2)
        
    
        
    

