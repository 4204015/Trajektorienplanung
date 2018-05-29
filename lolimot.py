import time
import tqdm
import numpy as np
import numexpr as ne
import scipy as sp
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LolimotRegressor(BaseEstimator, RegressorMixin):
    """
    Regression using Local (linear) models trained with the LOLIMOT algorithm.

    Parameters
    ----------
    sigma : float, default: 0.4
        Used only 'sigma_option' is set to 'const', to specify a constant standard deviation
        of the normalized Gaussian validity functions.

    smoothing : string, 'const' or 'proportional', default: 'proportional'
        Specifies whether a constant standard deviation is used, or is calculated proportional
        to the extension of the hyperrectangle of each local model.

    p : float, default: 1/3
        Proportionality factor between the rectangles' extension and the standard deviations.

    model_complexity : int, default: 10
        Maximum number of local models.

    x_range : list of tuples, optional, default: []
        Range of values for each input dimension N for which the model should be trained.
    """

    def __init__(self, sigma=0.4, smoothing='proportional', p=1/3, model_complexity=10, x_range=None, notebook=True):

        # TODO: move parameter validation to 'fit' (scikit api)
        if x_range:
            assert isinstance(x_range, list), "'x_range' has to be a list of range tuples."
            self.N = len(x_range)
            self.x_range = x_range
        else:
            self.x_range = []
            self.N = None

        self.sigma = sigma
        self.model_complexity = model_complexity

        self.smoothing = smoothing
        self.p = p
        self.notebook = notebook

        self.M_ = None          # Number of models
        self.model_range = []   # M x N
        self.Theta = None       # M x N+1
        self.A = None           # M x k
        self.Xi = None          # N x M
        self.Sigma = None       # N x M
        self.X = None           # k x N
        self.y = None           # k x _
        self.k = None           # Number of samples
        
        self.global_loss = []
        self.split_duration = []
        self.training_duration = 0

    # --- private functions ---
    def _get_theta(self, model_pointers=None):
        if not model_pointers:
            model_pointers = range(self.M_)
            num_model = self.M_
        else:
            num_model = len(model_pointers)
        Theta = np.zeros((num_model, self.N + 1))
        for i, m in enumerate(model_pointers):  # for model m
            #Q_m = np.diag(self.A[m, :])  # weight matrix Q
            Q_m = sp.sparse.spdiags(self.A[m, :], 0, self.A[m, :].size, self.A[m, :].size)
            X_reg = np.hstack((np.ones((self.k, 1)), self.X))  # regression matrix
            try:
                Theta[i, :] = np.linalg.lstsq(Q_m @ X_reg, self.y @ Q_m, rcond=None)[0].flatten()
                
                # old (explicit) implementation:
                #Theta[i, :] = (np.linalg.inv(X_reg.T @ Q_m @ X_reg) @ X_reg.T @ Q_m @ self.y).flatten()
            except np.linalg.LinAlgError as err:
                print(f"[WARNING]: Training was aborted because of singular matrix with M={self.M_}")
                raise
        return Theta
    
    def _update_validity_functions(self):
        c = np.zeros((self.M_, self.k))
        for m in range (self.M_):
            np.sum((self.X - self.Xi.T[m, :]) ** 2 / (self.Sigma.T[m, :] ** 2), out=c[m, :], axis=1)
        mu = ne.evaluate('exp(-0.5 * c)')
        mu_sum = np.sum(mu, axis=0) # summation along M-axis -> k
        np.divide(mu, mu_sum, out=self.A) 
        
    def _increase_model_complexity(self, increment=1):
        for _ in range(increment):
            self.M_ += 1
            self.Theta = np.vstack((self.Theta, np.zeros((1, self.N + 1))))
            self.A = np.vstack((self.A, np.zeros((1, self.k))))
            self.Xi = np.hstack((self.Xi, np.zeros((self.N, 1))))
            self.Sigma = np.hstack((self.Sigma, np.zeros((self.N, 1))))
            self.model_range.append([() for _ in range(self.N)])

    def _get_sigma(self, ranges):
        if self.smoothing == 'const':
            return self.sigma
        elif self.smoothing == 'proportional':
            return list(map(lambda r: np.abs(np.subtract(*r)) * self.p, ranges))
        else:
            raise ValueError(f"Inadmissible smoothing parameter: '{self.smoothing}'")          

    def _save_params(self):
        self.model_range_prev = deepcopy(self.model_range)
        self.Xi_prev = np.copy(self.Xi)
        self.Sigma_prev = np.copy(self.Sigma)
        self.Theta_prev = np.copy(self.Theta)

    def _recover_params(self):
        self.model_range = deepcopy(self.model_range_prev)
        self.Xi = np.copy(self.Xi_prev)
        self.Sigma = np.copy(self.Sigma_prev)
        self.Theta = np.copy(self.Theta_prev)

    def _get_local_loss(self):
        return self.A @ (self.y - self._get_model_output(self.X)) ** 2  # Loss function output -> M x _

    def _get_global_loss(self):
        return np.sum((self.y - self._get_model_output(self.X)) ** 2)
    
    def _split_along(self, j, l, m):
        # (a) ... split component model along j in two halves
        self.model_range[m] = deepcopy(self.model_range[l])
        r = self.model_range[l][j]
        ranges = [(np.min(r), np.mean(r)), (np.mean(r), np.max(r))]
        self.model_range[l][j], self.model_range[m][j] = ranges
                
        self.Xi[:, m] = deepcopy(self.Xi[:, l])   
        self.Xi[j, (l, m)] = list(map(lambda x: np.mean(x), ranges))
        
        self.Sigma[:, m] = deepcopy(self.Sigma[:, l])
        self.Sigma[j, (l, m)] = self._get_sigma(ranges)

        # (b) ... calculate validity functions all models
        self._update_validity_functions()

        # (c) ... get models' parameter
        self.Theta[(l, m), :] = self._get_theta((l, m))
        
    def _construct_component_models(self):
        # 1. Initialize global model
        self.M_ = 1
        self.Xi = np.zeros((self.N, self.M_))
        self.Sigma = np.zeros((self.N, self.M_))

        self.Xi[:, 0] = [np.mean(r) for r in self.x_range]
        self.Sigma[:, 0] = self._get_sigma(self.x_range)
        self.A = np.zeros((self.M_, self.k))
        self._update_validity_functions()
        self.Theta = self._get_theta((0,))
        self.model_range.append(self.x_range)
        self.global_loss.append(self._get_global_loss())
        
        tqdm.tqdm.monitor_interval = 0 # disable the monitor thread because bug in tqdm #481
        if self.notebook:
            pbar = tqdm.tqdm_notebook(total=self.model_complexity) # _tqdm_notebook
        else:
            pbar = tqdm.tqdm(total=self.model_complexity) # _tqdm_notebook
        pbar.update(1)
        while self.M_ < self.model_complexity:
            start_time = time.time()
            # 2. Find worst LLM
            l = np.argmax(self._get_local_loss())  # the model denoted by 'l' is considered for further refinement
            self._increase_model_complexity()
            m = self.M_ - 1  # denotes the most recent added model

            L_global = np.zeros(self.N)  # global model loss for every split attempt
            self._save_params()
            
            try:
                # 3. for every input dimension ...
                for j in range(self.N):
                    self._split_along(j, l, m)
    
                    # (d) ... calculate the tree's output error
                    L_global[j] = self._get_global_loss()
    
                    # Undo changes 'from _split_along'
                    self._recover_params()
    
                # 4. find best division (split) and apply
                j = np.argmin(L_global)
                self.global_loss.append(L_global[j])
                self._split_along(j, l, m)
                self.split_duration.append(time.time() - start_time)
                pbar.update(1)
            except np.linalg.LinAlgError:
                self.M_ -= 1  
                break  
        pbar.close()
        
    def _get_model_output(self, u):
        U = np.hstack((np.ones((self.k, 1)), u)).T
        y_hat = np.sum((self.Theta @ U) * self.A, axis=0)
        return y_hat

    # --- public functions ---

    def fit(self, X, y):
        """Fit the model according to the given training data."""

        X, y = check_X_y(X, y)

        start_time = time.time()

        # --- initialising model parameter ---
        self.X = X
        self.y = y
        self.k, N, *_ = X.shape

        if not self.N:
            self.N = N
        else:
            assert self.N == N, \
                f"Dimension N from 'x_range' input and 'X' does not agree: {self.N} ≠ {N}"

        if not self.x_range:
            for j in range(self.N):
                self.x_range.append((X[:, j].min(), X[:, j].max()))

        # --- modell fitting ---
        self._construct_component_models()
        # ----------------------
        
        self.training_duration = time.time() - start_time
        #print(f"[INFO] Finished model training after {time.time() - start_time:.4f} seconds.")
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['M_'])

        # Input validation
        X = check_array(X)

        self.X = X
        self.k = X.shape[0]
        self.A = np.zeros((self.M_, self.k))
        self._update_validity_functions()
        return self._get_model_output(X)

    def local_model_gen(self):
        if self.N == 1:
            for m, m_range in enumerate(self.model_range):
                u = np.linspace(*m_range[0])
                y = self.Theta[m, 1] * u + self.Theta[m, 0]
                c = self.Theta[m, 1] * self.Xi[:, m] + self.Theta[m, 0]
                yield u, y, c
        elif self.N ==2:
            for m, m_range in enumerate(self.model_range):
                u1 = np.linspace(*m_range[0], 2)
                u2 = np.linspace(*m_range[1], 2)
                u1, u2 = np.meshgrid(u1, u2)
                y = self.Theta[m, 2] * u2 + self.Theta[m, 1] * u1 + self.Theta[m, 0]
                c = self.Theta[m, 2] * self.Xi[1, m] + self.Theta[m, 1] * self.Xi[0, m] + self.Theta[m, 0]
                yield [u1, u2], np.reshape(y, (2, 2)), c                     
        else:
            print("Local models only available for N <= 2")
        
    def save_model(self):
        pass

    @staticmethod
    def load_model():
        pass

if __name__ == "__main__":
    from test import LolimotTest1D, LolimotTest2D
    import unittest
    unittest.main(verbosity=2)