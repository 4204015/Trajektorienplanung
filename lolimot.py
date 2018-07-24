import time
import tqdm
import pickle
import numpy as np
import numexpr as ne
import scipy.sparse as sps
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV


class Cache:
    pass


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

    model_complexity : int, default: 100
        Maximum number of local models.

    limit_sigma : bool, default: True
        Prevents sigma from getting smaller than 10e-18.

    tol : float, default:10e-5
        Tolerance of the global loss (training loss) for stopping criterion.

    notebook : bool, default: True
        Enables a good-looking progressbar when using in an interactive environment.

    refinement : bool, default: True
        If 'True' the worst model will be selected for further refinement in each pass.
        Otherwise a model is randomly chosen in consideration of its loss,
        which is interpreted as the probability to be selected.

    data_weighting : bool, default: False
        Enables the weighting of the data anti-proportional
        to the data density in order to compensate for the data distribution.

    input_range : list of tuples, optional
        Range of values for each input dimension N for which the model should be trained.
        If not passed, it will be determined when fitting to data.

    output_constrains: tuple, optional
        A tuple with the lower and the upper bound of the output. Is involved when splitting models.

    validation_set : list, optional
        Data set to estimate the generalization ability of the model during training.
        Should be like [X_val, y_val].

    plotter : object, optional
        Plotter object for live visualisation of the training process.
    """

    def __init__(self, sigma=0.4, smoothing='proportional', p=1/3, model_complexity=100, limit_sigma=True,
                 tol=10e-5, notebook=True, refinement='loser', data_weighting=False, input_range=None,
                 output_constrains=None, validation_set=None, plotter=None):

        # TODO: move parameter validation to 'fit' (scikit api)
        if input_range:
            assert isinstance(input_range, list), "'input_range' has to be a list of range tuples."
            self.N = len(input_range)
            self.x_range = input_range
        else:
            self.x_range = []
            self.N = None

        # Stopping criterions
        self.model_complexity = model_complexity
        self.tol = tol

        # Free parameter/options for the algorithm
        self.smoothing = smoothing  # string
        self.sigma = sigma
        self.p = p
        self.output_constrains = output_constrains
        self.data_weighting = data_weighting

        self.refinement = refinement  # bool
        self.limit_sigma = limit_sigma  # bool

        # Parameters of the models
        self.M_ = None          # Number of models
        self.model_range = []   # M x N
        self.Theta = None       # M x N+1
        self.A = None           # M x k
        self.Xi = None          # N x M
        self.Sigma = None       # N x M

        # Training and validation data
        self.X = None           # k x N
        self.y = None           # k x _
        self.k = None           # Number of samples

        self.y_hat = None       # k x _ - result after training
        self.X_train = None

        self.X_val, self.y_val = check_X_y(*validation_set) if validation_set else [None, None]

        # Tracking of the learning process / Visualisation
        self.global_loss = []
        self.validation_loss = []
        self.split_duration = []
        self.training_duration = 0
        self.plotter = plotter  # plotter object
        self.notebook = notebook  # bool

        self.cache = Cache()
        self.cache.online_prediction = None
        self.cache.last_M = 0

        # Numpy settings
        np.seterr(divide='raise')  # Division by zero leads to an exception instead of a warning
        np.random.seed(42)

    # --- private functions ---
    def _get_theta(self, model_pointers=None):
        if not model_pointers:
            model_pointers = range(self.M_)
            num_model = self.M_
        else:
            num_model = len(model_pointers)
        Theta = np.zeros((num_model, self.N + 1))
        for i, m in enumerate(model_pointers):  # for model m
            Q_m = sps.spdiags(self.A[m, :], diags=0, m=self.k, n=self.k)
            X_reg = np.hstack((np.ones((self.k, 1)), self.X))  # regression matrix
            Theta[i, :] = np.linalg.lstsq(Q_m @ self.R @ X_reg, self.y @ self.R @ Q_m, rcond=None)[0].flatten()

        return Theta

    def _update_validity_functions(self):
        c = np.zeros((self.M_, self.k))
        for m in range(self.M_):
            np.sum((self.X - self.Xi.T[m, :]) ** 2 / (self.Sigma.T[m, :] ** 2), out=c[m, :], axis=1)
        mu = ne.evaluate('exp(-0.5 * c)')
        mu_sum = np.sum(mu, axis=0)  # summation along M-axis -> k
        np.divide(mu, mu_sum, out=self.A) 
        
    def _increase_model_complexity(self, increment=1):
        for _ in range(increment):
            self.M_ += 1
            self.Theta = np.vstack((self.Theta, np.zeros((1, self.N + 1))))
            self.A = np.vstack((self.A, np.zeros((1, self.k))))
            self.Xi = np.hstack((self.Xi, np.zeros((self.N, 1))))
            self.Sigma = np.hstack((self.Sigma, np.zeros((self.N, 1))))
            self.model_range.append([() for _ in range(self.N)])

    def _decrease_model_complexity(self, decrement=1):
        for _ in range(decrement):
            self.M_ -= 1
            self.Theta = self.Theta[0:self.M_, :]
            self.A = self.A[0:self.M_, :]
            self.Xi = self.Xi[:, 0:self.M_]
            self.Sigma = self.Sigma[:, 0:self.M_]
            self.model_range.pop()

    def _get_sigma(self, ranges):
        if self.smoothing == 'const':
            return self.sigma
        elif self.smoothing == 'proportional':
            new_sigmas = np.array(list(map(lambda r: np.abs(np.subtract(*r)) * self.p, ranges)))
            if self.limit_sigma:
                new_sigmas[new_sigmas < 10e-18] = 10e-18
                return new_sigmas.tolist()
            else:
                return new_sigmas.tolist()
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

    def _predict_during_training(self, X):
        if self.cache.last_M == self.M_:
            return self.cache.online_prediction, self.cache.A
        else:
            self.cache.last_M = self.M_

            X_prev = np.copy(self.X)
            k_prev = np.copy(self.k)
            A_prev = np.copy(self.A)

            y = self.predict(X)
            self.cache.online_prediction = y
            self.cache.A = np.copy(self.A)

            self.X = np.copy(X_prev)
            self.k = np.copy(k_prev)
            self.A = np.copy(A_prev)
        return y, self.cache.A

    def _get_validation_loss(self):
        self.validation_loss.append(np.sum((self.y_val - self._predict_during_training(self.X_val)[0]) ** 2))
        return self.validation_loss[-1]

    def _get_model_volumes(self, idx=-1):
        volumes = []
        if idx == -1:
            for r in self.model_range:
                volumes.append(np.prod([np.abs(np.subtract(*r_n)) for r_n in r]))
            return volumes
        else:
            return np.prod([np.abs(np.subtract(*r_n)) for r_n in self.model_range[idx]])

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

    def _get_model_idx_for_refinement(self):
        if self.refinement == 'loser' and self.output_constrains is None:  # loser refinement
            return np.argmax(self._get_local_loss())

        elif self.refinement == 'limited':  # lower limit for model volume
            idxs = np.flip(np.argsort(self._get_local_loss()), axis=0)
            i = 0
            while not self._get_model_volumes(idxs[i]) > 1e-10:
                i += 1
                continue
            return idxs[i]

        elif self.refinement == 'probability':  # probability approach
            if self.M_ > 1:
                loss = self._get_local_loss()
                scaler = MinMaxScaler(feature_range=(np.min(loss), np.max(loss)))
                volumes = self._get_model_volumes()
                p = loss + np.log10((1 - scaler.fit_transform([volumes]))[0])
            else:
                p = self._get_local_loss()

            return np.random.choice(list(range(self.M_)), p=p/np.sum(p))

        elif self.output_constrains is not None:  # constrained output
            y, A = self._predict_during_training(self.X_val)
            constrain_violation = A @ (np.logical_or(
                (y < self.output_constrains[0]), (y > self.output_constrains[1])))
            return np.argmax(constrain_violation + np.multiply(self._get_local_loss(), 0.01))

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
        self.model_range.append(deepcopy(self.x_range))
        self.global_loss.append(self._get_global_loss())
        
        tqdm.tqdm.monitor_interval = 0  # disable the monitor thread because bug in tqdm #481
        if self.notebook:
            pbar = tqdm.tqdm_notebook(total=self.model_complexity)
        else:
            pbar = tqdm.tqdm(total=self.model_complexity)
        pbar.update(1)
        while (self.M_ < self.model_complexity) and not (self.tol > self.global_loss[-1]):
            start_time = time.time()
            # 2. Find worst LLM
            l = self._get_model_idx_for_refinement()  # the model denoted by 'l' is considered for further refinement
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
                self._decrease_model_complexity()
                print(f"[WARNING]: Training was aborted because of singular matrix with M={self.M_}")
                break
            except FloatingPointError:
                self._decrease_model_complexity()
                print(f"[WARNING]: Training was aborted because Sigma values a too small. M={self.M_}")
                break

            if self.plotter:
                plot_data = {}
                plot_data.update({'training_loss': L_global[j]})

                if not self.M_ % 5:
                    if self.X_val is not None:
                        plot_data.update({'validation_loss': self._get_validation_loss()})

                    y = []
                    lower = []
                    upper = []

                    for mr in self.model_range:
                        for r_n in mr:
                            ranges = [np.abs(np.subtract(*r_n))]
                            y.append(np.mean(ranges))
                            lower.append(np.min(ranges))
                            upper.append(np.max(ranges))

                    ranges_data = {'lower': np.min(lower), 'upper': np.max(upper), 'y': np.mean(y)}
                    plot_data.update({'model_ranges': ranges_data})

                self.plotter.update(plot_data)

        self.local_loss = self._get_local_loss()

        if self.tol > self.global_loss[-1]:
            print(f"[INFO]: Training finished because global loss is smaller than tol:={self.tol}.")

        pbar.close()
        self.X_train = self.X
        self.y_hat = self._get_model_output(self.X_train)
        
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

        if self.data_weighting:
            grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 10)}, cv=3)
            grid.fit(X)
            self.kde = KernelDensity(bandwidth=0.35, kernel='gaussian').fit(X)
            rep_dens = np.reciprocal(np.exp(self.kde.score_samples(X)))
            self.R = sps.spdiags(rep_dens/np.max(rep_dens), diags=0, m=self.k, n=self.k)
        else:
            self.R = np.identity(self.k)

        # --- model fitting ---
        self._construct_component_models()
        # ----------------------
        
        self.training_duration = time.time() - start_time
        # print(f"[INFO] Finished model training after {time.time() - start_time:.4f} seconds.")
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
        elif self.N == 2:
            for m, m_range in enumerate(self.model_range):
                u1 = np.linspace(*m_range[0], 2)
                u2 = np.linspace(*m_range[1], 2)
                u1, u2 = np.meshgrid(u1, u2)
                y = self.Theta[m, 2] * u2 + self.Theta[m, 1] * u1 + self.Theta[m, 0]
                c = self.Theta[m, 2] * self.Xi[1, m] + self.Theta[m, 1] * self.Xi[0, m] + self.Theta[m, 0]
                yield [u1, u2], np.reshape(y, (2, 2)), c                     
        else:
            print("Local models only available for N <= 2")
        
    def save(self, filename='lolimot.obj'):
        self.plotter = None
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename='lolimot.obj'):
        with open(filename, 'rb') as input_file:
            return pickle.load(input_file)

if __name__ == "__main__":
    from test import LolimotTest1D, LolimotTest2D
    import unittest
    unittest.main(verbosity=2)