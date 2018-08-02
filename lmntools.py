import time
import tqdm
import pickle
import numpy as np
import numexpr as ne
import scipy.sparse as sps
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import train_test_split


class Cache:
    def __init__(self, *args):
        for arg in args:
            setattr(self, arg, None)

    def clear(self):
        for key in self.__dict__.keys():
            self.__dict__[key] = None


class LMNRegressor(BaseEstimator, RegressorMixin):
    """
    Regression using a Local Model Network.

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

    training_tol : float, default: 1e-4
        Tolerance of the global loss (training loss) for stopping.

    early_stopping_tol : float, default: 1e+0
        Tolerance of the validation loss for stopping.

    notebook : bool, default: True
        Enables a good-looking progressbar when using in an interactive environment.

    refinement : bool, default: True
        If 'True' the worst model will be selected for further refinement in each pass.
        Otherwise a model is randomly chosen in consideration of its loss,
        which is interpreted as the probability to be selected.

    kde_bandwidth : float, int, default: 0
        Enables the weighting of the data anti-proportional
        to the data density in order to compensate for the data distribution.

    validation_size : float, int, or None, default None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
        to include in the validation split. If int, represents the absolute number of train samples.
        If None, no validation set will be split of.

    random_state : RandomState or an int seed, optional
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    plotter : object, optional
        Plotter object for live visualisation of the training process.
    """

    def __init__(self, sigma=0.4, smoothing='proportional', p=1/3, model_complexity=5, limit_sigma=True,
                 training_tol=1e-4, early_stopping=False, early_stopping_tol=0.075, refinement='loser',
                 kde_bandwidth=0, validation_size=None, random_state=None, notebook=True, plotter=None):

        self.random_state = random_state

        # Stopping criterions
        self.model_complexity = model_complexity  # int
        self.training_tol = training_tol  # float
        self.early_stopping = early_stopping  # bool
        self.early_stopping_tol = early_stopping_tol  # float

        # Free parameter/options for the algorithm
        self.smoothing = smoothing  # string
        self.sigma = sigma  # float
        self.p = p  # float
        self.kde_bandwidth = kde_bandwidth  # bool
        self.refinement = refinement  # bool
        self.limit_sigma = limit_sigma  # bool

        # Tracking of the learning process / Visualisation
        self.validation_size = validation_size  # bool
        self.plotter = plotter  # plotter object
        self.notebook = notebook  # bool

        self.cache = Cache('model_range_prev', 'Xi_prev', 'Sigma_prev', 'Theta_prev', 'last_M', 'prediction', 'A')

        # Numpy settings
        np.seterr(divide='raise')  # Division by zero leads to an exception instead of a warning

        # --- Attributes which will be set in the 'fit' method ---

        # Tracking of the learning process
        self.global_loss = []
        self.validation_loss = []
        self.split_duration = []

        # Training and validation data
        self.X = None           # k x N_
        self.y = None           # k x _
        self.k = None           # Number of samples
        self.X_val = None
        self.y_val = None

        # Data dependent parameters
        self.A = None           # M x k
        self.output_constrains = None
        self.model_range = []  # M x N_
        self.input_range = []  # N_ x 2
        self.training_duration = 0

        # Estimator attributes
        self.M_ = None      # Number of models
        self.Theta_ = None  # M x N+1
        self.Xi_ = None     # N x M
        self.Sigma_ = None  # N x M
        self.N_ = None      # k
        self.kde_ = None
        self.R_ = None

        self.random_state_ = None

    # --- properties ---
    @property
    def M(self):
        return self.M_

    @property
    def N(self):
        return self.N_

    @property
    def Theta(self):
        return self.Theta_

    @property
    def Xi(self):
        return self.Xi_

    @property
    def y_hat(self):
        return self.predict(self.X)

    @property
    def local_loss(self):
        return self._get_local_loss()

    # --- private functions ---
    def _get_theta(self, model_pointers=None):
        if not model_pointers:
            model_pointers = range(self.M_)
            num_model = self.M_
        else:
            num_model = len(model_pointers)
        Theta = np.zeros((num_model, self.N_ + 1))
        for i, m in enumerate(model_pointers):  # for model m
            Q_m = sps.spdiags(self.A[m, :], diags=0, m=self.k, n=self.k)
            X_reg = np.hstack((np.ones((self.k, 1)), self.X))  # regression matrix
            Theta[i, :] = np.linalg.lstsq(Q_m @ self.R_ @ X_reg, self.y @ self.R_ @ Q_m, rcond=None)[0].flatten()

        return Theta

    def _update_validity_functions(self, X, A):
        c = np.zeros(A.shape)
        for m in range(self.M_):
            np.sum((X - self.Xi_.T[m, :]) ** 2 / (self.Sigma_.T[m, :] ** 2), out=c[m, :], axis=1)
        mu = ne.evaluate('exp(-0.5 * c)')
        mu_sum = np.sum(mu, axis=0)  # summation along M-axis -> k
        np.divide(mu, mu_sum, out=A)
        
    def _increase_model_complexity(self, increment=1):
        for _ in range(increment):
            self.M_ += 1
            self.Theta_ = np.vstack((self.Theta_, np.zeros((1, self.N_ + 1))))
            self.A = np.vstack((self.A, np.zeros((1, self.k))))
            self.Xi_ = np.hstack((self.Xi_, np.zeros((self.N_, 1))))
            self.Sigma_ = np.hstack((self.Sigma_, np.zeros((self.N_, 1))))
            self.model_range.append([() for _ in range(self.N_)])

    def _decrease_model_complexity(self, decrement=1):
        for _ in range(decrement):
            self.M_ -= 1
            self.Theta_ = self.Theta_[0:self.M_, :]
            self.A = self.A[0:self.M_, :]
            self.Xi_ = self.Xi_[:, 0:self.M_]
            self.Sigma_ = self.Sigma_[:, 0:self.M_]
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
        self.cache.model_range_prev = deepcopy(self.model_range)
        self.cache.Xi_prev = np.copy(self.Xi_)
        self.cache.Sigma_prev = np.copy(self.Sigma_)
        self.cache.Theta_prev = np.copy(self.Theta_)

    def _recover_params(self):
        self.model_range = deepcopy(self.cache.model_range_prev)
        self.Xi_ = np.copy(self.cache.Xi_prev)
        self.Sigma_ = np.copy(self.cache.Sigma_prev)
        self.Theta_ = np.copy(self.cache.Theta_prev)

    def _get_local_loss(self):
        return self.A @ (self.y - self._get_model_output(self.X, self.A)) ** 2  # Loss function output -> M x _

    def _get_global_loss(self):
        return np.sum((self.y - self._get_model_output(self.X, self.A)) ** 2)

    def _predict_cached(self, X):
        if self.cache.last_M == self.M_:
            return self.cache.prediction, self.cache.A
        else:
            self.cache.last_M = self.M_
            y, A = self._predict(X)
            self.cache.prediction, self.cache.A = y, A
        return y, A

    def _get_validation_loss(self):
        self.validation_loss.append(np.sum((self.y_val - self._predict_cached(self.X_val)[0]) ** 2))
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

        self.Xi_[:, m] = deepcopy(self.Xi_[:, l])
        self.Xi_[j, (l, m)] = list(map(lambda x: np.mean(x), ranges))

        self.Sigma_[:, m] = deepcopy(self.Sigma_[:, l])
        self.Sigma_[j, (l, m)] = self._get_sigma(ranges)

        # (b) ... calculate validity functions all models
        self._update_validity_functions(self.X, self.A)

        # (c) ... get models' parameter
        self.Theta_[(l, m), :] = self._get_theta((l, m))

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

            return self.random_state_.choice(list(range(self.M_)), p=p/np.sum(p))

        elif self.output_constrains is not None:  # constrained output
            y, A = self._predict_cached(self.X_val)
            constrain_violation = A @ (np.logical_or(
                (y < self.output_constrains[0]), (y > self.output_constrains[1])))
            return np.argmax(constrain_violation + np.multiply(self._get_local_loss(), 0.01))
        else:
            raise NotImplementedError

    def _stopping_condition_met(self):
        # determine different stopping conditions
        complexity_reached = self.M_ >= self.model_complexity
        tolerance_reached = self.training_tol > self.global_loss[-1]
        if self.early_stopping and self.validation_loss[-4:-1]:
            current_mean_val_loss = np.mean(self.validation_loss[-4:-1])
            overfitting = ((self.validation_loss[-1] - current_mean_val_loss) >
                           self.early_stopping_tol * current_mean_val_loss) or\
                          ((self.validation_loss[-1] - np.min(self.validation_loss)) >
                           self.early_stopping_tol * np.min(self.validation_loss) * 1.5)
        else:
            # there are not enough information yet to determine overfitting
            overfitting = False

        # in case of the fulfilment of a stopping condition print the following message
        if complexity_reached:
            print(f"[INFO]: Training finished with a maximal model complexity:={self.model_complexity}.")
        elif tolerance_reached:
            print(f"[INFO]: Training finished because global loss is smaller than tol:={self.training_tol}.")
        elif overfitting and self.early_stopping:
            print(f"[INFO]: Early stopping of the training.")

        return complexity_reached or tolerance_reached or overfitting

    def _construct_component_models(self):
        # 1. Initialize global model
        self.M_ = 1
        self.Xi_ = np.zeros((self.N_, self.M_))
        self.Sigma_ = np.zeros((self.N_, self.M_))

        self.Xi_[:, 0] = [np.mean(r) for r in self.input_range]
        self.Sigma_[:, 0] = self._get_sigma(self.input_range)
        self.A = np.zeros((self.M_, self.k))
        self._update_validity_functions(self.X, self.A)
        self.Theta_ = self._get_theta((0,))
        self.model_range.append(deepcopy(self.input_range))
        self.global_loss.append(self._get_global_loss())
        
        tqdm.tqdm.monitor_interval = 0  # disable the monitor thread because bug in tqdm #481
        if self.notebook:
            pbar = tqdm.tqdm_notebook(total=self.model_complexity)
        else:
            pbar = tqdm.tqdm(total=self.model_complexity)
        pbar.update(1)
        while not self._stopping_condition_met():
            start_time = time.time()
            # 2. Find worst LLM
            r = self._get_model_idx_for_refinement()  # the model denoted by 'r' is considered for further refinement
            self._increase_model_complexity()
            m = self.M_ - 1  # denotes the most recent added model

            L_global = np.zeros(self.N_)  # global model loss for every split attempt
            self._save_params()
            
            try:
                # 3. for every input dimension ...
                for j in range(self.N_):
                    self._split_along(j, r, m)
    
                    # (d) ... calculate the tree's output error
                    L_global[j] = self._get_global_loss()
    
                    # Undo changes 'from _split_along'
                    self._recover_params()
    
                # 4. find best division (split) and apply
                j = np.argmin(L_global)
                self.global_loss.append(L_global[j])

                self._split_along(j, r, m)
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

        pbar.close()
        
    def _get_model_output(self, u, A):
        U = np.hstack((np.ones((A.shape[1], 1)), u)).T
        y_hat = np.sum((self.Theta_ @ U) * A, axis=0)
        return y_hat

    # --- public functions ---

    def fit(self, X, y, input_range=None, output_constrains=None, additional_validation_set=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        input_range : list of tuples, optional
            Range of values for each input dimension N_ for which the model should be trained.
            If not passed, it will be determined when fitting to data.

        output_constrains: tuple, optional
            A tuple with the lower and the upper bound of the output. Is involved when splitting models.

        additional_validation_set: list, optional
            Data set to estimate the generalization ability of the model during training.
            Should be like [X_val, y_val].

        Returns
        -------
        self : returns an instance of self.
        """
        # --- reset trainable attributes ---
        self.M_ = None           # Number of models
        self.Theta_ = None       # M x N_+1
        self.Xi_ = None          # N_ x M
        self.Sigma_ = None       # N_ x M

        for attr in [self.global_loss, self.validation_loss, self.split_duration,
                     self.model_range, self.input_range, self.cache]:
            attr.clear()
            
        self.k, self.N_, *_ = X.shape
        
        # --- input checks ---
        X, y = check_X_y(X, y, y_numeric=True)

        self.random_state_ = check_random_state(self.random_state)

        if self.early_stopping_tol != 0.075 and not self.early_stopping:
            print("[WARNING]: A tolerance for early stopping was set, but early stopping is disabled!")

        if input_range:
            assert self.N_ == len(input_range), \
                f"Dimension N from 'input_range' and 'X' does not agree: {self.N_} ≠ {len(input_range)}"

        # --- validation split ---
        if self.validation_size is not None and not np.isclose(self.validation_size, (0.0, )):
            if isinstance(self.validation_size, float):
                # proportion of the dataset to include in the validation split
                train_size = 1.0 - self.validation_size
            else:
                # absolute number of validation samples
                train_size = int(X.shape[0] - self.validation_size)

            X, self.X_val, y, self.y_val = train_test_split(X, y, train_size=train_size,
                                                            test_size=self.validation_size,
                                                            random_state=self.random_state_,
                                                            shuffle=True)

        if additional_validation_set and self.X_val is not None:
            X_val_add, y_val_add = check_X_y(*additional_validation_set)
            self.X_val = np.vstack((self.X_val, X_val_add))
            self.y_val = np.vstack((self.y_val, y_val_add))
        elif additional_validation_set:
            self.X_val, self.y_val = check_X_y(*additional_validation_set)

        # --- tracking training duration
        self.training_duration = 0
        start_time = time.time()  # start tracking the training duration

        # --- initialising model parameter ---
        self.X = X
        self.y = y
        self.output_constrains = output_constrains

        if not input_range:
            for j in range(self.N_):
                self.input_range.append((X[:, j].min(), X[:, j].max()))
        else:
            self.input_range = input_range

        if not np.isclose(self.kde_bandwidth, (0.0, )):
            self.kde_ = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian').fit(X)
            rep_dens = np.reciprocal(np.exp(self.kde_.score_samples(X)))
            self.R_ = sps.spdiags(rep_dens / np.max(rep_dens), diags=0, m=self.k, n=self.k)
        else:
            self.R_ = np.identity(self.k)

        # --- model fitting ---
        self._construct_component_models()
        # ----------------------
        
        self.training_duration = time.time() - start_time
        # print(f"[INFO] Finished model training after {time.time() - start_time:.4f} seconds.")

        return self

    def _predict(self, X):
        k = X.shape[0]
        A = np.zeros((self.M_, k))
        self._update_validity_functions(X, A)
        return self._get_model_output(X, A), A

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['M_', 'Theta_', 'Xi_', 'Sigma_'])

        # Input validation
        X = check_array(X)
        return self._predict(X)[0]

    def local_model_gen(self, sort_idx):
        Theta = self.Theta_[sort_idx, :]
        Xi = self.Xi_[:, sort_idx]
        if self.N_ == 1:
            for m, m_range in enumerate(np.array(self.model_range)[sort_idx]):
                u = np.linspace(*m_range[0])
                y = Theta[m, 1] * u + Theta[m, 0]
                c = Theta[m, 1] * Xi[:, m] + Theta[m, 0]
                yield u, y, c
        elif self.N_ == 2:
            for m, m_range in enumerate(self.model_range):
                u1 = np.linspace(*m_range[0], 2)
                u2 = np.linspace(*m_range[1], 2)
                u1, u2 = np.meshgrid(u1, u2)
                y = Theta[m, 2] * u2 + Theta[m, 1] * u1 + Theta[m, 0]
                c = Theta[m, 2] * Xi[1, m] + Theta[m, 1] * Xi[0, m] + Theta[m, 0]
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

    # --- online training ---

    def init_online_training(self, forgetting_factor=0.98, activity_threshold=0.1, init_covariance=1000, plotter=None):
        self.forgetting_factor = forgetting_factor
        self.activity_threshold = activity_threshold
        self.P_ = np.tile(np.identity(self.N_ + 1) * init_covariance, (self.M_, 1, 1))  # M x N x N
        self.plotter = plotter

    def online_training(self, X, y):
        x, y = check_X_y(X, y, y_numeric=True)
        x = np.vstack((np.ones(1), x))

        A = np.zeros((self.M_, 1))
        self._update_validity_functions(X, A)  # attention: use X instead of augmented x

        # --- local recursive weighted least square update ---
        for m in range(self.M_):
            if A[m, :] > self.activity_threshold:
                gamma = self.P_[m, :, :] @ x / (x.T @ self.P_[m, :, :] @ x + self.forgetting_factor * np.reciprocal(A[m, :]))
                self.Theta_[m, :] = self.Theta[m, :] + gamma @ (y - (x.T @ self.Theta[m, :]))
                self.P_[m, :, :] = 1/self.forgetting_factor * (self.P_[m, :, :] - gamma @ x.T @ self.P_[m, :, :])
        
        if self.plotter:
            # tracking of exemplary/arbitrary weights
            self.plotter.update({'theta00': self.Theta_[0, 0], 'theta01': self.Theta_[0, 1]})


class StructureOptimizer:
    """
    Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    """

class Lolimot(StructureOptimizer):
    """
    Local Linear Model Tree
    """


if __name__ == "__main__":
    from test import LolimotTest1D, LolimotTest2D
    import unittest
    unittest.main(verbosity=2)