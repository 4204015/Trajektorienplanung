import time
import numpy as np
from copy import deepcopy


class LolimotRegression(object):
	"""
	Regression using Local (linear) models trained with the LOLIMOT algorithm.

	Parameters
	----------
	sigma : float, default: 0.4
		Used only 'sigma_option' is set to 'const', to specify a constant standard deviation
		of the normalized Gaussian validity functions.

	smoothing : string, 'const' or 'proportional', default: 'const'
		Specifies whether a constant standard deviation is used, or is calculated proportional
		to the extension of the hyperrectangle of each local model.

	p : float, default: 1/3
		Proportionality factor between the rectangles' extension and the standard deviations.

	model_complexity : int, default: 100
		Maximum number of local models.

	x_range : list of tuples, optional, default: []
		Range of values for each input dimension N for which the model should be trained.
	"""

	def __init__(self, sigma=0.4, smoothing='const', p=1/3, model_complexity=100, x_range=[]):

		if x_range:
			assert isinstance(x_range, list), "'x_range' has to be a list of range tuples."
			self.N = len(x_range)
		else:
			self.N = None

		self.x_range = x_range
		self.sigma = sigma
		self.model_complexity = model_complexity

		self.smoothing=smoothing
		self.p = p

		self.M = None           # Number of models
		self.model_range = []   # M x N
		self.Theta = None       # M x N+1
		self.A = None           # M x k
		self.Xi = None          # N x M
		self.Sigma = None       # N x M
		self.X = None           # k x N
		self.y = None           # k x _
		self.k = None

		self.approximation_finished = False

	# --- private functions ---

	def _get_theta(self, model_pointers=None):
		if not model_pointers:
			model_pointers = range(self.M)
			n = self.M
		else:
			n = len(model_pointers)
		Theta = np.zeros((n, self.N + 1))
		for i, m in enumerate(model_pointers): # for model m
			Q_m = np.diag(self.A[m, :])  # weight matrix Q
			X_reg = np.hstack((self.X, np.ones((self.k, 1)))) # regression matrix
			theta_m = np.linalg.inv(X_reg.T @ Q_m @ X_reg) @ X_reg.T @ Q_m @ self.y
			Theta[i, :] = np.flip(theta_m, axis=0).flatten()
		return Theta

	def _update_validity_functions(self):
		Xi_ext = np.reshape(np.repeat([self.Xi], self.k, axis=0).T, (self.M, self.k, 1))
		U_ext = np.broadcast_to(self.X, (self.M, self.k, self.N))
		C = (U_ext - Xi_ext)**2 * np.reciprocal(self.Sigma)**2
		mu = np.exp(-0.5 * np.sum(C, axis=2))  # summation along N-axis -> Mxk
		mu_sum = np.sum(mu, axis=0)            # summation along M-axis -> k
		self.A = mu * np.reciprocal(mu_sum)

	def _increase_model_complexity(self, increment=1):
		for _ in range(increment):
			self.M += 1
			self.Theta = np.vstack((self.Theta, np.zeros((1, self.N+1))))
			self.A = np.vstack((self.A, np.zeros((1, self.k))))
			self.Xi = np.hstack((self.Xi, np.zeros((self.N, 1))))
			self.Sigma = np.hstack((self.Sigma, np.zeros((self.N, 1))))
			self.model_range.append([()])

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
		return self.A @ (self.y - self._get_model_output(self.X))**2 # Loss function output -> M x _

	def _get_global_loss(self):
		return np.sum((self.y - self._get_model_output(self.X))**2)

	def _split_along(self, j, l, m):
		# (a) ... split component model along j in two halves
		r = self.model_range[l][j]
		ranges = [(np.min(r), np.mean(r)), (np.max(r), np.mean(r))]
		self.model_range[l][j], self.model_range[m][j] = ranges

		self.Xi[j, (l, m)] = list(map(lambda r: np.mean(r), ranges))

		if self.smoothing == 'const':
			self.Sigma[j, (l, m)] = [self.sigma, self.sigma]
		else: # 'proportional'
			self.Sigma[j, (l, m)] = list(map(lambda r: np.absolute(r).sum() * self.p, ranges))

		#self.Sigma[j, (l, m)] = list(map(lambda r: np.absolute(r).sum() / np.absolute(self.x_range[j]).sum() * self.sigma, ranges))

		# (b) ... calculate validity functions all models
		self._update_validity_functions()

		# (c) ... get models' parameter
		self.Theta[(l, m), :] = self._get_theta((l, m))

	def _construct_component_models(self):
		# 1. Initialize global model
		self.M = 1
		self.Xi = np.zeros((self.N, self.M))
		self.Sigma = np.zeros((self.N, self.M))

		self.Xi[:, self.M-1] = [np.mean(r) for r in self.x_range]
		self.Sigma[:, :] = self.sigma
		self._update_validity_functions()
		self.Theta = self._get_theta((0, ))
		self.model_range.append(self.x_range)

		while True:
			# 2. Find worst LLM
			l = np.argmax(self._get_local_loss()) # the model denoted by 'l' is considered for further refinement
			self._increase_model_complexity()
			m = self.M - 1 # denotes the most recent added model

			L_global = np.zeros(self.N) # global model loss for every split attempt
			self._save_params()

			# 3. for every input dimension ...
			for j in range(self.N):
				self._split_along(j, l, m)

				# (d) ... calculate the tree's output error
				L_global[j] = self._get_global_loss()

				# Undo changes 'from _split_along'
				self._recover_params()

			# 4. find best division (split) and apply
			j = np.argmin(L_global)
			self._split_along(j, l, m)

			# 5. test for convergence
			if self.M >= self.model_complexity:
				break

	def _get_model_output(self, u):
		U = np.hstack((np.ones((self.k, 1)), u)).T
		y_hat = np.sum((self.Theta @ U) * self.A, axis=0)
		return np.reshape(y_hat, (u.shape[0], 1))

	# --- public functions ---

	def fit(self, X, y):
		"""Fit the model according to the given training data."""

		start_time = time.time()

		# --- initialising model parameter ---
		self.X = X
		self.y = y
		self.k, N, *_ = X.shape

		if not self.N:
			self.N = N
		else:
			assert self.N == N,\
			f"Dimension N from 'x_range' input and 'X' does not agree: {self.N} ≠ {N}"

		if not self.x_range:
			for j in range(self.N):
				self.x_range.append((X[:, j].min(), X[:, j].max()))

		# --- modell fitting ---
		self._construct_component_models()
		# ----------------------
		self.approximation_finished = True

		print(f"[INFO] Finished model training after {time.time() - start_time:.4f} seconds.")

	def predict(self, X):
		assert self.approximation_finished, "Use function 'fit' to train the model before predicting."
		self.X = X
		self.k = X.shape[0]
		self._update_validity_functions()
		return self._get_model_output(X)

	def local_model_gen(self):
		# TODO: Extend function to the case N>1
		for m, m_range in enumerate(self.model_range):
			u = np.linspace(*m_range[0])
			y = self.Theta[m, 1] * u + self.Theta[m, 0]
			c = self.Theta[m, 1] * self.Xi[:, m] + self.Theta[m, 0]
			yield u, y, c

	def save_model(self):
		pass

	@staticmethod
	def load_model():
		pass
