import numpy as np


class LolimotApproximator(object):
	def __init__(self, sigma, l=101, model_complexity=100, x_range=[]):

		if x_range:
			assert isinstance(x_range, list), "'u_range' has to be a list of range tuples."
			self.N = len(x_range)
		else:
			self.N = None

		self.x_range = x_range
		self.current_range = None
		self.sigma = sigma
		self.model_complexity = model_complexity
		self.M = 1

		self.Theta = None   # M x N+1
		self.A = None       # M x k
		self.Xi = None      # N x M
		self.Sigma = None   # N x M
		self.X = None       # k x N
		self.y = None       # k
		self.k = None

		self.approximation_finished = False

	# --- private functions ---

	def _get_theta(self, m=1):
		Theta = np.zeros((m, self.N + 1))
		for i in range(m):
			X = np.array([X[m], np.ones_like(X[m])]).T
			theta_i = np.linalg.pinv(X) @ y
			Theta[m, :] = np.flip(theta_i, axis=0)

		return Theta

	def _increase_model_complexity(self, inc=1):
		for i in range(inc):
			self.M += 1
			self.Theta = np.vstack((self.Theta, np.zeros(1, self.N+1)))
			self.A = np.vstack((self.A, np.zeros(1, self.k)))
			self.Xi = np.hstack((self.Xi, np.zeros(self.N, 1)))
			self.Sigma = np.hstack((self.Sigma, np.zeros(self.N, 1)))

	def _construct_component_models(self):

		# 1. Initialize global model
		self.Theta = self._get_theta()
		self.current_range = self.x_range

		# 2. for every input dimension ...
		for j in range(1, self.N):
			# (a) ... split component model along j in two halves
			self._increase_model_complexity()
			r = self.current_range[j]
			ranges = [np.array([r.min(), r.mean()]), np.array([r.max(), r.mean()])]

			self.Xi[j,self.M-2:self.M] = map(lambda r: r.mean(), ranges)

			# (b) ... calculate membership functions for both models
			self.Sigma[j, self.M - 2:self.M] = map(lambda r: (abs(r)).sum() / np.sum(abs(self.x_range[j])) * self.sigma,
			                                       ranges)

			# (c) ... get models' parameter
			self.Theta[self.M - 2:self.M, :] = self._get_theta()

			# (d) ... calculate the tree's output error
			pass

	def _get_model_output(self, x):
		X = np.hstack((np.ones((self.k, 1)), x)).T
		return np.sum((self.Theta @ X) * self.A, axis=0)

	# --- public functions ---

	def fit(self, X, y):
		"""Fit the model according to the given training data."""

		# --- initialising model parameter ---
		self.X = X
		self.y = y
		self.k, N, *_ = X.shape

		if not self.N:
			self.N = N
		else:
			assert self.N == N,\
			f"Dimension N from 'u_range' input and 'X' does not agree: {self.N} ≠ {N}"

		if not self.x_range:
			for j in range(1, self.N):
				self.x_range.append((X[j,:].min(), X[j,:].max()))

		# --- modell fitting ---
		self._construct_component_models()
		# ----------------------
		self.approximation_finished = True

	def predict(self, X):
		assert self.approximation_finished, "Use function 'fit' to train the model before predicting."
		return self._get_model_output(X)

	def save_model(self):
		pass

	@staticmethod
	def load_model():
		pass