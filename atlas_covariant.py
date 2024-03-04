import os
import pickle
import itertools as it
import numpy as np

from manifold_utils import LazyNeretinConvolution, quad_fit_full, get_quadratic_terms, get_quadratic_and_const_terms, NeretinSmallError

NoneType = type(None)

def kernel_fun(x, y):
	"""
	Function to be used as kernel for SVM in atlas_covariant
	class. Takes
	"""

class atlas_covariant:
	def __init__(self, data, d,
			lnc_start=None, lnc_step=None,
			lnc_hwl=None, next_rule="mean"):
		# basic assertions
		assert isinstance(data, np.ndarray)
		assert isinstance(d, int)
		assert len(data.shape) == 2, str(data.shape)

		# save input parameters
		self.X = data
		self.N, self.D = data.shape
		self.d = d
		self.next_rule = next_rule

		# coordinate chart storage
		self.chart_dict = {}

		# auxiliary parameters
		self.n_charts = 0
		self.max_logprobs = -np.inf * np.ones(self.N)

		# parameters for LNC
		if isinstance(lnc_start, NoneType):
			self.lnc_start = int(np.ceil(self.N**(1/4)))
		else:
			self.lnc_start = lnc_start
		if isinstance(lnc_step, NoneType):
###			self.lnc_step = int(np.ceil(self.N**(1/4)))
			self.lnc_step = int(np.ceil(self.N**(1/2)))
		else:
			self.lnc_step = lnc_step
		if isinstance(lnc_hwl, NoneType):
			self.lnc_hwl = 10
		else:
			self.lnc_hwl = lnc_hwl

		# Store maps for transition boundaries
		self.boundary_fun_dict = {}

	def chart_to_fun(self, X, x_0, L, M, h_mat):
		"""
		X should be of shape (P, D), where D is the
		ambient dimension and P is some positive integer
		"""
		# Translate such that x_0 is origin
		X_trans = X - x_0
		# Get tangential coordinates
		Tau = X_trans @ L
		# Get normal coordinates
		Nu = X_trans @ M
		# Get constant and quadratic monomials of
		# tangential coordinates
		Tau_sqr = get_quadratic_and_const_terms(Tau)

		# Compute distance between actual and computed
		# normal coordinates
		Nu_prime = Tau_sqr @ h_mat
		return -np.linalg.norm(Nu - Nu_prime, axis=1)**2

	def chart_given_center(self, x_0):
		lnc = LazyNeretinConvolution(self.X, x_0, self.d,
			self.lnc_start, self.lnc_step, self.lnc_hwl)
		try:
			last_ind = lnc.convolution_til_bust()
		except NeretinSmallError:
			raise ValueError("Sad")
		eff_last_ind = lnc.get_effective_index(last_ind)
		last_ind = lnc.inds[eff_last_ind]
		x_last = self.X[last_ind, :]
		rad = np.linalg.norm(x_last - x_0)
		# Perform local quadratic approx
		X_prime = self.X[lnc.inds[:eff_last_ind], :]
		quad_params = quad_fit_full(X_prime, self.d)
		L_p = quad_params["L"]
		M_p = quad_params["M"]
		h_mat = quad_params["h_mat"]

		# Store new chart
		self.chart_dict[self.n_charts] = (x_0, L_p, M_p, rad, h_mat)
		self.boundary_fun_dict[self.n_charts] = self.construct_boundary_fun(rad, h_mat)

	def add_new_chart(self):
		if self.n_charts == 0:
			ind = np.random.randint(self.N)
			x_0 = self.X[ind, :]
			self.chart_given_center(x_0)
			# Update log-probs
			x_0, L_p, M_p, rad, h_mat = self.chart_dict[self.n_charts]
			self.max_logprobs = self.chart_to_fun(self.X, x_0, L_p, M_p, h_mat)
		else:
			# Get mean min log-prob
			if self.next_rule == "mean":
				mmlp = np.mean(self.max_logprobs)
				cond = False
				while not cond:
					ind = np.random.randint(self.N)
					if self.max_logprobs[ind] <= mmlp:
						cond = True
			elif self.next_rule == "min":
				ind = np.argmin(self.max_logprobs)
				"""
				print(ind)
				print(np.min(self.max_logprobs))
				print("\n\n")
				"""
			else:
				raise ValueError("Invalid keyword for next_rule.")
			x_0 = self.X[ind, :]
			self.chart_given_center(x_0)
			# Update log-probs
			x_0, L_p, M_p, rad, h_mat = self.chart_dict[self.n_charts]
			new_probs = self.chart_to_fun(self.X, x_0, L_p, M_p, h_mat)
			self.max_logprobs = np.max(np.stack([new_probs, self.max_logprobs]),
						axis=0)
		# Update number of charts
		self.n_charts += 1

	def save_atlas(self, dirpath, filename):
		"""
		Save self using pickle module
		"""
		filepath = dirpath + "/" + filename
		if os.path.isfile(filepath):
			os.remove(filepath)
		with open(filepath, "wb") as f:
			pickle.dump(self, f)

	def get_ind_to_chart(self):
		assert self.n_charts > 0, "self.n_charts must be greater than zero."
		# Collect logprobs associated with each chart as a list
		Logprobs_pre = []
		for j in range(self.n_charts):
			x_0, L_p, M_p, _, h_mat = self.chart_dict[j]
			Logprobs_pre.append(self.chart_to_fun(self.X, x_0, L_p, M_p, h_mat))

		# Reformat collection of logprobs as matrix
		Logprobs = np.stack(Logprobs_pre).T

		# Return index with highest log-probability
		return np.argmin(Logprobs, axis=1)

	def xi_chart_to_ambient(self, xi, x_0, L, M, h_mat):
		xi_quad = get_quadratic_and_const_terms(xi)
		quad_term = M @ h_mat.T @ xi_quad
		lin_term = L @ xi
		return quad_term + lin_term + x_0

	def xi_ind_to_ambient(self, xi, ind):
		# Get chart from ind
		x_0, L_p, M_p, _, h_mat = self.chart_dict[ind]
		return self.xi_chart_to_ambient(xi, x_0, L_p, M_p, h_mat)

	def construct_boundary_fun(self, rad, h_mat):
		h_vec = h_mat[0, :]
		H = h_mat[1:, :]
		HtH = H @ H.T
		hth = np.dot(h_vec, h_vec)
		Hth = H @ h_vec
		Hth2 = 2*Hth
		rad_sqr = rad**2
		c = hth - rad_sqr
		def boundary_fun(tau):
			tau_quad = get_quadratic_terms(tau)
			quart = np.dot(tau_quad, np.dot(HtH, tau_quad))
			quad = np.dot(tau, tau)
			quad_long = 2*np.dot(Hth2, tau_quad)
			return (quart + quad + quad_long + c)
		return boundary_fun

	def sample_uniformly_from_chart_by_ind(self, ind, grid_len=100):
		boundary_fun = self.boundary_fun_dict[ind]
		x_0, L_p, M_p, rad, h_mat = self.chart_dict[ind]
		X_pre = []
		xi_is = np.linspace(-rad, rad, grid_len)
		for xi_pre in it.product(xi_is, repeat=self.d):
			xi = np.array(xi_pre)
			b_val = boundary_fun(xi)
			if b_val <= 0:
				x = self.xi_chart_to_ambient(xi, x_0, L_p, M_p, h_mat)
				X_pre.append(x)
		X = np.vstack(X_pre)
		return X

	def get_num_params_aic(self):
		# d for x_0
		# dimension of special orthogonal group for
		# L_p and M_p
		# dimension of dxd symmetric matrices, times (n-d)
		# (n-d) for constant translation terms
		dim_SO_d = int(self.d*(self.d-1) / 2)
		d_tri = int(self.d*(self.d+1) / 2)
		Dmd = self.D - self.d
		num_params_pre = self.d + dim_SO_d + Dmd*d_tri + Dmd
		return self.n_charts * num_params_pre

	def get_model_aic(self):
		half_logprobs = np.sum(self.max_logprobs)
		half_aic = self.get_num_params_aic() - half_logprobs
		return 2 * half_aic

	def identify_chart(self, x):
		# Reshape x
		X = x.reshape(1, self.D)
		# Store losses
		losses = []
		for ind in range(self.n_charts):
			x_0, L_p, M_p, rad, h_mat = self.chart_dict[ind]
			loss = self.chart_to_fun(X, x_0, L_p, M_p, h_mat)
			losses.append(loss[0])
		print(np.max(losses))
		print(np.argmax(losses))
		print("\n")
		return np.argmin(losses)

	def ingest_ambient_point_given_ind(self, x, ind):
		x_0, L_p, M_p, rad, h_mat = self.chart_dict[ind]
		x_trans = x_0 - x
		xi = L_p.T @ x_trans
		return xi

	def ingest_ambient_point(self, x):
		# Identify best chart
		ind = self.identify_chart(x)
		# Ingest point
		return self.ingest_ambient_point_given_ind(x, ind), ind

def load_atlas(filepath):
	with open(filepath, "rb") as f:
		atlas_obj = pickle.load(f)
	return atlas_obj
