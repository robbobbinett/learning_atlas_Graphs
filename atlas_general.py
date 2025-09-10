from collections import OrderedDict
import os
import pickle
import itertools as it
import json

import numpy as np

from manifold_utils import LazyNeretinConvolution, quad_fit_full, \
				get_quadratic_terms, \
				get_quadratic_and_const_terms, \
				NeretinSmallError, \
				sample_from_ellipsoid_individual

NoneType = type(None)

def kernel_fun(x, y):
	"""
	Function to be used as kernel for SVM in atlas_covariant
	class. Takes
	"""

class atlas_general:
#	def __init__(self, data, d):
	def __init__(self, d, D):
		# basic assertions
#		assert isinstance(data, np.ndarray)
		assert isinstance(d, int)
#		assert len(data.shape) == 2, str(data.shape)

		# save input parameters
#		self.X = data
#		self.N, self.D = data.shape
#		self.d = d
#		self.next_rule = next_rule
		self.d = d
		self.D = D

		# coordinate chart storage
		self.chart_dict = OrderedDict()

		# auxiliary parameters
		self.n_charts = 0
#		self.max_logprobs = -np.inf * np.ones(self.N)

		# Store maps for transition boundaries
		self.boundary_fun_dict = OrderedDict()

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

	def add_new_chart(self, x_0, L, M, h_mat, A, b, c):
		self.chart_dict[self.n_charts] = (x_0, L, M, h_mat,
							A, b, c)
		boundary_fun = self.construct_boundary_fun(x_0, L, M,
							h_mat, A, b, c)
		self.boundary_fun_dict[self.n_charts] = boundary_fun
		self.n_charts += 1

	def save_atlas(self, dirpath):
		"""
		Save self using pickle module
		"""
#		filepath = dirpath + "/" + filename
#		if os.path.isfile(filepath):
#			os.remove(filepath)
#		with open(filepath, "w") as f:
#			json.dump(self.chart_dict, f, indent=4)
		os.system(f"rm -rf {dirpath}")
		os.system(f"mkdir {dirpath}")
		for j in range(self.n_charts):
			subdirpath = f"{dirpath}/chart_{j}"
			os.system(f"mkdir {subdirpath}")

			x_0, L, M, h_mat, A, b, c = self.chart_dict[j]
			np.save(f"{subdirpath}/x_0.npy", x_0)
			np.save(f"{subdirpath}/L.npy", L)
			np.save(f"{subdirpath}/M.npy", M)
			np.save(f"{subdirpath}/h_mat.npy", h_mat)
			np.save(f"{subdirpath}/A.npy", A)
			np.save(f"{subdirpath}/b.npy", b)
			np.save(f"{subdirpath}/c.npy", c)

	def get_ind_to_chart(self):
		assert self.n_charts > 0, "self.n_charts must be greater than zero."
		# Collect logprobs associated with each chart as a list
		Logprobs_pre = []
		for j in range(self.n_charts):
			x_0, L_p, M_p, h_mat, _, _, _ = self.chart_dict[j]
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

	def construct_boundary_fun(self, x_0, L, M, h_mat, A, b, c):
		h_vec = h_mat[0, :]
		H = h_mat[1:, :].T
#		try:
#			temp = H.T @ M.T
#		except ValueError:
#			err_str = f"H.shape: {H.shape}; M.shape: {M.shape}"
#			raise ValueError(err_str)
		a = c + np.dot(x_0, A @ x_0) + np.dot(b, x_0)
		a_1 = 2 * L.T @ A.T @ x_0 + L.T @ b
		a_2 = H.T @ M.T @ A.T @ x_0 + H.T @ M.T @ b / 2
		A_2 = L.T @ A @ L
		A_3 = L.T @ A @ M @ H
		A_4 = H.T @ M.T @ A @ M @ H / 4
#		Ainv = np.linalg.inv(A)
#		y_0 = x_0 + Ainv @ b / 2
#		h_vec = h_mat[0, :]
#		H = h_mat[1:, :]
#		A_4 = H.T @ M.T @ A @ M @ H / 4
#		A_3 = L.T @ A @ M @ H
#		A_2b = L.T @ A @ L
#		A_2c = np.dot(y_0, A @ M @ H)
#		A_1 = 2 * np.dot(A @ L, y_0)
#		a = c + np.dot(y_0, A @ y_0)
		def boundary_fun(tau):
			tau_quad = get_quadratic_terms(tau)

			quart = np.dot(tau_quad, np.dot(A_4, tau_quad))
			cubic = np.dot(tau, A_3 @ tau_quad)
			quad = np.dot(tau, A_2 @ tau)
			quad_long = np.dot(a_2, tau_quad)
			lin = np.dot(a_1, tau)
			return -(quart + cubic + quad + quad_long + lin + a)
		return boundary_fun

	def sample_uniformly_from_chart_by_ind(self, ind, n_pts):
		boundary_fun = self.boundary_fun_dict[ind]
		X_pre = []
		_, _, _, _, A, b, c = self.chart_dict[ind]
		while len(X_pre) < n_pts:
			x = sample_from_ellipsoid_individual(A, b, c)
			xi = self.ingest_ambient_point_given_ind(x, ind)
			b_val = boundary_fun(xi)
			if b_val <= 0:
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
			x_0, L_p, M_p, h_mat, _, _, _, _ = self.chart_dict[ind]
			loss = self.chart_to_fun(X, x_0, L_p, M_p, h_mat)
			losses.append(loss[0])
		print(np.max(losses))
		print(np.argmax(losses))
		print("\n")
		return np.argmin(losses)

	def ingest_ambient_point_given_ind(self, x, ind):
		x_0, L_p, _, _, _, _, _ = self.chart_dict[ind]
		x_trans = x_0 - x
		xi = L_p.T @ x_trans
		return xi

	def ingest_ambient_point(self, x):
		# Identify best chart
		ind = self.identify_chart(x)
		# Ingest point
		return self.ingest_ambient_point_given_ind(x, ind), ind

def load_atlas(dirpath, d, D):
	atlas_obj = atlas_general(d, D)

	n_charts = len(os.listdir(dirpath))
	for j in range(n_charts):
		subdirpath = f"{dirpath}/chart_{j}"

		x_0 = np.load(f"{subdirpath}/x_0.npy")
		L = np.load(f"{subdirpath}/L.npy")
		M = np.load(f"{subdirpath}/M.npy")
		h_mat = np.load(f"{subdirpath}/h_mat.npy")
		A = np.load(f"{subdirpath}/A.npy")
		b = np.load(f"{subdirpath}/b.npy")
		c = np.load(f"{subdirpath}/c.npy")

		atlas_obj.add_new_chart(x_0, L, M, h_mat,
					A, b, c)

	return atlas_obj
