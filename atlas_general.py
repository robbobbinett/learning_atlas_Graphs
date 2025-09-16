from collections import OrderedDict
import os
import pickle
import itertools as it
import json

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from manifold_utils import LazyNeretinConvolution, quad_fit_full, \
				get_quadratic_terms, \
				get_quadratic_and_const_terms, \
				NeretinSmallError, \
				sample_from_ellipsoid_individual

NoneType = type(None)

class MemoryFIFOEnumeration:
	def __init__(self, start, eps, boundary_fun):
		# Dimensionaliity of the grid
		self.d = len(start)
		# Width of the grid
		self.eps = eps
		# FIFO queue
		self.candidates = []
		self.candidates.append(tuple(start))
		# Set of points already visited
		self.visited = set()
		self.visited.add(tuple(start.astype(int)))
		# Grid displacements for generating new observations
		self.generators = []
		for j in range(self.d):
			for sign in [-1, 1]:
				dx = np.zeros(self.d, dtype=int)
				dx[j] = sign
				self.generators.append(dx)
		# Boundary function for determining whether to
		# generate children
		self.boundary_fun = boundary_fun

	def enqueue(self, obj):
		if not (obj in self.visited):
			self.candidates.append(obj)

	def dequeue(self):
		candidate = self.candidates.pop(0)
		cand_vec = np.array(candidate, dtype=int)
		b_val = self.boundary_fun(self.eps * cand_vec)
		if b_val < 0:
			for gen_vec in self.generators:
				child_vec = cand_vec + gen_vec
				child = tuple(child_vec)
				self.enqueue(child)
				self.visited.add(child)

	def enumerate(self):
		while len(self.candidates) > 0:
			self.dequeue()

class AtlasGeneral:
	def __init__(self, d, D):
		# basic assertions
		assert isinstance(d, int)

		# save input parameters
		self.d = d
		self.D = D

		# coordinate chart storage
		self.chart_dict = OrderedDict()

		# auxiliary parameters
		self.n_charts = 0

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
		b_0 = boundary_fun(np.zeros(self.d))
		# Sometimes, origin of learned chart falls slightly
		# out of boundary
		self.boundary_fun_dict[self.n_charts] = boundary_fun
		self.n_charts += 1

	def save_atlas(self, dirpath):
		"""
		Save self using pickle module
		"""
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

	def xi_chart_to_ambient(self, xi, chart):
		x_0, L, M, h_mat, _, _, _ = self.chart_dict[chart]
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

		a = c + np.dot(x_0, A @ x_0) + np.dot(b, x_0)
		a_1 = 2 * L.T @ A.T @ x_0 + L.T @ b
		a_2 = 2 * H.T @ M.T @ A.T @ x_0 + H.T @ M.T @ b
		A_2 = L.T @ A @ L
		A_3 = 2 * L.T @ A @ M @ H
		A_4 = H.T @ M.T @ A @ M @ H

		def boundary_fun(tau):
			tau_quad = get_quadratic_terms(tau)

			quart = np.dot(tau_quad, np.dot(A_4, tau_quad))
			cubic = np.dot(tau, A_3 @ tau_quad)
			quad = np.dot(tau, A_2 @ tau)
			quad_long = np.dot(a_2, tau_quad)
			lin = np.dot(a_1, tau)
			return quart + cubic + quad + quad_long + lin + a
		return boundary_fun

	def sample_uniformly_from_chart_by_ind(self, ind, eps=0.1):
		start = np.zeros(self.d)

		boundary_fun = self.boundary_fun_dict[ind]
		memory_fifo_enumerator = MemoryFIFOEnumeration(start, eps,
							boundary_fun)
		memory_fifo_enumerator.enumerate()

		Xi_pre = list(memory_fifo_enumerator.visited)
		X_pre = []
		for xi_pre in Xi_pre:
			xi = eps * np.array(xi_pre)
			x = self.xi_chart_to_ambient(xi, ind)
			X_pre.append(x)

		return np.vstack(X_pre)

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
			x_0, L_p, M_p, h_mat, _, _, _ = self.chart_dict[ind]
			loss = self.chart_to_fun(X, x_0, L_p, M_p, h_mat)
			losses.append(loss[0])

		return np.argmin(losses)

	def ingest_ambient_point_given_ind(self, x, ind):
		x_0, L_p, _, _, _, _, _ = self.chart_dict[ind]
		x_trans = x_0 - x
		xi = L_p.T @ x_trans
		return xi

	def construct_dense_graph(self, delta=0.1, eps=1.0):
		"""Construct a fine mesh of the manifold by sampling from each
		coordinate chart. Edge weights determined by ambient distance.
		This graph is used to compute graph shortest-paths between points,
		which are used as a proxy for geodesic paths and path-lengths."""
		# Get points for graph vertices
		X_pre = []
		for chart in tqdm(range(self.n_charts)):
			x_0, L, M, h_mat, A, b, c = self.chart_dict[chart]
			boundary_fun = self.boundary_fun_dict[chart]
			X_smol = self.sample_uniformly_from_chart_by_ind(chart,
						eps=delta)
			X_pre.append(X_smol)
		X = np.vstack(X_pre)

		# Generate graph
		n_brute = X.shape[0]
		G = nx.Graph()
		for j in tqdm(range(n_brute)):
			X_j = X[j, :].reshape(1, 9)
			G.add_node(j)
			dist_vec = euclidean_distances(X_j, X)[0, :]
			for k in range(j+1, n_brute):
				dist = dist_vec[k]
				if dist <= eps:
					G.add_edge(j, k, weight=dist)

		self.X_brute = X
		self.G_brute = G
		# Cache shortest paths between points in self.X_brute
		self.shortest_path_cache = {}

	def shortest_path_cached(self, ind_0, ind_1):
		"""Return shortest path as list of nodes; memoized to improve
		performance. Must run construct_dense_graph method before calling
		this method."""
		key = (ind_0, ind_1)
		try:
			return self.shortest_path_cache[key]
		except KeyError:
			path = nx.shortest_path(self.G_brute, weight="weight",
					source=ind_0, target=ind_1)
			path_len = nx.shortest_path_length(self.G_brute,
					weight="weight", source=ind_0,
					target=ind_1)
			self.shortest_path_cache[key] = (path, path_len)

			return path, path_len

	def find_closest_brute_point(self, x):
		"""Find the closest point to x in self.X_brute. This method
		should only be called after calling the method
		construct_dense_graph """
		X = x.reshape(1, 9)
		dists = euclidean_distances(X, self.X_brute)[0, :]
		return np.argmin(dists)

	def approximate_shortest_path(self, xi_0, chart_0, xi_1, chart_1):
		"""Graph-shortest-path between two points in the atlas-graph.
		Make sure that dense subgraph has already been sampled before
		calling this method."""
		x_0 = self.xi_chart_to_ambient(xi_0, chart_0)
		x_1 = self.xi_chart_to_ambient(xi_1, chart_1)
		ind_0 = self.find_closest_brute_point(x_0)
		ind_1 = self.find_closest_brute_point(x_1)
		x_prime_0 = self.X_brute[ind_0, :]
		x_prime_1 = self.X_brute[ind_1, :]
		dist_0 = np.linalg.norm(x_0 - x_prime_0)
		dist_1 = np.linalg.norm(x_1 - x_prime_1)
		try:
			path_pre, path_len = self.shortest_path_cached(ind_0, ind_1)
		except AttributeError:
			raise AttributeError("Must run construct_dense_graph method before calling this method.")
		path = path_pre.copy()
		return dist_0, dist_1, path, path_len


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
