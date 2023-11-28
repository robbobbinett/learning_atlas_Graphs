import os
import pickle
import itertools as it
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from kmedoids import fasterpam
from tqdm import tqdm

from manifold_utils import quad_fit_full, get_quadratic_terms, get_quadratic_and_const_terms

NoneType = type(None)

def dist_mat_from_generator(gen, total):
	dist_mat_pre = []
	for _, row in tqdm(gen, total=total):
		row_list = []
		for _, w in row.items():
			row_list.append(w)
		dist_mat_pre.append(row_list)
	return np.array(dist_mat_pre)

class atlas_graph:
	def __init__(self, data, d):
		# Basic assertions
		assert isinstance(data, np.ndarray)
		assert isinstance(d, int)
		assert len(data.shape) == 2, str(data.shape)

		# Save input parameters
		self.X = data
		self.N, self.D = data.shape
		self.d = d

		# Coordinate chart storage
		self.chart_dict = {}

		# Auxiliary parameters
		self.max_logprobs = -np.inf * np.ones(self.N)

		# Store maps for transition boundaries
		self.boundary_fun_dict = {}

		# Count number of charts
		self.n_charts = 0

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
		#return np.linalg.norm(Nu - Nu_prime, axis=1)**2
		return -np.linalg.norm(Nu - Nu_prime, axis=1)**2

	def xi_chart_to_ambient(self, xi, x_0, L, M, h_mat):
		xi_quad = get_quadratic_and_const_terms(xi)
		quad_term = M @ h_mat.T @ xi_quad
		lin_term = L @ xi
		return quad_term + lin_term + x_0

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

	def ingest_ambient_point(self, x):
		# Identify best chart
		ind = self.identify_chart(x)
		# Ingest point
		return self.ingest_ambient_point_given_ind(x, ind), ind

	def identify_chart(self, x):
		# Reshape x
		X = x.reshape(1, self.D)
		# Store losses
		losses = []
		for ind in range(self.n_charts):
			x_0, L_p, M_p, rad, h_mat = self.chart_dict[ind]
			loss = self.chart_to_fun(X, x_0, L_p, M_p, h_mat)
			losses.append(loss[0])
		return np.argmax(losses)

	def ingest_ambient_point_given_ind(self, x, ind):
		x_0, L_p, M_p, rad, h_mat = self.chart_dict[ind]
		x_trans = x - x_0
		xi = L_p.T @ x_trans
		return xi

	def xi_ind_to_ambient(self, xi, ind):
		# Get chart from ind
		x_0, L_p, M_p, _, h_mat = self.chart_dict[ind]
		return self.xi_chart_to_ambient(xi, x_0, L_p, M_p, h_mat)

	def store_atlas(self, file_pre):
		for key in self.chart_dict.keys():
			x_0, L_p, M_p, rad_eff, h_mat = self.chart_dict[key]
			np.save(file_pre+"_x"+str(key)+".npy", x_0)
			np.save(file_pre+"_L"+str(key)+".npy", L_p)
			np.save(file_pre+"_M"+str(key)+".npy", M_p)
			np.save(file_pre+"_rad"+str(key)+".npy", rad_eff)
			np.save(file_pre+"_h_mat"+str(key)+".npy", h_mat)

	def load_atlas(self, file_pre, n_charts):
		self.chart_dict = {}
		for ind in range(n_charts):
			x = np.load(file_pre+"_x"+str(ind)+".npy")
			L = np.load(file_pre+"_L"+str(ind)+".npy")
			M = np.load(file_pre+"_M"+str(ind)+".npy")
			rad = np.load(file_pre+"_rad"+str(ind)+".npy")
			h_mat = np.load(file_pre+"_h_mat"+str(ind)+".npy")
			self.chart_dict[ind] = (x, L, M, rad, h_mat)

class atlas_pam(atlas_graph):
	def __init__(self, data, d, n_charts, km_max_iter=1000, save_dist_mat=False,
			load_dist_mat=False, load_atlas=False):
		# Initialize parent class
		super().__init__(data, d)

		# Set number of charts
		self.n_charts = n_charts
		# Learn atlas-graph using K medoids on neighborhood graph
		### Fit neighbor graph
		nn_sklearn = NearestNeighbors()
		nn_sklearn.fit(self.X)

		### Compute induced pairwise distances
		print("Getting graph as sparse matrix...")
		knn_graph = nn_sklearn.kneighbors_graph()
		print("Done")
		print("Getting graph from sparse matrix...")
		self.G = nx.from_scipy_sparse_array(knn_graph)
		print("Done")

		### Perform Floyd-Warshall on NetworkX
		if load_dist_mat:
			self.dist_mat = np.load("saved_dist_mat.npy")
		else:
			print("Generating distance matrix...")
			self.dist_mat = np.zeros((self.N, self.N))
			for j in tqdm(range(self.N)):
				for k in range(j+1, self.N):
					val = nx.shortest_path_length(self.G,
						source=j, target=k, weight="weight")
					self.dist_mat[j, k] = val
					self.dist_mat[k, j] = val
		if save_dist_mat:
			np.save("saved_dist_mat.npy", self.dist_mat)
		#self.dist_mat = dist_mat_from_generator(dist_mat_gen, self.N)
		print("Done")

		if isinstance(load_atlas, NoneType):
			### Perform k-medoids
			km = fasterpam(self.dist_mat, self.n_charts,
					max_iter=km_max_iter,
					n_cpu=1)

			### Create charts from k-medoids
			for ind in range(self.n_charts):
				inds = (km.labels == ind)
				pts = self.X[inds, :]
				quad_params = quad_fit_full(pts, self.d)
				x_0 = self.X[km.medoids[ind]]
				L_p = quad_params["L"]
				M_p = quad_params["M"]
				h_mat = quad_params["h_mat"]
				X_0 = x_0.reshape(1, self.D)
				dist_vec = euclidean_distances(X_0, pts)
				rad = np.max(dist_vec)
				# Save chart
				self.chart_dict[ind] = (x_0, L_p, M_p, rad, h_mat)
				# Save boundary function
				self.boundary_fun_dict[ind] = self.construct_boundary_fun(rad, h_mat)
				# Update logprobs
				new_logprobs = self.chart_to_fun(self.X, x_0, L_p, M_p, h_mat)
				self.max_logprobs = np.max([self.max_logprobs, new_logprobs], axis=0)
		else:
			assert isinstance(load_atlas, str)
			# Load charts
			self.load_atlas(load_atlas, self.n_charts)
			# Save boundary functions and update logprobs
			for ind in range(self.n_charts):
				x_0, L_p, M_p, rad, h_mat = self.chart_dict[ind]
				# Save boundary function
				self.boundary_fun_dict[ind] = self.construct_boundary_fun(rad, h_mat)
				# Update logprobs
				new_logprobs = self.chart_to_fun(self.X, x_0, L_p, M_p, h_mat)
				self.max_logprobs = np.max([self.max_logprobs, new_logprobs], axis=0)
