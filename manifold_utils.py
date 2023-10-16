import os
from collections import OrderedDict
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from scipy.spatial import Delaunay

cached_triu_indices = {}

def memoized_triu_indices(d):
	"""
	Accesses the precomputed indices if cached;
	otherwise calls np.triu_indices(d) and caches
	within cached_triu_indices
	"""
	try:
		return cached_triu_indices[d]
	except KeyError:
		inds = np.triu_indices(d)
		cached_triu_indices[d] = inds
		return inds

def get_quadratic_terms(x_mat):
	"""
	If x_mat is a NumPy array of shape (n, d), each of whose
	rows are vectors of length d with values x_i,
	this function returns a matrix of all terms x_i*x_j
	for i <= j per row. The ordering of these products is inherited
	from np.triu_indices.
	"""
	rows = []
	if len(x_mat.shape) == 2:
		N, d = x_mat.shape
		triu_inds = memoized_triu_indices(d)
		for j in range(N):
			vec = x_mat[j, :]
			rows.append(np.outer(vec, vec)[triu_inds])
		return np.vstack(rows)
	elif len(x_mat.shape) == 1:
		d = len(x_mat)
		triu_inds = memoized_triu_indices(d)
		return np.outer(x_mat, x_mat)[triu_inds]
	else:
		raise ValueError("x_mat should be a 1d or 2d array; instead, has shape "+str(x_mat.shape))

def get_quadratic_and_const_terms(x_mat):
	if len(x_mat.shape) == 2:
		N, d = x_mat.shape
		return np.hstack([np.ones((N, 1)), get_quadratic_terms(x_mat)])
	elif len(x_mat.shape) == 1:
		d = len(x_mat)
		return np.hstack([1.0, get_quadratic_terms(x_mat)])
	else:
		raise ValueError("x_mat should be a 1d or 2d array; instead, has shape "+str(x_mat.shape))

def quad_fit_naive(points, d, L=None):
	"""
	inputs:
	points (np.ndarray): data matrix of shape n_points x dim_ambient
	d (int): suspected instrinsic dimension of the manifold

	outputs:
	TODO
	"""
	# Get number of points and ambient dimension
	N, n = points.shape

	# Center points
	mean = np.mean(points, axis=0)
	X = points - mean

	# Compute top d principal components
	# Note that the ith column of U corresponds to the
	# ith entry of L
	# Note that the eigenvalues in L are given in
	# ascending order

	if L is None:
		# Get orthonormal basis for tangent plane
		tSVD = TruncatedSVD(d)
		tSVD.fit(X)
		L = tSVD.components_.T

	# Get projection onto normal plane
	P_perp = np.eye(n) - L @ L.T

	# Project points onto tangent and normal planes
	X_par = X @ L
	X_perp = X @ P_perp

	# For each point, compute values of constant and
	# quadratic terms
	t_mat = get_quadratic_and_const_terms(X_par)

	# Compute from Equations 11 in the supplementary material
	# of Sritharan et al. (2021)
	t_inner_inv = np.linalg.inv(t_mat.T @ t_mat)
	h_mat = t_inner_inv @ t_mat.T @ X_perp
	Sig_eps_pre = X_perp - t_mat @ h_mat
	Sig_eps = (Sig_eps_pre.T @ Sig_eps_pre) / N

	# Return computed values
	quad_params = {}
	quad_params["mean"] = mean
	quad_params["L"] = L
	quad_params["X_par"] = X_par
	quad_params["t_mat"] = t_mat
	quad_params["h_mat"] = h_mat
	quad_params["Sig_eps"] = Sig_eps

	return quad_params

def quad_fit_full(points, d, L=None, M=None):
	"""
	inputs:
	points (np.ndarray): data matrix of shape n_points x dim_ambient
	d (int): suspected instrinsic dimension of the manifold

	outputs:
	TODO
	"""
	# Get number of points and ambient dimension
	N, n = points.shape

	# Center points
	mean = np.mean(points, axis=0)
	X = points - mean

	# Compute top d principal components
	# Note that the ith column of U corresponds to the
	# ith entry of L
	# Note that the eigenvalues in L are given in
	# ascending order

	if L is None and M is None:
		# Get orthonormal bases for tangent plane
		# and normal plane
		pca = PCA()
		pca.fit(X)
		## Tangent plane
		L = pca.components_[:d, :].T
		## Normal plane
		M = pca.components_[d:, :].T

	# Project points onto tangent and normal planes
	X_par = X @ L
	X_perp = X @ M

	# For each point, compute values of constant and
	# quadratic terms
	t_mat = get_quadratic_and_const_terms(X_par)

	# Compute from Equations 11 in the supplementary material
	# of Sritharan et al. (2021)
	t_inner_inv = np.linalg.inv(t_mat.T @ t_mat)
	h_mat = t_inner_inv @ t_mat.T @ X_perp
	Sig_eps_pre = X_perp - t_mat @ h_mat
	Sig_eps = (Sig_eps_pre.T @ Sig_eps_pre) / N

	# Return computed values
	quad_params = {}
	quad_params["mean"] = mean
	quad_params["L"] = L
	quad_params["M"] = M
	quad_params["X_par"] = X_par
	quad_params["t_mat"] = t_mat
	quad_params["h_mat"] = h_mat
	quad_params["Sig_eps"] = Sig_eps

	return quad_params

def get_sig_eps_stats(Sig_eps):
	"""
	Get statistics about variances of residuals
	"""
	Sig_eps_diag = np.diag(Sig_eps)
	stats = {}
	stats["0.25 percentile"] = np.percentile(Sig_eps_diag, 25)
	stats["0.5 percentile"] = np.percentile(Sig_eps_diag, 50)
	stats["0.75 percentile"] = np.percentile(Sig_eps_diag, 75)
	stats["mean"] = np.mean(Sig_eps_diag)
	stats["max"] = np.max(Sig_eps_diag)

	return stats

def get_sig_h_stats(Sig_eps, t_mat):
	"""
	Get statistics about variances of quadratic coefficients
	"""
	t_inner_inv = np.linalg.inv(t_mat.T @ t_mat)
	Sig_h_diag = np.kron(np.diag(Sig_eps), np.diag(t_inner_inv))
	stats = {}
	stats["0.25 percentile"] = np.percentile(Sig_h_diag, 25)
	stats["0.5 percentile"] = np.percentile(Sig_h_diag, 50)
	stats["0.75 percentile"] = np.percentile(Sig_h_diag, 75)
	stats["mean"] = np.mean(Sig_h_diag)
	stats["max"] = np.max(Sig_h_diag)

	return stats

def h_mat_to_trilinear(h_mat):
	"""
	h_mat as output from quad_fit_naive
	d is intrinsic dimension of manifold
	D is ambient dimension
	"""
	# Trim constant terms
	h_mat_trimmed = h_mat[1:, :]
	d_tri, D = h_mat_trimmed.shape
	d = int((np.sqrt(1 + 8*d_tri) - 1) / 2)

	# Get upper-triangular indices
	row_inds, col_inds = memoized_triu_indices(d)

	# Initialize 3-tensor
	to_return = np.zeros((d, d, D), dtype=np.float64)

	# Iterate over codimensions
	for alpha in range(D):
		# Iterate over combinations of tangent dimensions
		for i, inds in enumerate(zip(row_inds, col_inds)):
			j, k  = inds
			if j == k:
				to_return[j, k, alpha] = h_mat_trimmed[i, alpha]
			else:
				to_return[j, k, alpha] = h_mat_trimmed[i, alpha] / 2
				to_return[k, j, alpha] = h_mat_trimmed[i, alpha] / 2
	return to_return

def curvatures_from_trilinear(H):
	d, _, D = H.shape
	eig_decomps = []
	for alpha in range(D):
		eig_decomps.append(np.linalg.eigh(H[:, :, alpha]))
	return eig_decomps

def generate_equilateral_simplex(d):
	"""Return the vertices of the 'beta-double-star' triangulation of S^d."""
	# Standard basis
	E = np.eye(d)

	# Follow the equations from the derivation
	# in the project Overleaf.
	t = (np.sqrt(d) + 1 - np.sqrt(d + 1)) / 2
	## Compute rho by numerator and denominator
	rho_num = d*np.sqrt(d) - d*np.sqrt(d+1) + np.sqrt(d) + np.sqrt(d**2 + d)
	rho = rho_num / (2*d + 2)

	alpha = np.ones(d, dtype=np.float64) / np.sqrt(d)

	# Compute betas-star in order to compute
	# betas-star-star
	betas_s = []
	for j in range(d):
		betas_s.append((1 - t)*E[j, :] + t*alpha)

	# Compute betas-star-star
	betas_ss = []
	betas_ss.append(-alpha) # beta_ss zero
	for beta_s in betas_s: # all other beta_ss
		betas_ss.append(beta_s/rho - alpha)

	Beta_ss = np.vstack(betas_ss)
	return Beta_ss

def lexsort_list_of_tuples(list_o_t):
	"""
	Lexicographically sort a list of tuples of floats,
	using np.lexsort to implement the actual lexicographic
	sorting.
	"""
	# Convert list of tuples to 2D NumPy array
	arr = np.vstack([np.array(toop) for toop in list_o_t])
	# Use NumPy's lexicographic sort
	sorted_inds = np.lexsort(arr.T)
	# Revert to list of tuples
	return [tuple(arr[ind, :]) for ind in sorted_inds]

class EmptyGeodeError(Exception):
	pass

class GlassGraph:
	def __init__(self, d, init_scheme="equilateral",
			graph_seed=None, node_to_simplex=None,
			boundary=None, A=None, v=None):
		self.d = d
		# equilateral initialization scheme
		if init_scheme == "equilateral":
			Beta_ss = generate_equilateral_simplex(d)

			# Rescale by A
			if (A is not None):
				Beta_ss = Beta_ss @ A
				self.A = A
			else:
				self.A = np.eye(self.d,
						dtype=np.float64)
			# Translate by v
			if (v is not None):
				Beta_ss = Beta_ss - v
				self.v = v
			else:
				self.v = np.zeros(self.d)

			# Initialize graph
			self.G = nx.Graph()
			## Initialize all nodes
			for j in range(d+1):
				self.G.add_node(j)
			## Initialize all edges
			for j in range(d+1):
				for k in range(j+1, d+1):
					self.G.add_edge(j, k)

			# Associate d-simplex with each node
			self.node_dict = OrderedDict()
			for j in range(d+1):
				simplex = []
				for k in range(d+1):
					if k != j:
						simplex.append(tuple(Beta_ss[k, :]))
				self.node_dict[j] = lexsort_list_of_tuples(simplex)

			# Associate (d-1)-simplex with each edge
			self.edge_dict = OrderedDict()
			for j in range(d+1):
				for k in range(j+1, d+1):
					face_j = self.node_dict[j]
					face_k = self.node_dict[k]
					# The following will already be sorted by that
					# face_j, face_k are already sorted
					self.edge_dict[(j, k)] = [pt for pt in face_j if pt in face_k]

			# Next integer to be used as a node name
			self.next_name = d+1

			# Keep track of which faces are on the boundary;
			# for the equilateral init_scheme, there is no boundary
			self.boundary = set()

		# Seeded initialization scheme
		elif init_scheme == "seeded":
			self.G = graph_seed
			self.node_dict = node_to_simplex

			# Check that all simplices have the same cardinality
			for _, simplex in self.node_dict.items():
				assert len(simplex) == self.d

			# Check that all points have the same dimension
			for _, simplex in self.node_dict.items():
				for pt in simplex:
					assert len(pt) == self.d

			# Associate facet with each edge
			self.edge_dict = OrderedDict()
			for edge in self.G.edges:
				sorted_edges = list(edge)
				sorted_edges.sort()
				node_a, node_b = sorted_edges
				face_a = self.node_dict[node_a]
				face_b = self.node_dict[node_b]
				facet = [vert for vert in face_a if vert in face_b]
				self.edge_dict[(node_a, node_b)] = facet

			# Next name for subsequent nodes
			self.next_name = np.max(list(self.G.nodes)) + 1

			# Keep track of which faces are on the boundary
			if not boundary:
				self.boundary = set()
				for node in self.G.nodes:
					if len(list(self.G.neighbors(node))) < self.d:
						self.boundary.add(node)
			else:
				self.boundary = boundary

			# Matrix A for ellipsoid definition
			if (A is None):
				self.A = np.eye(self.d, dtype=np.float64)
			else:
				self.A = A

			# Translation v for ellipsoid definition
			if (v is None):
				self.v = np.zeros(self.d, dtype=np.float64)
			else:
				self.v = v

		else:
			raise ValueError("Invalid string for init_scheme.")

	def get_next_name(self):
		"""This method should be used to initialize all nodes,
		with the exception of those nodes initialized in the
		__init__ method"""
		to_return = self.next_name
		self.next_name += 1
		return to_return

	def recompute_boundary(self):
		"""Recompute boundary from scratch"""
		self.boundary = set()
		for node in self.G.nodes:
			if len(list(self.G.neighbors(node))) < self.d:
				self.boundary.add(node)

	def centroid_split(self, face):
		"""Here, face is the integer 'name' of the simplex
		that is to be split."""
		orig_neighbors = list(self.G.neighbors(face))
		orig_edges = []
		for neigh in orig_neighbors:
			if face < neigh:
				orig_edges.append((face, neigh))
			else:
				orig_edges.append((neigh, face))
		pts = self.node_dict[face]
		facets = []
		for edge in orig_edges:
			facets.append(self.edge_dict[edge])

		# Get names of new nodes
		new_nodes = []
		for _ in range(self.d):
			new_nodes.append(self.get_next_name())

		# Remove parent node
		self.G.remove_node(face)
		del self.node_dict[face]
		self.boundary.discard(face)
		for edge in orig_edges:
			del self.edge_dict[edge]

		# Insert new nodes as clique
		self.G.add_nodes_from(new_nodes)
		for j in range(self.d):
			nnj = new_nodes[j]
			for k in range(j+1, self.d):
				nnk = new_nodes[k]
				self.G.add_edge(nnj, nnk)

		# Compute centroid of removed face
		cen_pre = np.mean(pts, axis=0)
		cen_pre_trans = cen_pre - self.v
		norm = np.sqrt(np.inner(cen_pre_trans, self.A @ cen_pre_trans))
		cen = tuple((cen_pre_trans/norm) + self.v)

		# Save simplex information of new nodes,
		# and also introduce non-clique edges and corresponding simplices
		for j, node in enumerate(new_nodes):
			simplex = [pts[k] for k in range(self.d) if k != j]
			tupled_facets = []
			for facet in facets:
				tupled_facets.append(tuple(facet))
			if tuple(simplex) in tupled_facets:
				for edge, facet in zip(orig_edges, facets):
					if simplex == facet:
						other = [ind for ind in edge if ind != face][0]
						self.G.add_edge(other, node)
						self.edge_dict[(other, node)] = simplex
			else:
				self.boundary.add(node)
			simplex = simplex.copy()
			simplex.append(cen)
			self.node_dict[node] = lexsort_list_of_tuples(simplex)

		# Save simplices for clique edges
		for j in range(self.d):
			node_1 = new_nodes[j]
			face_1 = self.node_dict[node_1]
			for k in range(j+1, self.d):
				node_2 = new_nodes[k]
				face_2 = self.node_dict[node_2]
				facet = [vert for vert in face_1 if vert in face_2]
				self.edge_dict[(node_1, node_2)] = facet

	def facet_split(self, node_a, node_b):
		"""Here, node_a and node_b are the integer
		'names' of the simplices that are to be split."""
		assert node_a < node_b
		# Get original faces and main facet of the split
		face_a = self.node_dict[node_a]
		face_b = self.node_dict[node_b]
		main_facet = self.edge_dict[(node_a, node_b)]

		# Get vertex opposite main facet for each
		# original face
		vert_a = [vert for vert in face_a if vert not in main_facet][0]
		vert_b = [vert for vert in face_b if vert not in main_facet][0]

		# Get edges adjoining original nodes
		orig_edges_a = []
		for neigh in self.G.neighbors(node_a):
			if neigh != node_b:
				if neigh < node_a:
					orig_edges_a.append((neigh, node_a))
				else:
					orig_edges_a.append((node_a, neigh))
		orig_edges_b = []
		for neigh in self.G.neighbors(node_b):
			if neigh != node_a:
				if neigh < node_b:
					orig_edges_b.append((neigh, node_b))
				else:
					orig_edges_b.append((node_b, neigh))

		# Get facets corresponding to mentioned edges
		facets_a = []
		for edge in orig_edges_a:
			facets_a.append(self.edge_dict[edge])
		facets_b = []
		for edge in orig_edges_b:
			facets_b.append(self.edge_dict[edge])

		# Remove parent nodes
		self.G.remove_nodes_from([node_a, node_b])
		del self.node_dict[node_a]
		del self.node_dict[node_b]
		self.boundary.discard(node_a)
		self.boundary.discard(node_b)
		del self.edge_dict[(node_a, node_b)]
		for edge in orig_edges_a:
			del self.edge_dict[edge]
		for edge in orig_edges_b:
			del self.edge_dict[edge]

		# Insert two (d-1)-cliques with
		# pairwise connections
		new_nodes_a = []
		new_nodes_b = []
		for _ in range(self.d-1):
			new_node = self.get_next_name()
			new_nodes_a.append(new_node)
			self.G.add_node(new_node)
		for _ in range(self.d-1):
			new_node = self.get_next_name()
			new_nodes_b.append(new_node)
			self.G.add_node(new_node)
		for j in range(self.d-1):
			# pairwise connection
			self.G.add_edge(new_nodes_a[j], new_nodes_b[j])
			# cliques
			for k in range(j+1, self.d-1):
				self.G.add_edge(new_nodes_a[j], new_nodes_a[k])
				self.G.add_edge(new_nodes_b[j], new_nodes_b[k])

		# Get centroid of main facet
		cen_pre = np.mean(main_facet, axis=0)
		cen_pre_trans = cen_pre - self.v
		norm = np.sqrt(np.inner(cen_pre_trans, self.A @ cen_pre_trans))
		cen = tuple((cen_pre_trans/norm) + self.v)

		# Associate d-simplex to each new node.
		# Also associate (d-1)-simplex to each inter-clique
		# edge;
		# Also add edges from neighbors of the parent
		# node to the new nodes, together with associated
		# (d-1)-simplices.
		for j, pt in enumerate(main_facet):
			new_node_a = new_nodes_a[j]
			new_node_b = new_nodes_b[j]
			simplex = [otra for otra in main_facet if otra != pt]

			# inter-clique edge
			ice = simplex.copy()
			ice.append(cen)
			ice = lexsort_list_of_tuples(ice)
			self.edge_dict[(new_node_a, new_node_b)] = ice

			# node_a
			simplex_a = simplex.copy()
			simplex_a.append(vert_a)
			simplex_a = lexsort_list_of_tuples(simplex_a)
			tupled_facets = []
			for facet in facets_a:
				tupled_facets.append(tuple(facet))
			if tuple(simplex_a) in tupled_facets:
				for edge, facet in zip(orig_edges_a, facets_a):
					if facet == simplex_a:
						other = [node for node in edge if node != node_a][0]
						self.G.add_edge(other, new_node_a)
						self.edge_dict[(other, new_node_a)] = simplex_a
			else:
				self.boundary.add(new_node_a)
			simplex_a = simplex_a.copy()
			simplex_a.append(cen)
			simplex_a = lexsort_list_of_tuples(simplex_a)
			self.node_dict[new_node_a] = simplex_a

			# node_b
			simplex_b = simplex.copy()
			simplex_b.append(vert_b)
			simplex_b = lexsort_list_of_tuples(simplex_b)
			tupled_facets = []
			for facet in facets_b:
				tupled_facets.append(tuple(facet))
			if tuple(simplex_b) in tupled_facets:
				for edge, facet in zip(orig_edges_b, facets_b):
					if facet == simplex_b:
						other = [node for node in edge if node != node_b][0]
						self.G.add_edge(other, new_node_b)
						self.edge_dict[(other, new_node_b)] = simplex_b
			else:
				self.boundary.add(new_node_b)
			simplex_b = simplex_b.copy()
			simplex_b.append(cen)
			simplex_b = lexsort_list_of_tuples(simplex_b)
			self.node_dict[new_node_b] = simplex_b

		# Associate (d-1)-simplex with each intra-clique
		# edge.
		for j in range(self.d-1):
			node_a_j = new_nodes_a[j]
			node_b_j = new_nodes_b[j]
			face_a_j = self.node_dict[node_a_j]
			face_b_j = self.node_dict[node_b_j]
			for k in range(j+1, self.d-1):
				node_a_k = new_nodes_a[k]
				node_b_k = new_nodes_b[k]
				face_a_k = self.node_dict[node_a_k]
				face_b_k = self.node_dict[node_b_k]

				facet_a = [vert for vert in face_a_j if vert in face_a_k]
				facet_b = [vert for vert in face_b_j if vert in face_b_k]

				self.edge_dict[(node_a_j, node_a_k)] = facet_a
				self.edge_dict[(node_b_j, node_b_k)] = facet_b

	def judicious_split(self, node):
		"""Split node (and possibly a single neighbor alongside)
		with either centroid or facet split. The one that is chosen
		is determined by the ratios of median lengths."""
		# Compute lengths of medians
		ls = []
		simplex = self.node_dict[node]
		for j, vert in enumerate(simplex):
			# Get facet opposite vertex
			facet = [simplex[k] for k in range(self.d) if k != j]
			# Get centroid of this facet
			cen = np.mean(facet, axis=0)
			# Compute median length
			ls.append(np.linalg.norm(np.array(vert) - cen))
		# Replace lengths of medians with log-lengths
		ls = np.log(ls)
		# Get max and min lengths as well as index
		# of the min length
		l_min = np.min(ls)
		l_min_ind = np.argmin(ls)
		l_max = np.max(ls)
		# Find out which of l_min and l_max
		# is "more outlier"
		mu = np.mean(ls)
		l_min_score = mu - l_min
		l_max_score = l_max - mu
		# If l_max is more outlier, do centroid split
		if l_max_score >= l_min_score:
			self.centroid_split(node)
		else:
			# Do centroid split by facet corresponding
			# to smallest median length
			facet = [simplex[j] for j in range(self.d) if j != l_min_ind]
			for neigh in self.G.neighbors(node): # search all facets
				if neigh < node:
					smaller = neigh
					larger = node
				else:
					smaller = node
					larger = neigh
				cand_facet = self.edge_dict[(smaller, larger)]
				if cand_facet == facet:
					break
			self.facet_split(smaller, larger)

class GeodeGraph:
	def __init__(self, graph_seed, node_to_simplex,
			boundary=None, allow_empty=False):
		"""
		graph_seed should be a networkx graph with
		integer node labels.
		node_to_simplex should be an OrderedDict instance
		from node-labels in the graph to a lexicographically
		sorted list of (k+1) points in d-
		dimensional space, represented as tuples (of
		np.float64) of length d.
		boundary, if not None, is a set containing the integer
		indices of full-dimensional faces that instersect the
		boundary of the simplicial complex. Otherwise, this
		is computed from scratch.
		"""
		if not allow_empty:
			if (len(node_to_simplex) == 0) or (len(graph_seed.nodes) == 0):
				raise EmptyGeodeError("Cannot make an empty GeodeGraph instance.")
			self.G = graph_seed
			self.node_dict = node_to_simplex

			# Check that all simplices have same cardinality
			simplices = []
			for _, item in self.node_dict.items():
				simplices.append(item)
			self.k = len(simplices[0])
			assert self.k > 1
			for simplex in simplices[1:]:
				assert len(simplex) == self.k

			# Check that all points have the same dimension
			self.d = len(simplices[0][0])
			for simplex in simplices:
				for pt in simplex:
					assert len(pt) == self.d

			# Associate facet with each edge
			self.edge_dict = OrderedDict()
			for edge in self.G.edges:
				sorted_edges = list(edge)
				sorted_edges.sort()
				node_a, node_b = sorted_edges
				face_a = self.node_dict[node_a]
				face_b = self.node_dict[node_b]
				facet = [vert for vert in face_a if vert in face_b]
				self.edge_dict[(node_a, node_b)] = facet

			# Next name for subsequent nodes
			self.next_name = np.max(list(self.G.nodes)) + 1

			# Keep track of which faces are on the boundary
			if not boundary:
				self.boundary = set()
				for node in self.G.nodes:
					if len(list(self.G.neighbors(node))) < self.k:
						self.boundary.add(node)
			else:
				self.boundary = boundary
		else:
			atts = ["G", "node_dict", "k", "d",
				"edge_dict", "next_name",
				"boundary"]
			for att in atts:
				setattr(self, att, None)

	def get_next_name(self):
		"""This method should be used to initialize all nodes,
		with the exception of those nodes initialized in the
		__init__ method"""
		to_return = self.next_name
		self.next_name += 1
		return to_return

	def recompute_boundary(self):
		"""Recompute boundary from scratch"""
		self.boundary = set()
		for node in self.G.nodes:
			if len(list(self.G.neighbors(node))) < self.k:
				self.boundary.add(node)

	def centroid_split(self, face):
		"""Here, face is the integer 'name' of the simplex
		that is to be split."""
		orig_neighbors = list(self.G.neighbors(face))
		orig_edges = []
		for neigh in orig_neighbors:
			if face < neigh:
				orig_edges.append((face, neigh))
			else:
				orig_edges.append((neigh, face))
		pts = self.node_dict[face]
		facets = []
		for edge in orig_edges:
			facets.append(self.edge_dict[edge])

		# Get names of new nodes
		new_nodes = []
		for _ in range(self.d+1):
			new_nodes.append(self.get_next_name())

		# Remove parent node
		self.G.remove_node(face)
		del self.node_dict[face]
		self.boundary.discard(face)
		for edge in orig_edges:
			del self.edge_dict[edge]

		# Insert new nodes as clique
		self.G.add_nodes_from(new_nodes)
		for j in range(self.d+1):
			nnj = new_nodes[j]
			for k in range(j+1, self.d+1):
				nnk = new_nodes[k]
				self.G.add_edge(nnj, nnk)

		# Compute centroid of removed face
		cen = tuple(np.mean(pts, axis=0))

		# Save simplex information of new nodes,
		# and also introduce non-clique edges and corresponding simplices
		for j, node in enumerate(new_nodes):
			simplex = [pts[k] for k in range(len(pts)) if k != j]
			tupled_facets = []
			for facet in facets:
				tupled_facets.append(tuple(facet))
			if tuple(simplex) in tupled_facets:
				for edge, facet in zip(orig_edges, facets):
					if simplex == facet:
						other = [ind for ind in edge if ind != face][0]
						self.G.add_edge(other, node)
						self.edge_dict[(other, node)] = simplex
			else:
				self.boundary.add(node)
			simplex = simplex.copy()
			simplex.append(cen)
			self.node_dict[node] = lexsort_list_of_tuples(simplex)

		# Save simplices for clique edges
		for j in range(self.d+1):
			node_1 = new_nodes[j]
			face_1 = self.node_dict[node_1]
			for k in range(j+1, self.d+1):
				node_2 = new_nodes[k]
				face_2 = self.node_dict[node_2]
				facet = [vert for vert in face_1 if vert in face_2]
				self.edge_dict[(node_1, node_2)] = facet

	def facet_split(self, node_a, node_b):
		"""Here, node_a and node_b are the integer
		'names' of the simplices that are to be split."""
		assert node_a < node_b
		# Get original faces and main facet of the split
		face_a = self.node_dict[node_a]
		face_b = self.node_dict[node_b]
		main_facet = self.edge_dict[(node_a, node_b)]

		# Get vertex opposite main facet for each
		# original face
		vert_a = [vert for vert in face_a if vert not in main_facet][0]
		vert_b = [vert for vert in face_b if vert not in main_facet][0]

		# Get edges adjoining original nodes
		orig_edges_a = []
		for neigh in self.G.neighbors(node_a):
			if neigh != node_b:
				if neigh < node_a:
					orig_edges_a.append((neigh, node_a))
				else:
					orig_edges_a.append((node_a, neigh))
		orig_edges_b = []
		for neigh in self.G.neighbors(node_b):
			if neigh != node_a:
				if neigh < node_b:
					orig_edges_b.append((neigh, node_b))
				else:
					orig_edges_b.append((node_b, neigh))

		# Get facets corresponding to mentioned edges
		facets_a = []
		for edge in orig_edges_a:
			facets_a.append(self.edge_dict[edge])
		facets_b = []
		for edge in orig_edges_b:
			facets_b.append(self.edge_dict[edge])

		# Remove parent nodes
		self.G.remove_nodes_from([node_a, node_b])
		del self.node_dict[node_a]
		del self.node_dict[node_b]
		self.boundary.discard(node_a)
		self.boundary.discard(node_b)
		del self.edge_dict[(node_a, node_b)]
		for edge in orig_edges_a:
			del self.edge_dict[edge]
		for edge in orig_edges_b:
			del self.edge_dict[edge]

		# Insert two (d-1)-cliques with
		# pairwise connections
		new_nodes_a = []
		new_nodes_b = []
		for _ in range(self.d):
			new_node = self.get_next_name()
			new_nodes_a.append(new_node)
			self.G.add_node(new_node)
		for _ in range(self.d):
			new_node = self.get_next_name()
			new_nodes_b.append(new_node)
			self.G.add_node(new_node)
		for j in range(self.d):
			# pairwise connection
			self.G.add_edge(new_nodes_a[j], new_nodes_b[j])
			# cliques
			for k in range(j+1, self.d):
				self.G.add_edge(new_nodes_a[j], new_nodes_a[k])
				self.G.add_edge(new_nodes_b[j], new_nodes_b[k])

		# Get centroid of main facet
		cen = tuple(np.mean(main_facet, axis=0))

		# Associate d-simplex to each new node.
		# Also associate (d-1)-simplex to each inter-clique
		# edge;
		# Also add edges from neighbors of the parent
		# node to the new nodes, together with associated
		# (d-1)-simplices.
		for j, pt in enumerate(main_facet):
			new_node_a = new_nodes_a[j]
			new_node_b = new_nodes_b[j]
			simplex = [otra for otra in main_facet if otra != pt]

			# inter-clique edge
			ice = simplex.copy()
			ice.append(cen)
			ice = lexsort_list_of_tuples(ice)
			self.edge_dict[(new_node_a, new_node_b)] = ice

			# node_a
			simplex_a = simplex.copy()
			simplex_a.append(vert_a)
			simplex_a = lexsort_list_of_tuples(simplex_a)
			tupled_facets = []
			for facet in facets_a:
				tupled_facets.append(tuple(facet))
			if tuple(simplex_a) in tupled_facets:
				for edge, facet in zip(orig_edges_a, facets_a):
					if facet == simplex_a:
						other = [node for node in edge if node != node_a][0]
						self.G.add_edge(other, new_node_a)
						self.edge_dict[(other, new_node_a)] = simplex_a
			else:
				self.boundary.add(new_node_a)
			simplex_a = simplex_a.copy()
			simplex_a.append(cen)
			simplex_a = lexsort_list_of_tuples(simplex_a)
			self.node_dict[new_node_a] = simplex_a

			# node_b
			simplex_b = simplex.copy()
			simplex_b.append(vert_b)
			simplex_b = lexsort_list_of_tuples(simplex_b)
			tupled_facets = []
			for facet in facets_b:
				tupled_facets.append(tuple(facet))
			if tuple(simplex_b) in tupled_facets:
				for edge, facet in zip(orig_edges_b, facets_b):
					if facet == simplex_b:
						other = [node for node in edge if node != node_b][0]
						self.G.add_edge(other, new_node_b)
						self.edge_dict[(other, new_node_b)] = simplex_b
			else:
				self.boundary.add(new_node_b)
			simplex_b = simplex_b.copy()
			simplex_b.append(cen)
			simplex_b = lexsort_list_of_tuples(simplex_b)
			self.node_dict[new_node_b] = simplex_b

		# Associate d-simplex with each intra-clique
		# edge.
		for j in range(self.d):
			node_a_j = new_nodes_a[j]
			node_b_j = new_nodes_b[j]
			face_a_j = self.node_dict[node_a_j]
			face_b_j = self.node_dict[node_b_j]
			for k in range(j+1, self.d):
				node_a_k = new_nodes_a[k]
				node_b_k = new_nodes_b[k]
				face_a_k = self.node_dict[node_a_k]
				face_b_k = self.node_dict[node_b_k]

				facet_a = [vert for vert in face_a_j if vert in face_a_k]
				facet_b = [vert for vert in face_b_j if vert in face_b_k]

				self.edge_dict[(node_a_j, node_a_k)] = facet_a
				self.edge_dict[(node_b_j, node_b_k)] = facet_b

	def judicious_split(self, node):
		"""Split node (and possibly a single neighbor alongside)
		with either centroid or facet split. The one that is chosen
		is determined by the ratios of median lengths."""
		# Compute lengths of medians
		ls = []
		simplex = self.node_dict[node]
		for j, vert in enumerate(simplex):
			# Get facet opposite vertex
			facet = [simplex[k] for k in range(self.d) if k != j]
			# Get centroid of this facet
			cen = np.mean(facet, axis=0)
			# Compute median length
			ls.append(np.linalg.norm(np.array(vert) - cen))
		# Replace lengths of medians with log-lengths
		ls = np.log(ls)
		# Get max and min lengths as well as index
		# of the min length
		l_min = np.min(ls)
		l_min_ind = np.argmin(ls)
		l_max = np.max(ls)
		# Find out which of l_min and l_max
		# is "more outlier"
		mu = np.mean(ls)
		l_min_score = mu - l_min
		l_max_score = l_max - mu
		# If l_max is more outlier, do centroid split
		if l_max_score >= l_min_score:
			self.centroid_split(node)
		else:
			# Do centroid split by facet corresponding
			# to smallest median length
			facet = [simplex[j] for j in range(self.d) if j != l_min_ind]
			for neigh in self.G.neighbors(node): # search all facets
				if neigh < node:
					smaller = neigh
					larger = node
				else:
					smaller = node
					larger = neigh
				cand_facet = self.edge_dict[(smaller, larger)]
				if cand_facet == facet:
					break
			self.facet_split(smaller, larger)

	def export_to_file(self, path_to_dir):
		# Create empty directory if none is
		# present. Otherwise, empty existing
		# directory.
		if not os.path.isdir(path_to_dir):
			os.mkdir(path_to_dir)
		else:
			for filename in os.listdir(path_to_dir):
				filepath = "/".join([path_to_dir, filename])
				if os.path.isfile(filepath):
					os.remove(filepath)

		# Export self.G
		filepath = "/".join([path_to_dir, "G.xml"])
		nx.write_graphml(self.G, filepath)

		# Pickle all other attributes
		atts = ["boundary", "d", "k", "node_dict",
			"edge_dict", "next_name"]
		for att in atts:
			filepath = "/".join([path_to_dir, att+".pkl"])
			file = open(filepath, "wb")
			pkl.dump(getattr(self, att), file)
			file.close()

def read_GeodeGraph(path_to_dir):
	"""
	Reverse of GeodeGraph.export_to_file
	"""
	# Read in exported attributes
	graph_seed = nx.read_graphml("/".join([path_to_dir, "G.xml"]))
	atts = ["boundary", "d", "k", "node_dict",
		"edge_dict", "next_name"]
	att_to_obj = {}
	for att in atts:
		filepath = "/".join([path_to_dir, att+".pkl"])
		file = open(filepath, "rb")
		att_to_obj[att] = pkl.load(file)
		file.close()

	# Create GeodeGraph instance from attributes
	geode_obj = GeodeGraph(nx.Graph(),
			OrderedDict(),
			allow_empty=True)
	for att in atts:
		setattr(geode_obj, att, att_to_obj[att])

	return geode_obj

class AtlasGraph:
	def __init__(self, geode_seed, node_to_ind,
			ind_to_chart, ind_to_ellipsoid):
		"""
		geode_seed (GeodeGraph): GeodeGraph attribute
		node_to_ind (dict): maps node indices in
			GeodeGraph instance to chart indices
		ind_to_chart (dict): maps chart indices to
			(p, L, r, h_mat)-tuples. Here, p is the
			point in ambient coordinates, L is a Stiefel
			matrix encoding the tangent plane at p,
			r is the ambient radius of the ellipsoid,
			and h_mat is the matrix of coefficients for
			the local quadratic approximation
		ind_to_ellipsoid (dict): maps chart indices to
			covariance matrix of ellipsoid in
			tangential coordinates
		"""
		self.geode = geode_seed

		# Make sure nodes in geode_seed and keys of
		# node_to_ind agree
		for node in self.geode.node_dict.keys():
			assert node in node_to_ind
		for key in node_to_ind.keys():
			assert key in self.geode.node_dict

		self.node_to_ind = node_to_ind
		self.ind_to_chart = ind_to_chart
		self.ind_to_ellipsoid = ind_to_ellipsoid

	def geode_from_index_restriction(self, inds):
		"""
		Returns a GeodeGraph instance representing
		the restriction of self.geode to nodes whose
		index is an element of inds.

		inds: list of int
		"""
		node_to_simplex = OrderedDict()
		graph_seed = self.geode.G.copy()
		for node, simplex in self.geode.node_dict.items():
			ind = self.node_to_ind[node]
			if ind in inds:
				node_to_simplex[node] = simplex
			else:
				graph_seed.remove_node(node)

		return GeodeGraph(graph_seed, node_to_simplex)

	def compute_LMR_overlap_digraph(self):
		inds = list(self.ind_to_chart.keys())
		graph = nx.DiGraph()
		graph.add_nodes_from(inds)
		for j, ind_j in enumerate(inds):
			chart_j = self.ind_to_chart[ind_j]
			p_j, L_j, r_j, h_j = chart_j
			Sigma_j = self.ind_to_ellipsoid[ind_j]
			for k, ind_k in enumerate(inds):
				if j != k:
					chart_k = self.ind_to_chart[ind_k]
					p_k, L_k, r_k, h_k = chart_k
					Sigma_k = self.ind_to_ellipsoid[ind_k]
					W_kj = L_k.T @ L_j
					W_kj_inv = np.linalg.inv(W_kj)
					Sigma_prime = W_kj.T @ Sigma_k @ W_kj
					center_prime = W_kj_inv @ L_k.T @ (p_k - p_j)
					# yeet

	def export_to_file(self, path_to_dir):
		if os.path.isdir(path_to_dir):
			# Clear pre-existing files
			for filename in os.listdir(path_to_dir):
				filepath = "/".join([path_to_dir, filename])
				if os.path.isfile(filepath):
					os.remove(filepath)
			# Clear subdirectory for storing self.geode
			geode_path = "/".join([path_to_dir, "geode"])
			if os.path.isdir(geode_path):
				for filename in os.listdir(geode_path):
					filepath = "/".join([geode_path, filename])
					if os.path.isfile(filepath):
						os.remove(filepath)
			else:
				os.mkdir(geode_path)
		else:
			os.mkdir(path_to_dir)
			geode_path = "/".join([path_to_dir, "geode"])
			os.mkdir(geode_path)

		# Store self.geode
		self.geode.export_to_file(geode_path)

		# Store other attributes
		atts = ["ind_to_chart", "ind_to_ellipsoid",
			"node_to_ind"]
		for att in atts:
			filepath = "/".join([path_to_dir, att+".pkl"])
			file = open(filepath, "wb")
			pkl.dump(getattr(self, att), file)
			file.close()

def read_AtlasGraph(path_to_dir):
	"""
	Reverse of AtlasGraph.export_to_file
	"""
	geode_path = "/".join([path_to_dir, "geode"])
	geode_seed = read_GeodeGraph(geode_path)

	atts = ["ind_to_chart", "ind_to_ellipsoid",
		"node_to_ind"]
	att_to_obj = {}
	for att in atts:
		filepath = "/".join([path_to_dir, att+".pkl"])
		file = open(filepath, "rb")
		att_to_obj[att] = pkl.load(file)
		file.close()

	return AtlasGraph(geode_seed,
			att_to_obj["node_to_ind"],
			att_to_obj["ind_to_chart"],
			att_to_obj["ind_to_ellipsoid"])

def sample_from_d_sphere(n_points, d):
	# Generate angular coordinates
	ang_coors_pre = np.random.randn(n_points, d)
	norms = np.linalg.norm(ang_coors_pre, axis=1)
	inv_norms = 1/norms
	ang_coors = np.einsum("ij,i->ij", ang_coors_pre, inv_norms)

	return ang_coors

def sample_from_d_ball(n_points, d):
	# Generate angular coordinates
	ang_coors_pre = np.random.randn(n_points, d)
	norms = np.linalg.norm(ang_coors_pre, axis=1)
	inv_norms = 1/norms
	ang_coors = np.einsum("ij,i->ij", ang_coors_pre, inv_norms)

	# Generate radial coordinate
	radii_pre = np.random.rand(n_points)
	radii = radii_pre**(1/d)

	return np.einsum("ij,i->ij", ang_coors, radii)

def child_from_village(Sigma, cut_geode,
			node_to_ind, ind_to_ellipsoid,
			n_interior_pts=10,
			n_boundary_pts=5):
	"""
	Sigma (np.ndarray): d x d positive-definite matrix
		defining child ellipsoid. Child ellipsoid
		is assumed to be centered at the origin
	cut_geode (GeodeGraph): GeodeGraph instance representing
		the projection of pertinent nodes in the
		AtlasGraph to the tangent space of the child
		centerpoint
	node_to_ind (dict): maps node indices in cut_geode
		to ellipsoid indices
	ind_to_ellipsoid (dict): maps ellipsoid indices to
		(Sigma_p, center_p) tuples. Here, Sigma_p is
		the d x d positive-definite matrix, and
		center_p is the d-dimensional vector,
		defining the ellipsoid.
	n_interior_pts (int): number of randomly generated
		points used to make (interior of) Delaunay
		triangulation
	n_boundary_pts (int): number of randomly generated
		points used to make (boundary of) Delaunay
		triangulation
	"""
	# Get boundary simplices intersecting new
	# ellipsoid
	nodes_of_interest = []
	for ind in cut_geode.boundary:
		simplex = cut_geode.node_dict[ind]
		scores = []
		for vert in simplex:
			vec = np.array(vert)
			scores.append(np.inner(vec, Sigma @ vec))
		conds = [score < 1 for score in scores]
		if np.count_nonzero(conds) >= (cut_geode.k - 1):
			nodes_of_interest.append(ind)

	# Restrict those simplices to their outermost facet
	# (here, "outermost" is defined relative to the center
	# and PD matrix of the ellipsoid corresponding to the
	# parent simplex)
	facet_dict = OrderedDict()
	for node in nodes_of_interest:
		face = cut_geode.node_dict[node]
		# Get PD matrix and center of parent ellipsoid
		ind = node_to_ind[node]
		### Parafox
		Sigma_p, center_p = ind_to_ellipsoid[ind]
		scores = []
		for vert in face:
			vec = np.array(vert) - center_p
			scores.append(np.inner(vec, Sigma_p @ vec))
		void_ind = np.argmin(scores)
		facet_dict[node] = [face[j] for j in range(len(face)) if j != void_ind]

	# Project each parent-boundary vertex to the
	# surface of child ellipsoid
	vert_to_proj = {}
	proj_to_vert = {}
	for node in nodes_of_interest:
		facet = facet_dict[node]
		ind = node_to_ind[node]
		Sigma_p, center_p = ind_to_ellipsoid[ind]
		for vert in facet:
			if vert not in vert_to_proj:
				vec = np.array(vert) - center_p
				score = np.sqrt(np.inner(vec, Sigma_p @ vec))
				new_vert = tuple(vec/score + center_p)
				vert_to_proj[vert] = new_vert
				proj_to_vert[new_vert] = vert

	# Randomly sample from uniform distribution on child
	# ellipsoid. First, sample from uniform distribution on
	# unit ball of same dimension. Then apply affine
	# transformation

	## Precompute affine transformation matrix
	Lambda, V = np.linalg.eigh(Sigma)
	aff_mat = np.diag(Lambda**(-1/2)) @ V.T

	## Sample boundary, then apply affine transformation
	sphere_pts_pre_pre = sample_from_d_sphere(n_boundary_pts,
					cut_geode.d)
	sphere_pts_pre = (sphere_pts_pre_pre @ aff_mat)
	## Only keep points not covered by a parent ellipsoid
	not_covered = []
	for j in range(n_boundary_pts):
		x = sphere_pts_pre[j, :]
		cond = True
		for _, toople in ind_to_ellipsoid.items():
			Sigma_p, center_p = toople
			vec = x - center_p
			score = np.inner(vec, Sigma_p@vec)
			if score <= 1.0:
				cond = False
		if cond:
			not_covered.append(j)
	sphere_pts = sphere_pts_pre[not_covered, :]

	## Sample interior, and apply affine transformation
	ball_pts_pre = sample_from_d_ball(n_interior_pts, cut_geode.d)
	ball_pts = (ball_pts_pre @ aff_mat)

	## Merge boundary with interior
	ell_pts = np.vstack([sphere_pts, ball_pts])

	# Combine points from parent boundary to ell_pts
	new_pts_pre = []
	for pt in proj_to_vert.keys():
		new_pts_pre.append(pt)
	if len(new_pts_pre) > 0:
		new_pts = np.vstack(new_pts_pre)
		comb_pts = np.vstack([new_pts, ell_pts])
	else:
		comb_pts = ell_pts

	# Compute Delaunay triangulation of comb_pts
	delaunay = Delaunay(comb_pts)
	delaunay_pts = delaunay.points
	delaunay_simps = delaunay.simplices
	delaunay_neighbors = delaunay.neighbors

	# Collect indices parent ellipsoids
	ell_inds_set = set()
	for node in nodes_of_interest:
		ell_inds_set.add(node_to_ind[node])
	ell_inds = list(ell_inds_set)

	# Transform points to agree with parent
	# boundary coordinates. Note that this is
	# already done for points originating from
	# a parent
	for j in range(ell_pts.shape[0]):
		x = ell_pts[j, :]
		# Compute score with regards to each
		# parent ellipsoid. If score < 1, then
		# save the image of x projected onto the
		# boundary of the parent ellipsoid
		x_norm_c = np.sqrt(np.inner(x, Sigma @ x))
		score_vec = x/x_norm_c
		ys = []
		for ind in ell_inds:
			Sigma_p, center_p = ind_to_ellipsoid[ind]
			vec = score_vec - center_p
			score = np.inner(vec, Sigma_p @ vec)
			if score < 1:
				# Compute t according to quadratic
				t = solve_shrinking_quadratic(x, center_p, Sigma_p)
				ys.append(t*x)
		# if ys is empty, rad_rat = 1.0
		if len(ys) == 0:
			rad_rat = 1.0
		else:
			# Find y with smallest score with regards to
			# child ellipsoid
			y_scores = []
			for y in ys:
				y_scores.append(np.inner(y, Sigma @ y))
			rad_rat = np.sqrt(np.min(y_scores))
		# Adjust by rad_rat
		proj_to_vert[tuple(x)] = rad_rat*x

	# Instantiate GeodeGraph representation of child
	## Make Networkx graph_seed
	graph_seed = nx.Graph()
	n_simplex = delaunay_simps.shape[0]
	### Add nodes
	graph_seed.add_nodes_from(list(range(n_simplex)))
	### Add edges
	for j in range(n_simplex):
		neighbors = delaunay_neighbors[j, :]
		for neigh in neighbors:
			if j < neigh:
				graph_seed.add_edge(j, neigh)
	## Make node_to_simplex
	node_to_simplex = OrderedDict()
	for j in range(n_simplex):
		proj_face = delaunay_pts[delaunay_simps[j, :]]
		vert_face_list = []
		for k in range(proj_face.shape[0]):
			proj_pt = proj_face[k, :]
			proj_tup = tuple(proj_pt)
			vert_pt = proj_to_vert[proj_tup]
			vert_face_list.append(tuple(vert_pt))
		simplex = lexsort_list_of_tuples(vert_face_list)
		node_to_simplex[j] = simplex
	## GeodeGraph representation
	child_geode = GeodeGraph(graph_seed, node_to_simplex)

	## Adjust facet_dict to accomodate projected
	## coordinates versus original coordinates
	adj_facet_dict = OrderedDict()
	for node, facet in facet_dict.items():
		new_facet_pre = [vert_to_proj[vert] for vert in facet]
		adj_facet_dict[node] = lexsort_list_of_tuples(new_facet_pre)

	# Merge child and parent geodes
	merge_geodes_from_delaunay(child_geode, cut_geode, adj_facet_dict,
					delaunay.find_simplex,
					vert_to_proj,
					proj_to_vert,
					delaunay_pts)
	"""
	## Update self
	new_ell_ind = self.get_next_ell_ind()
	self.ind_to_ellipsoid[new_ell_ind] = (Sigma, center)
	for node in self.geode.G.nodes:
		if node > max_ind_pre:
			self.node_to_ind[node] = new_ell_ind
	"""

def solve_shrinking_quadratic(x, cp, Sigma_p):
	x_norm_sqr = np.inner(x, Sigma_p @ x)
	cp_norm_sqr = np.inner(cp, Sigma_p @ cp)
	mixed = np.inner(x, Sigma_p @ cp)
	t = (mixed - np.sqrt(mixed**2 - x_norm_sqr*(cp_norm_sqr - 1))) / x_norm_sqr
	return t

def geode_from_glass(glass_graph, center="origin"):
	"""
	Create GeodeGraph instance from GlassGraph instance.
	In terms of simplicial complexes, this is done by
	inserting the origin as a node in the GlassGraph and
	connecting it to all other nodes.
	"""
	# Get networkx graph from GlassGraph instance
	graph_seed = glass_graph.G.copy()

	# Get node_to_simplex dictionary from GlassGraph instance
	node_dict_pre = glass_graph.node_dict

	# Add center as vertex to all simplices in
	# node_dict_pre
	if center == "origin":
		center = tuple(np.zeros(glass_graph.d,))
	elif center == "mean":
		vertices = []
		for _, simplex in glass_graph.node_dict.items():
			for pt in simplex:
				if pt not in vertices:
					vertices.append(pt)
		center = tuple(np.mean(vertices, axis=0))
	else:
		raise ValueError("center must be either 'origin' or 'mean'.")
	node_dict = OrderedDict()
	for key, item in node_dict_pre.items():
		to_append_pre = item + [center]
		to_append = lexsort_list_of_tuples(to_append_pre)
		node_dict[key] = to_append

	return GeodeGraph(graph_seed, node_dict,
			boundary=set(node_dict.keys()))

def geode_from_function(geode_seed, fun, graph_type="geode"):
	"""
	Takes a GeodeGraph instance geode_seed whose vertices
	have dimension d and a function fun from R^d to R^D
	for some D. Returns the GeodeGraph instance that occurs
	when transforming all vertices according to fun.
	"""
	node_to_simplex = OrderedDict()
	for key, item in geode_seed.node_dict.items():
		new_simplex_pre = [fun(pt) for pt in item]
		new_simplex = lexsort_list_of_tuples(new_simplex_pre)
		node_to_simplex[key] = new_simplex
	if graph_type == "geode":
		return GeodeGraph(geode_seed.G.copy(), node_to_simplex,
				boundary=geode_seed.boundary.copy())
	elif graph_type == "glass":
		return GlassGraph(geode_seed.d, init_scheme="seeded",
				graph_seed=geode_seed.G.copy(),
				node_to_simplex=node_to_simplex,
				boundary=geode_seed.boundary.copy())
	else:
		raise ValueError("graph_type must be either 'geode' or 'glass'.")

def atlas_from_function(atlas_seed, fun_dict):
	# Check that atlas_seed.ind_to_ellipsoid keys
	# match with fun_dict keys
	for key in atlas_seed.ind_to_ellipsoid.keys():
		assert key in fun_dict
	for key in fun_dict.keys():
		assert key in atlas_seed.ind_to_ellipsoid

	node_to_simplex = OrderedDict()
	for key, item in atlas_seed.geode.node_dict.items():
		fun = fun_dict[atlas_seed.node_to_ind[key]]
		new_simplex_pre = [fun(pt) for pt in item]
		new_simplex = lexsort_list_of_tuples(new_simplex_pre)
		node_to_simplex[key] = new_simplex

	geode_seed = GeodeGraph(atlas_seed.geode.G.copy(), node_to_simplex)
	return AtlasGraph(geode_seed, atlas_seed.node_to_ind.copy(),
				atlas_seed.ind_to_ellipsoid.copy())

def translate_geode_graph(geode_graph, translation, graph_type="geode"):
	"""
	geode_graph should be an instance of type
	GeodeGraph; translation should be a NumPy
	array of type np.float64 such that
	geode_graph.d == len(translation).
	"""
	# Create translation function
	def trans_fun(pt):
		return tuple(np.array(pt) + translation)
	return geode_from_function(geode_graph, trans_fun,
			graph_type=graph_type)

def project_geode_graph(geode_graph, projection, graph_type="geode"):
	"""
	geode_graph should be an instance of type
	GeodeGraph; projection should be a NumPy
	array of type np.float64 and shape (D, d).
	It should hold that geode_graph.d == d.
	"""
	# Create projection function
	def proj_fun(pt):
		return tuple(projection @ np.array(pt))
	return geode_from_function(geode_graph, projection,
			graph_type=graph_type)

def dualize_for_geode_graph_extension(geode_graph,
		center_c, radius_c, center_p):
	"""
	geode_graph: GeodeGraph instance
	center_c: NumPy array of type np.float64,
		representing center of child sphere
	radius_c: positive np.float64 or float,
		representing radius of child sphere
	center_p: NumPy array of type np.float64,
		representing center of parent sphere

	NOTE: it should hold that
	geode_graph.d == len(center_c) == len(center_p)
	"""
	# Collect indices of simplices in geode_graph that
	# intersect the child sphere
	nodes_of_interest = []
	for ind in geode_graph.boundary:
		simplex = geode_graph.node_dict[ind]
		radii = np.array([np.linalg.norm(pt - center_c) for pt in simplex])
		conds = [radius < radius_c for radius in radii]
		if np.count_nonzero(conds) >= (geode_graph.k - 1):
			nodes_of_interest.append(ind)

	# Restrict these simplices to their outermost facet
	# ("outermost" being defined as "opposite vertex
	# closest to center_p").
	facet_dict = OrderedDict()
	for node in nodes_of_interest:
		face = geode_graph.node_dict[node]
		dists = [np.linalg.norm(pt - center_p) for pt in face]
		void_ind = np.argmin(dists)
		facet_dict[node] = [face[j] for j in range(len(face)) if j != void_ind]

	# Create graph showing connections among facets
	# NOTE: this is a brute-force, quadratic implementation
	# that should be replaced in the future
	n_facets = len(facet_dict)
	G = nx.Graph()
	G.add_nodes_from(nodes_of_interest)

	for j in range(n_facets):
		node_j = nodes_of_interest[j]
		facet_j = facet_dict[node_j]
		for k in range(j+1, n_facets):
			node_k = nodes_of_interest[k]
			facet_k = facet_dict[node_k]
			overlap = [pt for pt in facet_j if pt in facet_k]
			if (len(facet_j) - len(overlap)) == 1:
				sorted_edge = [node_j, node_k]
				sorted_edge.sort()
				first, last = sorted_edge
				G.add_edge(first, last)

	# Create primal GlassGraph object
	d = len(facet_k[0])
	primal_obj = GlassGraph(d, init_scheme="seeded",
			graph_seed=G.copy(),
			node_to_simplex=facet_dict.copy())

	# Collect vertices in boundary subfacets of
	# primal_obj
	boundary_verts = []
	for node in primal_obj.boundary:
		facet = primal_obj.node_dict[node]
		for pt in facet:
			subfacet = tuple(ptt for ptt in facet if ptt != pt)
			edge_subfacets = []
			for neigh in primal_obj.G.neighbors(node):
				sorted_edge = [node, neigh]
				sorted_edge.sort()
				first, last = sorted_edge
				edge_subfacets.append(tuple(primal_obj.edge_dict[(first, last)]))
			if subfacet not in edge_subfacets:
				for ptt in subfacet:
					if ptt not in boundary_verts:
						boundary_verts.append(ptt)

	## Collect vertices
	vertices = []
	for node in nodes_of_interest:
		facet = facet_dict[node]
		for pt in facet:
			if pt not in vertices:
				vertices.append(pt)

	# base_point will be a projection of a point onto
	# the line from one circle-center to the other.
	# The specific point to be projected will be
	# one whose projection is furthest from the center
	# of the child sphere.
	center_diff = center_p - center_c
	unit_diff = center_diff / np.linalg.norm(center_diff)
	max_dot_prod = 0.0
	for v in vertices:
		max_dot_prod = np.max([max_dot_prod, np.inner(unit_diff, np.array(v) - center_c)])
	base_point = center_c + max_dot_prod * unit_diff

	# Get conjugate point for each vertex.
	# If a vertex is in boundary_verts, then
	# it is self-conjugate
	conjugates = {}
	r_sqr = radius_c**2
	cmb = center_c - base_point
	cmb_norm_sqr = np.inner(cmb, cmb)
	for v in vertices:
		if v in boundary_verts:
			conjugates[v] = v
		else:
			vmb = v - base_point
			vmb_norm_sqr = np.inner(vmb, vmb)
			vmb_cmb = np.inner(vmb, cmb)
			# Get positive root of quadratic
			t = (np.inner(vmb, cmb) + np.sqrt(vmb_cmb**2 - vmb_norm_sqr*(cmb_norm_sqr - r_sqr))) / vmb_norm_sqr
			conjugates[v] = np.array(base_point + t*(v - base_point))

	# Create dual version of facet_dict, where index
	# scheme does note overlap with pre-existing index
	# scheme.
	# Also, save mapping from prior index scheme to
	# new index scheme.
	dual_dict = OrderedDict()
	scheme_map = {}
	for j, ind in enumerate(list(facet_dict.keys())):
		new_ind = j + primal_obj.next_name
		scheme_map[ind] = new_ind
		facet = facet_dict[ind]
		tuple_pre = tuple(conjugates[pt] for pt in facet)
		dual_dict[new_ind] = lexsort_list_of_tuples(tuple_pre)

	# Create dual GlassGraph object
	dual_obj = GlassGraph(d, init_scheme="seeded",
			graph_seed=nx.relabel_nodes(G, scheme_map),
			node_to_simplex=dual_dict)

	to_return = {}
	to_return["nodes_of_interest"] = nodes_of_interest
	to_return["facet_dict"] = facet_dict
	to_return["vertices"] = vertices
	to_return["conjugates"] = conjugates
	to_return["base_point"] = base_point
	to_return["primal_obj"] = primal_obj
	to_return["dual_obj"] = dual_obj

	return to_return

def merge_primal_dual_glass(primal_obj, dual_obj, conjugates):
	"""
	Merge primal and dual GlassGraph objects into one.
	Effectively, the primal object will be altered to
	represent the merging of the two graphs.
	"""
	# Add information from dual_obj attributes into
	# primal_obj attributes
	for key, item in dual_obj.node_dict.items():
		primal_obj.node_dict[key] = item
	for key, item in dual_obj.edge_dict.items():
		primal_obj.edge_dict[key] = item
	primal_obj.G.add_nodes_from(dual_obj.G)
	for u, v in dual_obj.G.edges:
		primal_obj.G.add_edge(u, v)

	# Collect subfacets of primal and dual objects
	primal_subfacets = {}
	for node in primal_obj.boundary:
		facet = primal_obj.node_dict[node]
		edge_subfacets = []
		for neigh in primal_obj.G.neighbors(node):
			sorted_edge = [node, neigh]
			sorted_edge.sort()
			first, last = sorted_edge
			edge_subfacets.append(tuple(primal_obj.edge_dict[(first, last)]))
		for pt in facet:
			subfacet = tuple(ptt for ptt in facet if ptt != pt)
			if subfacet not in edge_subfacets:
				primal_subfacets[subfacet] = node

	dual_subfacets = {}
	for node in dual_obj.boundary:
		facet = dual_obj.node_dict[node]
		edge_subfacets = []
		for neigh in dual_obj.G.neighbors(node):
			sorted_edge = [node, neigh]
			sorted_edge.sort()
			first, last = sorted_edge
			edge_subfacets.append(tuple(dual_obj.edge_dict[(first, last)]))
		for pt in facet:
			subfacet = tuple(ptt for ptt in facet if ptt != pt)
			if subfacet not in edge_subfacets:
				dual_subfacets[subfacet] = node

	# Glue together
	for subfacet, primal_node in primal_subfacets.items():
		dual_node = dual_subfacets[subfacet]
		primal_obj.G.add_edge(primal_node, dual_node)
		primal_obj.edge_dict[(primal_node, dual_node)] = subfacet

	# Update primal_obj.next_name
	primal_obj.next_name = np.max(primal_obj.G.nodes) + 1

	# Update boundary
	primal_obj.recompute_boundary()

def merge_geodes(child_geode, parent_geode, facet_dict):
	# Rename nodes in child_geode for merge into parent
	renaming = OrderedDict()
	for ind in child_geode.node_dict.keys():
		renaming[ind] = parent_geode.get_next_name()

	# Merge nodes in child_geode into parent
	## Add nodes
	for old_ind, new_ind in renaming.items():
		parent_geode.G.add_node(new_ind)
		parent_geode.node_dict[new_ind] = child_geode.node_dict[old_ind]
	## Add edges
	for old_ind, new_ind in renaming.items():
		for old_neigh in child_geode.G.neighbors(old_ind):
			sorted_old = [old_ind, old_neigh]
			sorted_old.sort()
			first_old, last_old = sorted_old
			new_neigh = renaming[old_neigh]
			sorted_new = [new_ind, new_neigh]
			sorted_new.sort()
			first_new, last_new = sorted_new
			parent_geode.G.add_edge(first_new, last_new)
			parent_geode.edge_dict[(first_new, last_new)] = child_geode.edge_dict[(first_old, last_old)]

	# Glue together
	# NOTE: cannot do geode updates on child
	# before merge
	for node, facet in facet_dict.items():
		new_node = renaming[node]
		parent_geode.G.add_edge(node, new_node)
		parent_geode.edge_dict[(node, new_node)] = facet

	# Update boundary
	parent_geode.recompute_boundary()

def merge_geodes_from_delaunay(child_geode, parent_geode, facet_dict,
				find_simplices,
				vert_to_proj,
				proj_to_vert,
				delaunay_pts):
	renaming = OrderedDict()
	for ind in child_geode.node_dict.keys():
		renaming[ind] = parent_geode.get_next_name()

	# Merge nodes in child geode into parent
	## Add nodes
	for old_ind, new_ind in renaming.items():
		parent_geode.G.add_node(new_ind)
		parent_geode.node_dict[new_ind] = child_geode.node_dict[old_ind]
	## Add edges
	for old_ind, new_ind in renaming.items():
		for old_neigh in child_geode.G.neighbors(old_ind):
			sorted_old = [old_ind, old_neigh]
			sorted_old.sort()
			first_old, last_old = sorted_old
			new_neigh = renaming[old_neigh]
			sorted_new = [new_ind, new_neigh]
			sorted_new.sort()
			first_new, last_new = sorted_new
			parent_geode.G.add_edge(first_new, last_new)
			parent_geode.edge_dict[(first_new, last_new)] = child_geode.edge_dict[(first_old, last_old)]

	# Glue together
	for node, facet in facet_dict.items():
		node_conj = find_simplices(np.vstack(facet))[0]
		# A returned value of -1 is effectively
		# a KeyError
		assert node_conj != -1
		new_node = renaming[node_conj]
		parent_geode.G.add_edge(node, new_node)
		parent_geode.edge_dict[(node, new_node)] = facet

	# Update boundary
	parent_geode.recompute_boundary()

def merge_disjoint_geodes(geode_a, geode_b):
	# Rename nodes in geode_b for merge into geode_a
	renaming = OrderedDict()
	for ind in geode_b.node_dict.keys():
		renaming[ind] = geode_a.get_next_name()

	# Merge nodes in child_geode into parent
	## Add nodes
	for old_ind, new_ind in renaming.items():
		geode_a.G.add_node(new_ind)
		geode_a.node_dict[new_ind] = geode_b.node_dict[old_ind]
	## Add edges
	for old_ind, new_ind in renaming.items():
		for old_neigh in geode_b.G.neighbors(old_ind):
			sorted_old = [old_ind, old_neigh]
			sorted_old.sort()
			first_old, last_old = sorted_old
			new_neigh = renaming[old_neigh]
			sorted_new = [new_ind, new_neigh]
			sorted_new.sort()
			first_new, last_new = sorted_new
			geode_a.G.add_edge(first_new, last_new)
			geode_a.edge_dict[(first_new, last_new)] = geode_b.edge_dict[(first_old, last_old)]

	# Update boundary
	geode_a.recompute_boundary()

def chakraborty_express(X, Y, t):
	"""
	X and Y are N x d Stiefel matrices (i.e. their columns are
	orthonormal). This function computes the "thin" SVD of
	(1 - X@X.T)@Y@(X.T @ Y)^{-1}. It then uses this information
	to translate along a geodesic of the Grassmannian according
	to eqn 4 of Chakraborty et al. 2017

	For context, see between eqns 4 and 5 from Chakraborty et al.
	2017

	NOTE: It must be the case that X.T @ Y is invertible
	"""
	# Get terms A and B of the matrix product A@Y@B in the
	# doctring
	N, d = X.shape	# X,Y should have same shape
	I = np.eye(N)	# N x N identity matrix
	A = I - (X @ X.T)
	B = np.linalg.inv(X.T @ Y)

	# Compute SVD of matrix product
	P = A @ Y @ B
	U, s, Vh = np.linalg.svd(P)

	# Compute matrix of angle arctangents
	Theta = np.diag(np.arctan(s))

	# Move length t along geodesic from X to Y on Grassmann
	cos_term = X @ Vh.T
	cos_term_complete = cos_term @ np.cos(Theta*t)
	sin_term_complete = U[:, :d] @ np.sin(Theta*t)
	new_mat = cos_term_complete + sin_term_complete

	# Perform Gram-Schmidt on columns of new_mat
	Q, _ = np.linalg.qr(new_mat)

	return Q

def iga(stiefel_list):
	"""
	Performs IGA from Chakraborty et al. Each matrix in stiefel_list must
	be a Stiefel matrix, and all Stiefel matrices must have same shape.

	This method performs an online approximation of the Grassmann average
	using iterated calls to chakraborty_express
	"""
	for j, Y in enumerate(stiefel_list):
		if j == 0:
			X = Y
		else:
			X = chakraborty_express(X, Y, (j+1)**(-1))
	return X

def arccos_catch_nan_scalar(scalar):
	"""
	Implement numpy.arccos on a scalar value, except catch NaNs and
	convert to zero
	"""
	temp = np.arccos(scalar)
	return 0 if np.isnan(temp) else temp

# Implement numpy.arccos on an array, except catch NaNs and convert
# to zero
arccos_catch_nan = np.vectorize(arccos_catch_nan_scalar)

# Exception to catch when LazyNeretinConvolution
# reaches all points
class NeretinEndError(Exception):
	pass

# Exception to catch when LazyNeretinConvolution
# is not doing anything statistically significant
class NeretinSmallError(Exception):
	pass

class LazyNeretinConvolution:
	def __init__(self, X, x_0, d,
			start, step, hwl):
		# Save attributes from args
		self.X = X
		self.N, self.D = self.X.shape
		self.x_0 = x_0
		self.d = d
		self.start = start
		self.step = step
		self.hwl = hwl

		# Sort points by pairwise distance from self.x_0
		X_0 = self.x_0.reshape(1, self.D)
		self.dist_vec = euclidean_distances(X_0, self.X)[0, :]
		self.inds = list(range(self.N))
		self.inds.sort(key=lambda x: self.dist_vec[x])

		# Create system for storing tangent approximations
		self.tan_apps = []
		self.n_pts = 0

		# Create system for storing pairwise Grassmann distances
		self.distances = {}

	def get_effective_index(self, ind):
		return self.start + ind*self.step

	def __getitem__(self, ind):
		if ind < self.n_pts:
			return self.tan_apps[ind]
		elif ind == self.n_pts:
			# Compute effective index
			ind_eff = self.get_effective_index(ind)
			if ind_eff >= self.N:
				raise NeretinEndError("ind_eff has exceeded number of points.")

			# Look at points with index up to ind_eff
			X_prime = self.X[self.inds[:ind_eff], :]
			cov_mat = np.cov(X_prime, rowvar=False)
			_, V = np.linalg.eigh(cov_mat)
			tan_app = V[:, -self.d:]
			self.tan_apps.append(tan_app)
			self.n_pts += 1
			return tan_app
		else:
			raise KeyError("ind must be less than or equal to self.n_pts. The input value of ind was "+str(ind)+"; the value of self.n_pts was "+str(self.n_pts)+".")

	def clear(self):
		self.tan_apps = []
		self.distances.clear()
		self.n_pts = 0

	def get_Grassmann_dist(self, ind_j, ind_k):
		if (ind_j > ind_k):
			assert ValueError("ind_j must be less than or equal to ind_k.")
		elif (ind_j == ind_k):
			self.distances[(ind_j, ind_k)] = 0.0
			return 0.0
		else:
			try:
				return self.distances[(ind_j, ind_k)]
			except KeyError:
				L_j = self[ind_j]
				L_k = self[ind_k]
				Lambda, _ = np.linalg.eig(L_j.T @ L_k)
				dist = np.sum(np.arccos(Lambda)**2)
				self.distances[(ind_j, ind_k)] = dist
				return dist

	def convolution_til_bust(self):
		two_hwl_p1 = 2*self.hwl + 1
		cond_outer = True
		n_std = 1.0
		while cond_outer:
			ind = self.hwl
			for j in range(ind + 1):
				self[j]
			old_mean = np.nan
			old_std = np.nan
			cond = True
			while cond:
				try:
					dists = []
					for j in range(ind - self.hwl, ind + self.hwl + 1):
						if j <= ind:
							lesser = j
							greater = ind
						else:
							lesser = ind
							greater = j
						dists.append(self.get_Grassmann_dist(lesser, greater))
					moving_mean = np.mean(dists)
					moving_std = np.std(dists)
					if not np.isnan(old_mean):
						if moving_mean >= old_mean + n_std*old_std:
							cond = False
							cond_outer = False
					old_mean = moving_mean
					old_std = moving_std
					ind += 1
				except NeretinEndError:
					self.clear()
					n_std *= 0.5
					cond = False
		if n_std < 0.1:
			raise NeretinSmallError("Shrunk to too few standard deviations.")
		return ind

def rooted_tan_plane_conformity(L, L_prime, p, p_prime):
	W = L.T @ L_prime
	Lambda_sqr, _ = np.linalg.eigh(W @ W.T)
	Lambda = np.sqrt(Lambda_sqr)
	# Trim entries of Lambda to 1.0 or below to avoid
	# error in the arccos call
	for j, val in enumerate(Lambda):
		if val > 1.0:
			Lambda[j] = 1.0
	grass_dist = np.sum(np.arccos(Lambda)**2)
	dist_vec = p - p_prime
	unit_dist_vec = dist_vec/np.linalg.norm(dist_vec)
	membership = np.sqrt(np.sum((L.T @ unit_dist_vec)**2))
	membership_prime = np.sqrt(np.sum((L_prime.T @ unit_dist_vec)**2))

	conformity = {}
	conformity["grass_dist"] = grass_dist
	conformity["membership"] = membership
	conformity["membership_prime"] = membership_prime
	return conformity
