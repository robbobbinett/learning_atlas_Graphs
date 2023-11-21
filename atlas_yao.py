import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from manifold_utils import get_quadratic_and_const_terms, h_mat_to_trilinear, naive_approximate_dist
from atlas_pam import *

def gaussian_kernel(x):
	return np.exp(-x**2)

# Value below from deterministic atlas from paper 1
bias_vec_amb = np.array([ 2.,  0., -2.,  0.,  0.,  0., -2., -0.,  2.])

class atlas_yao(atlas_pam):
	def __init__(self, X, X_pos, X_neg, n_charts, km_max_iter=1000,
			kernel_fun=gaussian_kernel, grid_len=100):
		# Make sure all three datasets are nine-dimensional
		Xs = [X, X_pos, X_neg]
		for XX in Xs:
			assert XX.shape[1] == 9
		# Initialize using parent constructor
		super().__init__(X, 2, n_charts)
		# Save attributes respective to class data
		self.X_pos = X_pos
		self.X_neg = X_neg
		self.N_pos = X_pos.shape[0]
		self.N_neg = X_neg.shape[0]
		self.chart_assignments_pos = np.zeros(self.N_pos,
						dtype=int)
		self.chart_assignments_neg = np.zeros(self.N_neg,
						dtype=int)
		for j in range(self.N_pos):
			x = X_pos[j, :]
			self.chart_assignments_pos[j] = self.identify_chart(x)
		for j in range(self.N_neg):
			x = X_neg[j, :]
			self.chart_assignments_neg[j] = self.identify_chart(x)
		### Dicts of coordinate chart representations
		self.Xi_pos_dict = {}
		self.Xi_neg_dict = {}
		for j in range(n_charts):
			Xi_pos = np.zeros((self.N_pos, 2))
			Xi_neg = np.zeros((self.N_neg, 2))
			for k in range(self.N_pos):
				x = X_pos[k, :]
				Xi_pos[k, :] = self.ingest_ambient_point_given_ind(x, j)
			for k in range(self.N_neg):
				x = X_neg[k, :]
				Xi_neg[k, :] = self.ingest_ambient_point_given_ind(x, j)
			self.Xi_pos_dict[j] = Xi_pos
			self.Xi_neg_dict[j] = Xi_neg
		# Create enormous graph that comprises a fine mesh of
		# the union of the images of the coordinate charts
		print("Constructing enormous graph for brute-force geodesic approximation.")
		# TODO: enormous graph construction function
		cond = False
		eps = 0.6
		while not cond:
			print("Trying eps = "+str(eps))
			self.X_brute, self.G_brute = self.construct_enormous_graph(grid_len=grid_len,
							eps=eps)
			if nx.is_connected(self.G_brute):
				cond = True
			eps += 0.1
		n_brute = self.X_brute.shape[0]
		self.chart_assignments_brute = np.zeros(n_brute,
						dtype=int)
		for j in range(n_brute):
			x = self.X_brute[j, :]
			self.chart_assignments_brute[j] = self.identify_chart(x)
		self.Xi_brute_dict = {}
		for j in range(n_charts):
			Xi_brute = np.zeros((n_brute, 2))
			for k in range(n_brute):
				x = self.X_brute[k, :]
				Xi_brute[k, :] = self.ingest_ambient_point_given_ind(x, j)
			self.Xi_brute_dict[j] = Xi_brute
		# Save K
		self.K_dict = {}
		for ind in range(n_charts):
			_, _, _, _, h_mat = self.chart_dict[ind]
			self.K_dict[ind] = h_mat_to_trilinear(h_mat)
		# Create cache of shortest paths
		self.shortest_path_cache = {}
		# Cache kernel function
		self.kernel = kernel_fun

	def construct_enormous_graph(self, grid_len=100, eps=0.1):
		# Get points for graph vertices
		X_pre = []
		for ind in tqdm(range(self.n_charts)):
			XX = self.sample_uniformly_from_chart_by_ind(ind,
					grid_len=grid_len)
			X_pre.append(XX)
		X = np.vstack(X_pre)
		n_brute = X.shape[0]
		# Construct graph
		G = nx.Graph()
		for j in tqdm(range(n_brute)):
			X_j = X[j, :].reshape((1, 9))
			G.add_node(j)
			dist_vec = euclidean_distances(X_j, X)[0, :]
			for k in range(j+1, n_brute):
				dist = dist_vec[k]
				if dist <= eps:
					G.add_edge(j, k, weight=dist)
		return X, G

	def evaluate_pos_flow_spectrum(self, xi_0, chart, dist_max=1.0):
		# Prune away points too far by ambient metric
		x_0 = self.xi_ind_to_ambient(xi_0, chart)
		X_0 = x_0.reshape(1, 9)
		dist_vec = euclidean_distances(X_0, self.X_pos)[0, :]
		to_keep = (dist_vec <= dist_max)
		# Aggregate logarithms
		xi_prime_list = []
		for j in range(self.N_pos):
			if to_keep[j]:
				chart_otra = self.chart_assignments_pos[j]
				xi_otra = self.Xi_pos_dict[chart_otra][j, :]
				xi_prime, dist = self.logarithm_through_graph(xi_0,
							chart, xi_otra,
							chart_otra)
				if dist <= dist_max:
					kern_coeff = self.kernel(dist)
					xi_prime_list.append(kern_coeff*xi_prime)
		try:
			Xi_prime = np.vstack(xi_prime_list)
			cov_mat = np.cov(Xi_prime, rowvar=False)
			_, V = np.linalg.eigh(cov_mat)
			eig_vec = V[:, 1]
			to_return = eig_vec
		except (ValueError, np.linalg.LinAlgError) as error:
			to_return = np.random.randn(2)
			#raise ValueError(str(xi_prime_list))
		return to_return, np.mean(Xi_prime, axis=0)

	def evaluate_neg_flow_spectrum(self, xi_0, chart, dist_max=1.0):
		# Prune away points too far by ambient metric
		x_0 = self.xi_ind_to_ambient(xi_0, chart)
		X_0 = x_0.reshape(1, 9)
		dist_vec = euclidean_distances(X_0, self.X_neg)[0, :]
		to_keep = (dist_vec <= dist_max)
		# Aggregate logarithms
		xi_prime_list = []
		for j in range(self.N_neg):
			if to_keep[j]:
				chart_otra = self.chart_assignments_neg[j]
				xi_otra = self.Xi_neg_dict[chart][j, :]
				xi_prime, dist = self.logarithm_through_graph(xi_0,
							chart, xi_otra,
							chart_otra)
				if dist <= dist_max:
					kern_coeff = self.kernel(dist)
					xi_prime_list.append(kern_coeff*xi_prime)
		try:
			Xi_prime = np.vstack(xi_prime_list)
			cov_mat = np.cov(Xi_prime, rowvar=False)
			_, V = np.linalg.eigh(cov_mat)
			eig_vec = V[:, 1]
			to_return = eig_vec
		except (ValueError, np.linalg.LinAlgError) as error:
			to_return = np.random.randn(2)
			#raise ValueError(str(xi_prime_list))
		return to_return, np.mean(Xi_prime, axis=0)

	def logarithm_through_graph(self, xi_0, chart_0, xi_1, chart_1):
		dist_0, dist_1, path, path_len = self.approximate_shortest_path(xi_0,
							chart_0, xi_1,
							chart_1)
		xi_prime = np.zeros(2)
		xi_dequeue = xi_1.copy()
		chart_dequeue = chart_1
		dist = dist_1
		while len(path) > 0:
			ind_queue = path.pop()
			chart_queue = self.chart_assignments_brute[ind_queue]
			xi_queue = self.Xi_brute_dict[chart_queue][ind_queue, :]
			if chart_dequeue == chart_queue:
				xi_prime += xi_dequeue - xi_queue
				K_queue = self.K_dict[chart_queue]
				dist += naive_approximate_dist(xi_queue, xi_dequeue, K_queue)
				xi_dequeue = xi_queue
			else:
				x = self.xi_ind_to_ambient(xi_dequeue, chart_dequeue)
				amb_vec = self.xi_xi_prime_chart_to_meta_tan_plane(xi_dequeue,
						xi_prime, chart_dequeue)
				xi_prime = self.x_amb_vec_chart_to_meta_tan_project(x,
						amb_vec, chart_queue)
				center_queue, L_queue, _, _, _ = self.chart_dict[chart_queue]
				xi_dequeue_alt = L_queue.T @ (x - center_queue)
				xi_prime += xi_dequeue_alt - xi_queue
				K_queue = self.K_dict[chart_queue]
				dist += naive_approximate_dist(xi_queue,
							xi_dequeue,
							K_queue)
				xi_dequeue = xi_queue
				chart_dequeue = chart_queue
		xi_prime += xi_dequeue - xi_0
		dist += dist_0
		return xi_prime, dist

	def approximate_shortest_path(self, xi_0, chart_0, xi_1, chart_1):
		x_0 = self.xi_ind_to_ambient(xi_0, chart_0)
		x_1 = self.xi_ind_to_ambient(xi_1, chart_1)
		ind_0 = self.find_closest_brute_point(x_0)
		ind_1 = self.find_closest_brute_point(x_1)
		x_prime_0 = self.X_brute[ind_0, :]
		x_prime_1 = self.X_brute[ind_1, :]
		dist_0 = np.linalg.norm(x_0 - x_prime_0)
		dist_1 = np.linalg.norm(x_1 - x_prime_1)
		path_pre, path_len = self.shortest_path_cached(ind_0, ind_1)
		path = path_pre.copy()
		return dist_0, dist_1, path, path_len

	def shortest_path_cached(self, ind_0, ind_1):
		key = (ind_0, ind_1)
		try:
			return self.shortest_path_cache[key]
		except KeyError:
			path = nx.shortest_path(self.G_brute,
					weight="weight",
					source=ind_0,
					target=ind_1)
			path_len = nx.shortest_path_length(self.G_brute,
					weight="weight",
					source=ind_0,
					target=ind_1)
			value = (path, path_len)
			self.shortest_path_cache[key] = value
			return value

	def find_closest_brute_point(self, x):
		X = x.reshape((1, 9))
		dists = euclidean_distances(X, self.X_brute)[0, :]
		return np.argmin(dists)

	def xi_xi_prime_chart_to_meta_tan_plane(self, xi, xi_prime, chart):
		# Compute gradient of normal components
		normal_grads = []
		_, L, M, _, _ = self.chart_dict[chart]
		K = self.K_dict[chart]
		for j in range(7):
			K_mat = K[:, :, j]
			normal_grads.append(K_mat @ xi)
		normal_mat_pre = np.vstack(normal_grads)
		normal_mat = 2*M @ normal_mat_pre
		return (normal_mat + L) @ xi_prime

	def x_amb_vec_chart_to_meta_tan_project(self, x, amb_vec, chart):
		center, L, M, _, _ = self.chart_dict[chart]
		K = self.K_dict[chart]
		xi = L.T @ (x - center)
		normal_grads = []
		for j in range(7):
			K_mat = K[:, :, j]
			normal_grads.append(K_mat @ xi)
		normal_mat_pre = np.vstack(normal_grads)
		normal_mat = 2*M @ normal_mat_pre
		jac_mat = normal_mat + L
		jac_mat_pinv = np.linalg.pinv(jac_mat)
		return jac_mat_pinv @ amb_vec

	def riemannian_principal_boundary(self, xi_init_pos, chart_init_pos,
					xi_init_neg, chart_init_neg,
					stepsize=0.1, n_iters=500,
					dist_max=1.0):
		### TODO: bias_vec_amb
		# Initialization for ideal flows
		### Positive flow
		xi_pos = xi_init_pos.copy()
		chart_pos = chart_init_pos
		V_pos, _ = self.evaluate_pos_flow_spectrum(xi_pos,
					chart_pos, dist_max=dist_max)
		xi_prime_pos = V_pos[:, 1]
		##### Align initial vector
		amb_vec_pos = self.xi_xi_prime_chart_to_meta_tan_plane(xi_pos,
					xi_prime_pos, chart_pos)
		if np.dot(amb_vec_pos, bias_vec_amb) < 0:
			xi_prime_pos *= -1
		xi_pos_list = [xi_pos.copy()]
		chart_pos_list = [chart_pos]
		xi_prime_pos_list = [xi_prime_pos.copy()]
		amb_prev_pos = amb_vec_pos
		### Negative flow
		xi_neg = xi_init_neg.copy()
		chart_neg = chart_init_neg
		V_neg, _ = self.evaluate_neg_flow_spectrum(xi_neg,
					chart_neg, dist_max=dist_max)
		xi_prime_neg = V_neg[:, 1]
		##### Align initial vector
		amb_vec_neg = self.xi_xi_prime_chart_to_meta_tan_plane(xi_neg,
					xi_prime_neg, chart_neg)
		if np.dot(amb_vec_neg, bias_vec_amb) < 0:
			xi_prime_neg *= -1
		xi_neg_list = [xi_neg.copy()]
		chart_neg_list = [chart_neg]
		xi_prime_neg_list = [xi_prime_neg.copy()]
		amb_prev_neg = amb_vec_neg
		# Initialization for principal boundary
		xi_bou, chart_bou = self.approximate_frechet_mean(xi_pos,
					chart_pos, xi_neg, chart_neg)
		xi_prime_bou_pos = self.vector_transport_through_graph(xi_pos,
					chart_pos, xi_bou, chart_bou,
					xi_prime_pos)
		xi_prime_bou_neg = self.vector_transport_through_graph(xi_neg,
					chart_neg, xi_bou, chart_bou,
					xi_prime_neg)
		xi_prime_bou = (xi_prime_bou_pos + xi_prime_bou_neg) / 2
		xi_bou_list = [xi_bou.copy()]
		chart_bou_list = [chart_bou]
		xi_prime_bou_list = [xi_prime_bou.copy()]
		xi_prime_bou_pos_list = [xi_prime_bou_pos.copy()]
		xi_prime_bou_neg_list = [xi_prime_bou_neg.copy()]
		boundary_fun = self.boundary_fun_dict[chart_bou]
		# Perform iterations
		for j in tqdm(range(n_iters)):
			# Perform first steps
			xi_pos, chart_pos, amb_prev_pos, xi_prime_pos = self.principal_flow_pos_iter(xi_pos,
						chart_pos, stepsize,
						dist_max=dist_max,
						amb_prev=amb_prev_pos)
			xi_neg, chart_neg, amb_prev_neg, xi_prime_neg = self.principal_flow_neg_iter(xi_neg,
						chart_neg, stepsize,
						dist_max=dist_max,
						amb_prev=amb_prev_neg)
			# Yao iteration
			xi_prime_bou_pos = self.vector_transport_through_graph(xi_pos,
						chart_pos, xi_bou,
						chart_bou, xi_prime_pos)
			xi_prime_bou_neg = self.vector_transport_through_graph(xi_neg,
						chart_neg, xi_bou,
						chart_bou, xi_prime_neg)
			lam = self.get_optimal_lambda(xi_bou,
						xi_prime_bou,
						xi_prime_bou_pos,
						xi_prime_bou_neg,
						chart_bou)
			xi_prime_bou = lam*xi_prime_bou_pos + (1 - lam)*xi_prime_bou_neg
			xi_bou = xi_bou + stepsize*xi_prime_bou
			### If boundary function transgressed, change
			### coordinate charts
			f = boundary_fun(xi_bou)
			if f >= 0:
				center, L, M, _, _ = self.chart_dict[chart_bou]
				x = self.xi_ind_to_ambient(xi_bou,
						chart_bou)
				dists = []
				for chart_bou_prime in range(self.n_charts):
					center_prime, _, _, _, _ = self.chart_dict[chart_bou_prime]
					dist = np.linalg.norm(x - center_prime)
					dists.append(dist)
				chart_bou = np.argmin(dists)
				center, L, _, _, _ = self.chart_dict[chart_bou]
				xi_bou = L.T @ (x - center)
				boundary_fun = self.boundary_fun_dict[chart_bou]
			# Store iterate
			xi_pos_list.append(xi_pos.copy())
			chart_pos_list.append(chart_pos)
			xi_prime_pos_list.append(xi_prime_pos.copy())
			xi_neg_list.append(xi_neg.copy())
			chart_neg_list.append(chart_neg)
			xi_prime_neg_list.append(xi_prime_neg.copy())
			xi_bou_list.append(xi_bou.copy())
			chart_bou_list.append(chart_bou)
			xi_prime_bou_list.append(xi_prime_bou.copy())
			xi_prime_bou_pos_list.append(xi_prime_bou_pos.copy())
			xi_prime_bou_neg_list.append(xi_prime_bou_neg.copy())
		return xi_pos_list, chart_pos_list, xi_prime_pos_list, xi_neg_list, chart_neg_list, xi_prime_neg_list, xi_bou_list, chart_bou_list, xi_prime_bou_list, xi_prime_bou_pos_list, xi_prime_bou_neg_list

	def approximate_frechet_mean(self, xi_0, chart_0, xi_1, chart_1):
		dist_0, dist_1, path, path_len = self.approximate_shortest_path(xi_0,
							chart_0, xi_1,
							chart_1)
		if len(path) <= 2:
			x_temp = self.xi_ind_to_ambient(xi_1, chart_1)
			center_0, L_0, _, _, _ = self.chart_dict[chart_0]
			xi_new = L_0.T @ (x_temp - center_0)
			xi_mean = (xi_0 + xi_new)/2
			# Recompute xi_mean's chart assignment
			x_mean = self.xi_ind_to_ambient(xi_new, chart_0)
			chart_mean = identify_chart(x_mean)
			center_mean, L_mean, _, _, _ = self.chart_dict[chart_mean]
			xi_mean_final = L_mean.T @ (x_mean - center_mean)
			return xi_mean_final, chart_mean
		else:
			pl = len(path)
			mid_ind_ind = int(np.floor(pl/2))
			mid_ind = path[mid_ind_ind]
			x_mid = self.X_brute[mid_ind, :]
			chart_mid = self.identify_chart(x_mid)
			center_mid, L_mid, _, _, _ = self.chart_dict[chart_mid]
			xi_mid = L_mid.T @ (x_mid - center_mid)
			return xi_mid, chart_mid

	def vector_transport_through_graph(self, xi_0, chart_0, xi_1,
						chart_1, xi_prime):
		dist_0, dist_1, path, path_length = self.approximate_shortest_path(xi_0,
							chart_0, xi_1,
							chart_1)
		xi_prime_temp = xi_prime.copy()
		xi_dequeue = xi_0.copy()
		chart_dequeue = chart_0
		while len(path) > 0:
			ind_queue = path.pop(0)
			chart_queue = self.chart_assignments_brute[ind_queue]
			xi_queue = self.Xi_brute_dict[chart_queue][ind_queue, :]
			xi_prime_temp = self.vector_transport_hop(xi_dequeue,
						chart_dequeue, xi_queue,
						chart_queue,
						xi_prime_temp)
			xi_dequeue = xi_queue.copy()
			chart_dequeue = chart_queue
		xi_prime_temp = self.vector_transport_hop(xi_dequeue,
						chart_dequeue, xi_1,
						chart_1,
						xi_prime_temp)
		return xi_prime_temp

	def vector_transport_hop(self, xi_0, chart_0, xi_1, chart_1,
					xi_prime):
		if chart_0 == chart_1:
			return xi_prime
		else:
			amb_vec = self.xi_xi_prime_chart_to_meta_tan_plane(xi_0,
					xi_prime, chart_0)
			_, L_1, _, _, _ = self.chart_dict[chart_1]
			to_return  = L_1.T @ amb_vec
			return to_return

	def principal_flow_pos_iter(self, xi, chart, stepsize,
					amb_prev=None, dist_max=1.0):
		V, xi_prime_mean = self.evaluate_pos_flow_spectrum(xi,
					chart, dist_max=dist_max)
		if isinstance(amb_prev, NoneType):
			xi_prime = V[:, 1]
			eig_vec_otra = V[:, 0]
			eig_vec_otra_proj = np.outer(eig_vec_otra,
						eig_vec_otra)
			amb_prev = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						xi_prime, chart)
			xi += stepsize*(xi_prime + eig_vec_otra_proj@xi_prime_mean)
		else:
			eig_vec_0 = V[:, 0]
			eig_vec_1 = V[:, 1]
			amb_0 = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						eig_vec_0, chart)
			amb_1 = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						eig_vec_1, chart)
			inner_0 = np.inner(amb_prev, amb_0)
			inner_1 = np.inner(amb_prev, amb_1)
			c = 5.0
			if np.abs(inner_0) >= np.abs(inner_1):
				amb = amb_0
				inner = inner_0
				xi_prime = eig_vec_0
				eig_vec_otra_proj = np.outer(eig_vec_1,
						eig_vec_1)
			else:
				amb = amb_1
				inner = inner_1
				xi_prime = eig_vec_1
				eig_vec_otra_proj = np.outer(eig_vec_0,
						eig_vec_0)
			if inner <= 0:
				xi_prime = -xi_prime + c*eig_vec_otra_proj@xi_prime_mean
				xi += stepsize*xi_prime
				amb_prev = -amb.copy()
			else:
				xi_prime = xi_prime + c*eig_vec_otra_proj@xi_prime_mean
				xi += stepsize*xi_prime
				amb_prev = amb.copy()
		# If boundary function transgressed, change coordinate
		# charts
		boundary_fun = self.boundary_fun_dict[chart]
		f = boundary_fun(xi)
		if f >= 0:
			center, L, M, _, h_mat = self.chart_dict[chart]
			xi_quad = get_quadratic_and_const_terms(xi)
			x = (M@h_mat.T@xi_quad) + (L@xi) + center
			dists = []
			xi_prime_pre = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						xi_prime, chart)
			for chart_prime in range(self.n_charts):
				center_prime, _, _, _, _ = self.chart_dict[chart_prime]
				dist = np.linalg.norm(x - center_prime)
				dists.append(dist)
			chart = np.argmin(dists)
			center, L, _, _, _ = self.chart_dict[chart]
			xi = L.T @ (x - center)
			xi_prime = self.x_amb_vec_chart_to_meta_tan_project(x,
						xi_prime_pre, chart)
		return xi, chart, amb_prev, xi_prime

	def principal_flow_neg_iter(self, xi, chart, stepsize,
					amb_prev=None, dist_max=1.0):
		V, xi_prime_mean = self.evaluate_neg_flow_spectrum(xi,
					chart, dist_max=dist_max)
		if isinstance(amb_prev, NoneType):
			xi_prime = V[:, 1]
			eig_vec_otra = V[:, 0]
			eig_vec_otra_proj = np.outer(eig_vec_otra,
						eig_vec_otra)
			amb_prev = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						xi_prime, chart)
			xi += stepsize*(xi_prime + eig_vec_otra_proj@xi_prime_mean)
		else:
			eig_vec_0 = V[:, 0]
			eig_vec_1 = V[:, 1]
			amb_0 = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						eig_vec_0, chart)
			amb_1 = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						eig_vec_1, chart)
			inner_0 = np.inner(amb_prev, amb_0)
			inner_1 = np.inner(amb_prev, amb_1)
			c = 5.0
			if np.abs(inner_0) >= np.abs(inner_1):
				amb = amb_0
				inner = inner_0
				xi_prime = eig_vec_0
				eig_vec_otra_proj = np.outer(eig_vec_1,
						eig_vec_1)
			else:
				amb = amb_1
				inner = inner_1
				xi_prime = eig_vec_1
				eig_vec_otra_proj = np.outer(eig_vec_0,
						eig_vec_0)
			if inner <= 0:
				xi_prime = -xi_prime + c*eig_vec_otra_proj@xi_prime_mean
				xi += stepsize*xi_prime
				amb_prev = -amb.copy()
			else:
				xi_prime = xi_prime + c*eig_vec_otra_proj@xi_prime_mean
				xi += stepsize*xi_prime
				amb_prev = amb.copy()
		# If boundary function transgressed, change coordinate
		# charts
		boundary_fun = self.boundary_fun_dict[chart]
		f = boundary_fun(xi)
		if f >= 0:
			center, L, M, _, h_mat = self.chart_dict[chart]
			xi_quad = get_quadratic_and_const_terms(xi)
			x = (M@h_mat.T@xi_quad) + (L@xi) + center
			dists = []
			xi_prime_pre = self.xi_xi_prime_chart_to_meta_tan_plane(xi,
						xi_prime, chart)
			for chart_prime in range(self.n_charts):
				center_prime, _, _, _, _ = self.chart_dict[chart_prime]
				dist = np.linalg.norm(x - center_prime)
				dists.append(dist)
			chart = np.argmin(dists)
			center, L, _, _, _ = self.chart_dict[chart]
			xi = L.T @ (x - center)
			xi_prime = self.x_amb_vec_chart_to_meta_tan_project(x,
						xi_prime_pre, chart)
		return xi, chart, amb_prev, xi_prime

	def get_optimal_lambda(self, xi, xi_prime_0, xi_prime_1,
				xi_prime_2, chart):
		return 0.5
