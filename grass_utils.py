import numpy as np
from tqdm import tqdm

def random_sample_grass_as_stiefel(n, k, n_pts=1000, seed=482):
	# Set random seed
	np.random.seed(seed)
	# Collect Stiefel matrices
	Xs = []
	for _ in range(n_pts):
		X_pre = np.random.randn(n, k)
		X, _ = np.linalg.qr(X_pre)
		Xs.append(X)

	return Xs

def stiefel_list_to_projector_list(Xs):
	Ps = []
	for X in tqdm(Xs):
		Ps.append(X @ X.T)
	return Ps

def commutation_matrix(m, n):
	mn = m * n
	K = np.zeros((mn, mn))
	for j in range(m):
		for k in range(n):
			K[k + n*j, j + m*k] = 1.0
	return K

class commutation_matrix_dict:
	def __init__(self):
		self.book = {}

	def __getitem__(self, key):
		try:
			return self.book[key]
		except KeyError:
			m, n = key
			K = commutation_matrix(m, n)
			self.book[(m, n)] = K
			return K

def yatta_inv_from_A(A):
	inner = A.T @ A
	lam, V = np.linalg.eigh(inner)
	term_inv = 1/(1 + lam)
	return V @ np.diag(term_inv) @ V.T

cm_dict = commutation_matrix_dict()

def sqrt_det_metric_from_A(A):
	# Terms pertinent to all blocks
	m, n = A.shape
	Q = cm_dict[(m, n)]
	yatta_inv = yatta_inv_from_A(A)
	a_yatta_inv = A @ yatta_inv
	yatta_inv_at = yatta_inv @ A.T
	a_yatta_inv_at = A @ yatta_inv_at
	I_m = np.eye(m)
	I_n = np.eye(n)
	# M_ul
	left = -np.kron(yatta_inv_at, yatta_inv) @ Q
	right = -np.kron(yatta_inv, yatta_inv_at)
	M_ul = left + right
	# M_ur
	left = -np.kron(a_yatta_inv_at, yatta_inv) @ Q
	center = -np.kron(a_yatta_inv, yatta_inv_at)
	right = np.kron(I_m, yatta_inv) @ Q
	M_ur = left + center + right
	# M_ll
	left = -np.kron(yatta_inv_at, a_yatta_inv) @ Q
	center = -np.kron(yatta_inv, a_yatta_inv_at)
	right = np.kron(yatta_inv, I_m)
	M_ll = left + center + right
	# M_lr
	one = np.kron(a_yatta_inv, I_m)
	two = np.kron(I_m, a_yatta_inv) @ Q
	three = -np.kron(a_yatta_inv_at, a_yatta_inv) @ Q
	four = -np.kron(a_yatta_inv, a_yatta_inv_at)
	M_lr = one + two + three + four
	# Get inner products
	M_ul_inner = M_ul.T @ M_ul
	M_ur_inner = M_ur.T @ M_ur
	M_ll_inner = M_ll.T @ M_ll
	M_lr_inner = M_lr.T @ M_lr
	M_inners = [M_ul_inner, M_ur_inner, M_ll_inner, M_lr_inner]
	"""
	for M_inner in M_inners:
		print(M_inner.shape)
	"""
	# Compute square root of determinant
	M_sqr_sum = M_ul_inner.copy()
	M_sqr_sum += M_ur_inner
	M_sqr_sum += M_ll_inner
	M_sqr_sum += M_lr_inner
	"""
	print("\n")
	print(M_sqr_sum)
	"""
	debt = np.linalg.det(M_sqr_sum)
	sqrt_debt = np.sqrt(debt)
	return sqrt_debt

class grass_integrator:
	def __init__(self, n, k, seed=493):
		# Save n and k
		self.n = n
		self.k = k
		# Set random seed
		np.random.seed(seed)
		# Store state information
		self.int_val = np.nan
		self.int_vals = []
		self.n_iters = 0

	def sample_interval_prod(self):
		A_pre = np.random.randn(self.n, self.k)
		A = (A_pre - 0.5) * 2
		return A

	def iterate_integrator(self):
		# Sample A
		A = self.sample_interval_prod()
		val = sqrt_det_metric_from_A(A)
		if self.n_iters == 0:
			self.int_val = val
			self.n_iters += 1
		else:
			n_iters_pre = self.n_iters
			self.n_iters += 1
			self.int_val = self.int_val*(n_iters_pre / self.n_iters) + val / self.n_iters
		self.int_vals.append(self.int_val)

	def perform_integration(self, n_iters):
		for _ in tqdm(range(n_iters)):
			self.iterate_integrator()
