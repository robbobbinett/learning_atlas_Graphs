import itertools as it
import numpy as np
import sympy as sp
from tqdm import tqdm

from manifold_utils import *

twopi = 2*np.pi

vals = [-1, 0, 1]
NoneType = type(None)
atan2 = np.arctan(2)

def k_from_theta_phi(theta, phi):
	"""Return the polynomial representing (theta, phi)
	according to the K9 representation of the Klein
	bottle."""
	sint = np.sin(theta)
	cost = np.cos(theta)
	sinp = np.sin(phi)
	cosp = np.cos(phi)

	def k(x, y):
		summand_a = cosp * (x*cost + y*sint)**2
		summand_b = sinp * (x*cost + y*sint)
		return summand_a + summand_b

	return k

def k_to_vec(k):
	"""Evaluates the K9 polynomial of a (theta, phi)-
	tuple at the points in vals"""
	patch = np.zeros((3, 3))
	for jx, x in enumerate(vals):
		for jy, y in enumerate(vals):
			patch[jx, jy] = k(x, y)
	return patch.reshape(9)

def kernel(x):
	"""Guassian kernel"""
	return np.exp(-x**2)

# Set up SymPy parameters that will be helpful for finding closest
# theta-phi projection
### Define atom variables
sp.var("x,y,t,p", real=True)
### Trigonometric terms
cost = sp.cos(t)
sint = sp.sin(t)
cosp = sp.cos(p)
sinp = sp.sin(p)
### Left and right summands of injection into function-space
left_summand = cosp * (x*cost + y*sint)**2
right_summand = sinp * (x*cost + y*sint)
### Injection into function-space
k = left_summand + right_summand
### Fixed values of x, y for injection into R^9
vals_sqr = list(it.product(vals, vals))
### R^9 vector as list
k_list = []
for x_0, y_0 in vals_sqr:
	k_list.append(k.subs({x: x_0, y: y_0}, simultaneous=True))
### R^9 vector as SymPy matrix
k9 = sp.ImmutableMatrix(k_list)
### Lambdify k9
k9_fun = sp.lambdify([t, p], k9)
### Partial derivatives of k9
k9_t = k9.diff(t)
k9_p = k9.diff(p)
### Lambdify derivatives
k9_t_fun = sp.lambdify([t, p], k9_t)
k9_p_fun = sp.lambdify([t, p], k9_p)

def theta_phi_grad_from_sympy(theta, phi):
	df_dt = k9_t_fun(theta, phi).reshape(9)
	df_dp = k9_p_fun(theta, phi).reshape(9)
	return np.vstack([df_dt, df_dp])

# Matrix of points on K9 for brute determination of
# closest point
N = int(1e4)
rtN = 100
thetas = np.linspace(0, np.pi, rtN)
phis = np.linspace(0, twopi, rtN)
theta_phi = np.zeros((N, 2))
k9_pts = np.zeros((N, 9))
ind = 0
print("Computing brute matrix...")
for theta in tqdm(thetas):
	for phi in phis:
		theta_phi[ind, 0] = theta
		theta_phi[ind, 1] = phi
		k9_pts[ind, :] = k9_fun(theta, phi).reshape(9)
		ind += 1
print("Done")

def find_closest_theta_phi_brute(x):
	X = x.reshape(1, 9)
	dist_vec = euclidean_distances(X, k9_pts)[0, :]
	ind = np.argmin(dist_vec)
	theta, phi = theta_phi[ind, :]
	return theta, phi

"""
class GraphPAM:
	def __init__(self, dist_mat, k, med_ind_init=None):
		# Save params
		self.dist_mat = dist_mat
		self.k = k
		self.n = dist_mat.shape[0]

		# Randomly assign initial medoids
		if isinstance(med_ind_init, NoneType):
			cond = True # Make sure no repeats
			while cond:
				self.med_inds = np.random.randint(self.n, size=(k,))
				if len(np.unique(self.med_inds)) == k:
					cond = False
		else:
			assert isinstance(med_ind_init, np.ndarray)
			assert med_ind_init.shape == (k,)
			self.mid_inds = mid_ind_init

		# Assign points to initial medoids
		self.assignment_dict = {}
		for ind in range(self.n):
			if ind in self.med_inds:
				self.assignment_dict[ind] = ind
			else:
				dist_vec_pre = self.dist_mat[ind, :]
				dist_vec = dist_vec_pre[self.med_inds]
				min_dist_ind_pre = np.argmin(dist_vec)
				self.assignment_dict[ind] = self.med_inds[min_dist_ind_pre]

		# Compute initial cost
		self.cost = self.compute_cost()

	def compute_cost(self):
		cost = 0.0
		for ind in range(self.n):
			if ind not in self.med_inds:
				assigned_ind = self.assignment_dict[ind]
				cost += self.dist_mat[ind, assigned_ind]
		return cost

	def perform_pam(self, max_iter=100):
		cond = True
		iter_count = 0
		while cond:
			cost_best = self.cost
			med_swap = None
			not_med_swap = None
			for ind in range(self.n):
				if ind not in self.med_inds:
					for med_ind in self.med_inds:
						med_ind_temp = np.array([item for item in self.med_inds if item != med_ind else ind])
						otra = GraphPAM(self.dist_mat, self.k, med_ind_init=med_ind_temp)
						if otra.cost < cost_best:
							cost_best = otra.cost
							med_swap = med_ind
							not_med_swap = ind
							del otra
			if isinstance(med_swap, NoneType) and isinstance(not_med_swap, NoneType):
				cond = False
			else:
				med_ind_temp = np.array([item for item in self.med_inds if item != med_swap else not_med_swap])
				self.med_inds = med_ind_temp
				self.compute_cost()
			if iter_count >= max_iter:
				cond = False
			iter_count += 1
"""
