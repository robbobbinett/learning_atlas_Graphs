import itertools as it
import numpy as np
import sympy as sp
from tqdm import tqdm

from manifold_utils import *
from atlas_general import AtlasGeneral


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

class RPBKleinOnAtlasGeneral:
	"""Perform Riemannian Principal Boundary Algorithm on the Klein
	bottle to discriminate convex and concave patches, using an
	AtlasGeneral representation"""
	def __init__(self, atlas, X_pos, X_neg):
		self.atlas = atlas
		# If no dense subgraph, create dense subgraph
		if not hasattr(self.atlas, "G_brute"):
			self.atlas.construct_dense_graph()

		# Store concave, convex observations
		self.X_pos = X_pos
		self.X_neg = X_neg
		self.N_pos = self.X_pos.shape[0]
		self.N_neg = self.X_neg.shape[0]
		self.chart_assignments_pos = self.assign_charts(self.X_pos)
		self.chart_assignments_neg = self.assing_charts(self.X_neg)

		### Dicts of coordinate chart representations
		self.Xi_pos_dict = {}
		self.Xi_neg_dict = {}
		for chart in range(self.atlas.n_charts):
			x_0, L, _, _, _, _, _ = self.atlas.chart_dict[chart]
			self.Xi_pos_dict[chart] = (self.X_pos - x_0) @ L
			self.Xi_neg_dict[chart] = (self.X_neg - x_0) @ L

		### Cache quadratic terms
		self.T_pos_dict = {}
		self.T_neg_dict = {}
		for chart in range(self.atlas.n_charts):
			Xi_pos = self.Xi_pos_dict[chart]
			Xi_neg = self.Xi_neg_dict[chart]
			self.T_pos_dict[chart] = get_quadratic_and_const_terms(Xi_pos)
			self.T_neg_dict[chart] = get_quadratic_and_const_terms(Xi_neg)
