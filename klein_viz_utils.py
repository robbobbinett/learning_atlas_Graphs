import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

# Eequally spaced charts in polar coordinates
n_flags = 8
thetas = np.linspace(0, np.pi, n_flags,
			endpoint=False)
phis = np.linspace(0, 2*np.pi, n_flags,
			endpoint=False)

# theta-phi tuple to vector representation of patch
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

vals = [-1, 0, 1]
def k_to_vec(k):
	"""Evaluates the K9 polynomial of a (theta, phi)-
	tuple at the points in vals"""
	patch = np.zeros((3, 3))
	for jx, x in enumerate(vals):
		for jy, y in enumerate(vals):
			patch[jx, jy] = k(x, y)
	return patch.reshape(9)

# Matrix of points on K9 for brute determination of
# closest point
N = int(1e4)
rtN = 100
thetas_long = np.linspace(0, np.pi, rtN)
phis_long = np.linspace(0, 2*np.pi, rtN)
theta_phi = np.zeros((N, 2))
k9_pts = np.zeros((N, 9))
ind = 0
print("Computing brute theta-phi tuples...")
for theta in tqdm(thetas_long):
	for phi in phis_long:
		theta_phi[ind, 0] = theta
		theta_phi[ind, 1] = phi
		k = k_from_theta_phi(theta, phi)
		patch_vec = k_to_vec(k)
		k9_pts[ind, :] = patch_vec
		ind += 1
print("Done")

def find_closest_theta_phi_brute(x):
	X = x.reshape(1, 9)
	dist_vec = euclidean_distances(X, k9_pts)[0, :]
	ind = np.argmin(dist_vec)
	theta, phi = theta_phi[ind, :]
	return theta, phi
