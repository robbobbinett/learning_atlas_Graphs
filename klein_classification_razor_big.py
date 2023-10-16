import numpy as np
from tqdm import tqdm

n_pts = 1000
vals = [-1, 0, 1]
twopi = 2*np.pi
#two_atan_2 = 2 * np.arctan(2)
two_atan_2 = 2 * np.arctan(0.1)

def k_from_theta_phi(theta, phi):
	sint = np.sin(theta)
	cost = np.cos(theta)
	sinp = np.sin(phi)
	cosp = np.cos(phi)
	def k(x, y):
		summand_a = cosp * (x*cost + y*sint)**2
		summand_b = sinp * (x*cost + y*sint)
		return summand_a + summand_b
	return k

def generate_data(n_pts=n_pts, seed=108,
		data_dir="data/klein"):
	# Set random seed
	np.random.seed(seed)
	# Store angular parameters
	thetas_pos = []
	thetas_neg = []
	phis_pos = []
	phis_neg = []
	# Store patches
	patches_pos = []
	patches_neg = []
	for _ in tqdm(range(n_pts)):
		# positive
		theta = np.pi * np.random.rand()
		### Ensure that tan(phi) is between -2 and 2
		phi_pre = two_atan_2 * (np.random.rand() - 0.5)
		if phi_pre >= 0:
			phi = phi_pre
		else:
			phi = twopi + phi_pre
		k_pos = k_from_theta_phi(theta, phi)
		patch_pos = np.zeros((3, 3))
		for jx, x in enumerate(vals):
			for jy, y in enumerate(vals):
				patch_pos[jx, jy] = k_pos(x, y)
		### Store values
		thetas_pos.append(theta)
		phis_pos.append(phi)
		patches_pos.append(patch_pos)

		# negative
		theta = np.pi * np.random.rand()
		### Ensure that tan(phi) is between -2 and 2
		phi_pre = two_atan_2 * (np.random.rand() - 0.5)
		phi = phi_pre + np.pi
		k_neg = k_from_theta_phi(theta, phi)
		patch_neg = np.zeros((3, 3))
		for jx, x in enumerate(vals):
			for jy, y in enumerate(vals):
				patch_neg[jx, jy] = k_neg(x, y)
		### Store values
		thetas_neg.append(theta)
		phis_neg.append(phi)
		patches_neg.append(patch_neg)

	# Save stored values
	np.save(data_dir+"/thetas_pos_razor_big.npy", thetas_pos)
	np.save(data_dir+"/thetas_neg_razor_big.npy", thetas_neg)
	np.save(data_dir+"/phis_pos_razor_big.npy", phis_pos)
	np.save(data_dir+"/phis_neg_razor_big.npy", phis_neg)
	np.save(data_dir+"/patches_pos_razor_big.npy", patches_pos)
	np.save(data_dir+"/patches_neg_razor_big.npy", patches_neg)

generate_data()
