import numpy as np
from tqdm import tqdm

from yao_utils_otra import *

data_dir = "data/klein"

def sample_k9(n_pts):
	X_pre = []
	theta_list = []
	phi_list = []
	np.random.seed(600)
	for _ in tqdm(range(n_pts)):
		theta = np.pi * np.random.rand()
		phi = twopi * np.random.rand()
		theta_list.append(theta)
		phi_list.append(phi)
		k = k_from_theta_phi(theta, phi)
		x = k_to_vec(k)
		X_pre.append(x)
	X = np.vstack(X_pre)
	np.save(data_dir+"/klein_uniform_"+str(n_pts)+".npy", X)
	np.save(data_dir+"/klein_uniform_thetas_"+str(n_pts)+".npy", theta_list)
	np.save(data_dir+"/klein_uniform_phis_"+str(n_pts)+".npy", phi_list)

sample_k9(10000)
