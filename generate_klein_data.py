import numpy as np
from tqdm import tqdm

from yao_utils_otra import *

data_dir = "data/klein"

def sample_k9(n_pts):
	X_pre = []
	np.random.seed(600)
	for _ in tqdm(range(n_pts)):
		theta = np.pi * np.random.rand()
		phi = twopi * np.random.rand()
		k = k_from_theta_phi(theta, phi)
		x = k_to_vec(k)
		X_pre.append(x)
	X = np.vstack(X_pre)
	np.save(data_dir+"/klein_uniform_"+str(n_pts)+".npy", X)

sample_k9(1000)
