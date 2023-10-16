import numpy as np

from grass_utils import *

data_dir = "data/grass"

nk_pairs = [(2, 2), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3)]

# Sample size
N = 100

for n, k in nk_pairs:
	Xs = random_sample_grass_as_stiefel(n, k, n_pts=N, seed=n*k)
	Ps_pre = stiefel_list_to_projector_list(Xs)
	Ps = np.stack(Ps_pre)
	var_str = "-".join([str(item) for item in [n, k, N]])
	np.save(data_dir+"/grass_"+var_str+".npy", Ps)
