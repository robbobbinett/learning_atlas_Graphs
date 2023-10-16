import numpy as np
from tqdm import tqdm

data_dir = "data/sphere"

def draw_klein_samples(n_pts, seed_coeff=386):
	seed = seed_coeff * n_pts
	np.random.seed(seed)
	pts_pre = []
	for _ in range(n_pts):
		pt = np.random.randn(3)
		pt_norm = np.linalg.norm(pt)
		pts_pre.append(pt / pt_norm)
	return np.vstack(pts_pre)

for size in tqdm([10, 100, 1000, 10000]):
	pts = draw_klein_samples(size)
	np.save(data_dir+"/sphere_uniform_"+str(size)+".npy", pts)
