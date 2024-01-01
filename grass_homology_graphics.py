import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from tqdm import tqdm

data_dir = "data/grass"

# Sapmle size
N = 200

def compute_homology(n, k, d, p=11):
	n_sqr = n**2
	# Load in sample
	var_str = "-".join([str(item) for item in [n, k, N]])
	Ps = np.load(data_dir+"/grass_"+var_str+".npy")
	# Vectorize projection matrices
	pvecs_pre = []
	for j in range(N):
		P = Ps[j, :, :]
		pvecs_pre.append(P.reshape(n_sqr))
	pvecs = np.vstack(pvecs_pre)

	# Compute persistent homology
	rips_dict = ripser(pvecs, maxdim=d, coeff=p)
	# Save persistance diagram
	var_str_long = "-".join([str(item) for item in [n, k, N, d, p]])
	file = open(data_dir+"/grass_hom_"+var_str_long+".npy", "wb")
	pkl.dump(rips_dict["dgms"], file)
	file.close()

def draw_homology(n, k, d, p=11):
	n_sqr = n**2
	# Load in persistence diagrams
	var_str_long = "-".join([str(item) for item in [n, k, N, d, p]])
	file = open(data_dir+"/grass_hom_"+var_str_long+".npy", "rb")
	dgms = pkl.load(file)
	plot_diagrams(dgms)
	plt.show()

nk_pairs = [(2, 2), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3)]

for n, k in tqdm(nk_pairs):
	compute_homology(n, k, 2)

for n, k in tqdm(nk_pairs):
	draw_homology(n, k, 2)
