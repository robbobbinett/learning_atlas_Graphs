import numpy as np
from tqdm import tqdm
from grass_utils import *

def test_commutation_matrix():
	np.random.seed(386)
	ms = list(range(1, 10))
	ns = list(range(1, 10))
	for m in tqdm(ms):
		for n in ns:
			mn = m*n
			K = commutation_matrix(m, n)
			assert np.all(np.isclose(K @ K.T, np.eye(mn)))
			X = np.random.randn(m, n)
			vecX = X.reshape(mn)
			vecXt = X.T.reshape(mn)
			assert np.all(np.isclose(K.T @ vecX, vecXt))

def test_yatta_inv_from_A():
	np.random.seed(386)
	ms = list(range(1, 10))
	ns = list(range(1, 10))
	for m in tqdm(ms):
		for n in ns:
			I_n = np.eye(n)
			A_0 = np.zeros((m, n))
			yatta_inv = yatta_inv_from_A(A_0)
			assert np.all(np.isclose(I_n, yatta_inv))
