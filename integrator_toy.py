import numpy as np
import matplotlib.pyplot as plt

from grass_utils import *

n_ts = 1000

m = 5
n = 2

ts = np.linspace(0, 1, n_ts)

A_pre = np.ones((m, n))
vals = []
for t in ts:
	A = A_pre * t
	val = sqrt_det_metric_from_A(A)
	vals.append(val)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()
ax.semilogy(vals)

plt.show()
