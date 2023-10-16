import numpy as np
import matplotlib.pyplot as plt

from grass_utils import *

n = 5
k = 2

gi = grass_integrator(n, k)
gi.perform_integration(10000000)

"""
A_0 = np.zeros((n, k))
val_0 = sqrt_det_metric_from_A(A_0)
print(val_0)
print("\n\n")
"""

int_vals_smol = gi.int_vals[:100]
for val in int_vals_smol:
	print(val)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

ax.plot(gi.int_vals)

plt.show()
