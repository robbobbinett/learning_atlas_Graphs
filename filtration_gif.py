from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import gudhi
import networkx as nx
from tqdm import tqdm
import imageio

from peyers_utils import color_dict, shape_dict, get_sample_labels_tissue, tissue_list, microbiota_list

# +
counts_M = pd.read_csv("peyers_data/counts_M.csv", index_col=0)
normCounts_M = pd.read_csv("peyers_data/normCounts_M.csv", index_col=0)
md_M = pd.read_csv("peyers_data/md_M.txt", index_col=0,
                  sep="\t")

counts_tissue = pd.read_csv("peyers_data/counts_tissue.csv", index_col=0)
normCounts_tissue = pd.read_csv("peyers_data/normCounts_tissue.csv",
                                index_col=0)
md_tissue = pd.read_csv("peyers_data/md_tissue.txt", index_col=0,
                       sep="\t")

# +
X_M = counts_M.to_numpy().T
X_M_norm = normCounts_M.to_numpy().T
X_tissue = counts_tissue.to_numpy().T
X_tissue_norm = normCounts_tissue.to_numpy().T

logX_M = np.log(X_M + 1)
logX_M_norm = np.log(X_M_norm + 1)
logX_tissue = np.log(X_tissue + 1)
logX_tissue_norm = np.log(X_tissue_norm + 1)

dist_mat_M = euclidean_distances(logX_M)
dist_mat_M_norm = euclidean_distances(logX_M_norm)
dist_mat_tissue = euclidean_distances(logX_tissue)
dist_mat_tissue_norm = euclidean_distances(logX_tissue_norm)


def get_top_pcs(X, p):
    pca = PCA(n_components=p)
    Y = pca.fit_transform(X)
    return Y



X_tissue_norm_2 = get_top_pcs(X_tissue_norm, 2)
X_tissue_norm_3 = get_top_pcs(X_tissue_norm, 3)

rips_2 = gudhi.RipsComplex(points=X_tissue_norm_2)
rips_3 = gudhi.RipsComplex(points=X_tissue_norm_3)

st_2 = rips_2.create_simplex_tree(max_dimension=2)
st_3 = rips_3.create_simplex_tree(max_dimension=3)

diag_2 = st_2.persistence(homology_coeff_field=2, min_persistence=0)
diag_3 = st_3.persistence(homology_coeff_field=2, min_persistence=0)

fig_2 = plt.figure(figsize=(10, 10))
ax_2 = fig_2.add_subplot()
gudhi.plot_persistence_diagram(diag_2, axes=ax_2)
ax_2.set_title("Persistence diagram, 2 PCs")

fig_3 = plt.figure(figsize=(10, 10))
ax_3 = fig_3.add_subplot()
gudhi.plot_persistence_diagram(diag_3, axes=ax_3)
ax_3.set_title("Persistence diagram, 3 PCs")

plt.close(fig_2)
plt.close(fig_3)

# +
pps_2 = st_2.persistence_pairs()
pps_3 = st_3.persistence_pairs()

intrv_2 = st_2.persistence_intervals_in_dimension(1)
intrv_3 = st_3.persistence_intervals_in_dimension(1)

def isolate_persistence_pairs(pps, dim=1):
    dp1 = dim + 1
    i_pps = []
    for sower, reaper in pps:
        if len(sower) == dp1:
            i_pps.append((sower, reaper))
    return i_pps

i_pps_2 = isolate_persistence_pairs(pps_2)
i_pps_3 = isolate_persistence_pairs(pps_3)

# def birth_death_from_persistence_pairs(st, pps):
def birth_death_from_persistence_intervals(st, intrv):
    births = []
    deaths = []
    for birth, death in intrv:
        births.append(birth)
        deaths.append(death)
#     for sower, reaper in pps:
#         births.append(st.filtration(sower))
#         deaths.append(st.filtration(reaper))
    return births, deaths

# births_2, deaths_2 = birth_death_from_persistence_pairs(st_2, i_pps_2)
# births_3, deaths_3 = birth_death_from_persistence_pairs(st_3, i_pps_3)
births_2, deaths_2 = birth_death_from_persistence_intervals(st_2, intrv_2)
births_3, deaths_3 = birth_death_from_persistence_intervals(st_3, intrv_3)

lifetimes_2 = [(death - birth) for birth, death in zip(births_2, deaths_2)]
lifetimes_3 = [(death - birth) for birth, death in zip(births_3, deaths_3)]

# Sort indices by decreasing lifetime
i_pp_inds_2 = list(range(len(i_pps_2)))
i_pp_inds_2.sort(key=lambda x: lifetimes_2[x], reverse=True)
i_pp_inds_3 = list(range(len(i_pps_3)))
i_pp_inds_3.sort(key=lambda x: lifetimes_3[x], reverse=True)

# +
fig_2 = plt.figure(figsize=(10, 10))
ax_2 = fig_2.add_subplot()

ax_2.scatter(X_tissue_norm_2[:, 0], X_tissue_norm_2[:, 1])
# for ind in i_pp_inds_2[:2]:
for ind in i_pp_inds_2:
    sower, reaper = i_pps_2[ind]
    xs = [X_tissue_norm_2[sub_ind, 0] for sub_ind in reaper]
    xs = xs + [X_tissue_norm_2[reaper[0], 0]]
    ys = [X_tissue_norm_2[sub_ind, 1] for sub_ind in reaper]
    ys = ys + [X_tissue_norm_2[reaper[0], 1]]
    ax_2.plot(xs, ys, color="k", linestyle="dashed")

fig_3 = plt.figure(figsize=(10, 10))
ax_3 = fig_3.add_subplot(projection="3d")

ax_3.scatter(X_tissue_norm_3[:, 0], X_tissue_norm_3[:, 1],
             X_tissue_norm_3[:, 2])
for ind in i_pp_inds_3:
    sower, reaper = i_pps_3[ind]
    xs = [X_tissue_norm_3[sub_ind, 0] for sub_ind in reaper]
    xs = xs + [X_tissue_norm_3[reaper[0], 0]]
    ys = [X_tissue_norm_3[sub_ind, 1] for sub_ind in reaper]
    ys = ys + [X_tissue_norm_3[reaper[0], 1]]
    zs = [X_tissue_norm_3[sub_ind, 2] for sub_ind in reaper]
    zs = zs + [X_tissue_norm_3[reaper[0], 2]]
    ax_3.plot(xs, ys, zs, color="k", linestyle="dashed")

plt.close(fig_2)
plt.close(fig_3)
# -



# # Look at individual features on the 2-PC graphic

# +
# fig_2 = plt.figure(figsize=(10, 10))
# ax_2 = fig_2.add_subplot()

# ax_2.scatter(X_tissue_norm_2[:, 0], X_tissue_norm_2[:, 1])
# for ind in i_pp_inds_2[:2]:
for ind in i_pp_inds_2:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(X_tissue_norm_2[:, 0], X_tissue_norm_2[:, 1])
    
    sower, reaper = i_pps_2[ind]
    xs = [X_tissue_norm_2[sub_ind, 0] for sub_ind in reaper]
    xs = xs + [X_tissue_norm_2[reaper[0], 0]]
    ys = [X_tissue_norm_2[sub_ind, 1] for sub_ind in reaper]
    ys = ys + [X_tissue_norm_2[reaper[0], 1]]
    ax.plot(xs, ys, color="k", linestyle="dashed")
    
    ax.set_title("Feature "+str(ind))
    plt.close(fig)


# +
for ind in i_pp_inds_2:
    birth_time = births_2[ind]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(X_tissue_norm_2[:, 0], X_tissue_norm_2[:, 1])
    
    # Iterate over simplices
    for simp, val in st_2.get_filtration():
        if len(simp) == 2:
            # Filter out older simplices
            if val <= birth_time:
                xs = [X_tissue_norm_2[ind, 0] for ind in simp]
                ys = [X_tissue_norm_2[ind, 1] for ind in simp]
                ax.plot(xs, ys, color="k", linestyle="dashed")
                
    ax.set_title("Feature "+str(ind))
    plt.close(fig)

# Get max filtration value
filts = []
for _, filt in st_2.get_filtration():
    filts.append(filt)
max_filt = np.max(filts)

# Look at esges of complex for 1000 panes between filtration zero
# and a high value
fig_dir = "filtration_gif"
high_val = 30

xlim = (-50, 70)
ylim = (-50, 50)

ts = np.linspace(0.0, high_val, 100)

for j, t in tqdm(enumerate(ts), total=len(ts)):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.scatter(X_tissue_norm_2[:, 0], X_tissue_norm_2[:, 1])

    for simp, filt in st_2.get_filtration():
        if len(simp) == 2:
            if filt <= t:
                xs = [X_tissue_norm_2[ind, 0] for ind in simp]
                ys = [X_tissue_norm_2[ind, 1] for ind in simp]
                ax.plot(xs, ys, color="k", linestyle="dashed")
    ax.set_title("Filtration = "+str(t))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.savefig(fig_dir+"/filtration_pane_"+str(j)+".png")
    plt.close(fig)

images = []
for j in range(len(ts)):
    images.append(imageio.imread(fig_dir+"/filtration_pane_"+str(j)+".png"))
imageio.mimsave(fig_dir+"/filtration_gif.gif", images)
