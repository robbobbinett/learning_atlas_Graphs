# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from ripser import ripser
from persim import plot_diagrams
import networkx as nx

from peyers_utils import color_dict, shape_dict, get_sample_labels_tissue, tissue_list, microbiota_list

# +
normCounts_tissue = pd.read_csv("peyers_data/normCounts_tissue.csv", index_col=0)
X_tissue_norm = normCounts_tissue.to_numpy().T

logX_tissue_norm = np.log(X_tissue_norm + 1)

dist_mat_tissue_norm = euclidean_distances(logX_tissue_norm)


# +
def look_at_homology(dist_mat):
    rips_dict = ripser(dist_mat, maxdim=2,
                      distance_matrix=True)
    dgms = rips_dict["dgms"]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    plot_diagrams(dgms, ax=ax)
    
    #plt.show()

def get_most_variable_genes(X, p):
    var_vec = np.var(X, axis=0)
    ind_list = list(range(len(var_vec)))
    ind_list.sort(key=lambda x: var_vec[x], reverse=True)
    top_inds = ind_list[:p]
    return top_inds


# +
p = 3000

top_inds_p = get_most_variable_genes(X_tissue_norm, p)

X_tissue_norm_p = X_tissue_norm[:, top_inds_p]
# -

# <h3>Trim away duodenum rows</h3>

# +
sample_tissue_list, sample_microbiota_list = get_sample_labels_tissue()

to_keep = [("Duodenum" not in tissue) for tissue in sample_tissue_list]
X_tissue_norm_p_trim = X_tissue_norm_p[to_keep, :]

X_tissue_norm_trim = X_tissue_norm[to_keep, :]

# +
logX_tissue_norm_p_trim = np.log10(X_tissue_norm_p_trim + 1)
dist_mat_p_trim = euclidean_distances(X_tissue_norm_p_trim,
                                        X_tissue_norm_p_trim)
dist_mat_log_p_trim = euclidean_distances(logX_tissue_norm_p_trim,
                                        logX_tissue_norm_p_trim)

# +
pca = PCA(n_components=3)
pca.fit(X_tissue_norm_trim)

umap = UMAP(n_components=3)
Y_tissue_norm_umap = umap.fit_transform(X_tissue_norm_trim)

tsne = TSNE(n_components=3)
Y_tissue_norm_tsne = tsne.fit_transform(X_tissue_norm_trim)

#Y_M_norm = pca.transform(X_M_norm)
Y_tissue_norm = pca.transform(X_tissue_norm_trim)

# +
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

sample_tissue_list_trim = [sample_tissue_list[j] for j, val in enumerate(to_keep) if val]
sample_microbiota_list_trim = [sample_microbiota_list[j] for j, val in enumerate(to_keep) if val]

sample_group_dict = {}
for tissue in tissue_list:
    for micro in microbiota_list:
        key = (tissue, micro)
        #sample_group_dict[key] = np.zeros(78, dtype=bool)
        sample_group_dict[key] = np.zeros(60, dtype=bool)

for j, key in enumerate(zip(sample_tissue_list_trim, sample_microbiota_list_trim)):
    sample_group_dict[key][j] = True

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

for tissue in tissue_list:
    for micro in microbiota_list:
        key = (tissue, micro)
        color = color_dict[tissue]
        shape = shape_dict[micro]
        to_keep = sample_group_dict[key]
        Y_kept = Y_tissue_norm[to_keep, :]
        ax.scatter(Y_kept[:, 0], Y_kept[:, 1], Y_kept[:, 2],
                      c=color, marker=shape, s=100, alpha=1.0)

ax.set_xlabel("PC 1", fontsize=16, labelpad=20)
ax.set_ylabel("PC 2", fontsize=16, labelpad=20)
ax.set_zlabel("PC 3", fontsize=16, labelpad=20)
ax.set_title("Without Duodenum", fontsize=24)

fig.savefig("pca3_reference.jpg")

#plt.show()

plt.close(fig)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

for tissue in tissue_list:
    for micro in microbiota_list:
        key = (tissue, micro)
        color = color_dict[tissue]
        shape = shape_dict[micro]
        to_keep = sample_group_dict[key]
        Y_kept = Y_tissue_norm_umap[to_keep, :]
        ax.scatter(Y_kept[:, 0], Y_kept[:, 1], Y_kept[:, 2],
                      c=color, marker=shape, s=100)

fig.savefig("umap3_reference.jpg")

#plt.show()

plt.close(fig)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

for tissue in tissue_list:
    for micro in microbiota_list:
        key = (tissue, micro)
        color = color_dict[tissue]
        shape = shape_dict[micro]
        to_keep = sample_group_dict[key]
        Y_kept = Y_tissue_norm_tsne[to_keep, :]
        ax.scatter(Y_kept[:, 0], Y_kept[:, 1], Y_kept[:, 2],
                      c=color, marker=shape, s=100)

fig.savefig("tsne3_reference.jpg")

#plt.show()

plt.close(fig)

# +
for ind in range(3):
    j, k = [i for i in range(3) if i != ind]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    
    for tissue in tissue_list:
        for micro in microbiota_list:
            key = (tissue, micro)
            color = color_dict[tissue]
            shape = shape_dict[micro]
            to_keep = sample_group_dict[key]
            Y_kept = Y_tissue_norm[to_keep, :]
            ax.scatter(Y_kept[:, j], Y_kept[:, k],
                          c=color, marker=shape)

    ax.set_xlabel("PC " + str(j + 1))
    ax.set_ylabel("PC " + str(k + 1))

#plt.show()

# Make legend
fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot()
ax.set_axis_off()
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)

legend_elements = []
for tissue in tissue_list:
    legend_elements.append(Patch(facecolor=color_dict[tissue],
                                 edgecolor=None, label=tissue))

ax.legend(handles=legend_elements, loc="center")

fig.savefig("tissue_legend.jpg")

plt.close(fig)

fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot()
ax.set_axis_off()
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)

legend_elements = []
for micro in microbiota_list:
    legend_elements.append(Line2D([0], [0], color="k", marker=shape_dict[micro],
                                label=micro))
    #legend_elements.append(Patch(facecolor="k", edgecolor=None,
    #                             label=micro, marker=shape_dict[micro]))

ax.legend(handles=legend_elements, loc="center")

fig.savefig("shape_legend.jpg")

plt.close(fig)
