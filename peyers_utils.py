import numpy as np
import pandas as pd

data_dir = "peyers_data"

tissue_list = []
tissue_list.append("wt_Ileum")
tissue_list.append("wt_Jejunum")
tissue_list.append("wt_Duodenum")
tissue_list.append("villin_cre_Ileum")
tissue_list.append("villin_cre_Jejunum")
tissue_list.append("villin_cre_Duodenum")

microbiota_list = []
microbiota_list.append("Jax")
microbiota_list.append("Jax_SFB")
microbiota_list.append("GF")

def make_color_dict():
	color_list = []
	with open("hexadecimal.txt", "r") as file:
		for line in file.readlines():
			color_list.append(line[:-1].replace('"', ''))
	color_dict = dict(zip(tissue_list, color_list))
	return color_dict

color_dict = make_color_dict()

shape_list = ["o", "^", "s"]
shape_dict = dict(zip(microbiota_list, shape_list))

def get_sample_labels_tissue():
	sample_tissue_list = []
	sample_microbiota_list = []
	normCounts_tissue = pd.read_csv("peyers_data/normCounts_tissue.csv",
                                index_col=0)
	for col in normCounts_tissue.columns:
		# Trim away numerical suffix
		segs = col.split("_")
		col = "_".join(segs[:-1])
		# Find microbiota status
		if "Jax" in col:
			if "SFB" in col:
				sample_microbiota_list.append("Jax_SFB")
			else:
				sample_microbiota_list.append("Jax")
		else:
			sample_microbiota_list.append("GF")
		# Find tissue_status
		if "Ileum" in col:
			if "wt" in col:
				sample_tissue_list.append("wt_Ileum")
			else:
				sample_tissue_list.append("villin_cre_Ileum")
		if "Jejunum" in col:
			if "wt" in col:
				sample_tissue_list.append("wt_Jejunum")
			else:
				sample_tissue_list.append("villin_cre_Jejunum")
		if "Duodenum" in col:
			if "wt" in col:
				sample_tissue_list.append("wt_Duodenum")
			else:
				sample_tissue_list.append("villin_cre_Duodenum")

	return sample_tissue_list, sample_microbiota_list
