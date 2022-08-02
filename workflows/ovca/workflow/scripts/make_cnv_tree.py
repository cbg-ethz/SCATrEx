# Tree from paper, but without all the ancestors, since we don't have a nice way to
# get the CNVs at those nodes without SCICoNE (which starts from the read counts, not from CNVs)
import scatrex
from scatrex import models

import scanpy as sc
import numpy as np

cnv_matrix_path = snakemake.input["cnv_matrix_path"]
clone_sizes_path = snakemake.input["clone_sizes_path"]
output_file = snakemake.output["fname"]

tree_dict = dict()
tree_dict["root"] = {
    "parent": "NULL",
    "params": 2 * np.ones((cnv_matrix.shape[1])),
    "size": 0,
}
for node in cnv_matrix.index:
    # Ignore A, B, C, D because we don't have scRNA data from it
    if node == "A" or node == "B" or node == "C" or node == "D":
        continue
    tree_dict[node] = {
        "parent": "root",
        "params": cnv_matrix.loc[node].values,
        "size": clone_sizes[node],
    }

# tree_dict['Anc1'] = {'parent': 'root',
#                     'params': 2*np.ones((cnv_matrix.shape[1])),
#                     'size': 0}
# tree_dict['Anc2'] = {'parent': 'Anc1',
#                     'params': 2*np.ones((cnv_matrix.shape[1])),
#                     'size': 0}
# tree_dict['Anc3'] = {'parent': 'I',
#                     'params': 2*np.ones((cnv_matrix.shape[1])),
#                     'size': 0}

tree_dict["E"]["parent"] = "root"
tree_dict["F"]["parent"] = "root"
tree_dict["I"]["parent"] = "root"
tree_dict["G"]["parent"] = "I"
tree_dict["H"]["parent"] = "I"

# Change names of I G H to make I appear last
tree_dict["IG"] = tree_dict["I"]
tree_dict["IG"]["parent"] = "root"
tree_dict["GH"] = tree_dict["G"]
tree_dict["GH"]["parent"] = "G"
tree_dict["HI"] = tree_dict["H"]
tree_dict["HI"]["parent"] = "G"

tree_dict["G"] = tree_dict["IG"]
tree_dict["H"] = tree_dict["GH"]
tree_dict["I"] = tree_dict["HI"]

del tree_dict["IG"]
del tree_dict["GH"]
del tree_dict["HI"]

json.dumps(tree_dict)
