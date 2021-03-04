import scatrex
from scatrex import models

import scanpy as sc
import numpy as np

simulated_data = snakemake.input['simulated_data']
simulated_clones = snakemake.input['simulated_clones']
simulated_clones_labels = snakemake.input['simulated_clones_labels']
simulated_observed_tree = snakemake.input['simulated_observed_tree']
output_file = snakemake.output['fname']

adata = sc.read_csv(simulated_data)
clones_adata = sc.read_csv(simulated_clones)
clones_adata.obs['node'] = np.loadtxt(simulated_clones_labels, delimiter=',', dtype='str')

# Create a new object for the inference
sca = scatrex.SCATrEx(model=models.cna, verbose=True)
sca.add_data(adata)
sca.set_observed_tree(simulated_observed_tree)

# Learn SCATrEx tree
search_kwargs = {'n_iters': 1000, 'n_iters_elbo': 1000,
                'local': True,
                'moves': ['add', 'merge', 'pivot_reattach', 'swap', 'subtree_reattach', 'full']}
sca.learn_tree(reset=True, search_kwargs=search_kwargs)

est_labels = np.array(sca.adata.obs['node'])

np.savetxt(output_file, est_labels, delimiter=',', fmt="%s")
