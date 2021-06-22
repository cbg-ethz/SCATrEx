seed = int(snakemake.params['seed'])
n_clones = int(snakemake.params['n_clones'])
n_genes = int(snakemake.params['n_genes'])
n_cells = int(snakemake.params['n_cells'])
n_extras = int(snakemake.params['n_extras'])
n_factors = int(snakemake.params['n_factors'])
simulated_data = snakemake.output['simulated_data']
simulated_labels = snakemake.output['simulated_labels']
simulated_clones = snakemake.output['simulated_clones']
simulated_clones_labels = snakemake.output['simulated_clones_labels']
simulated_observed_tree = snakemake.output['simulated_observed_tree']

import scatrex
from scatrex import models

model_args = dict(log_lib_size_mean=10, num_global_noise_factors=n_factors, global_noise_factors_precisions_shape=10)
sim_sca = scatrex.SCATrEx(model=models.cna, verbose=True, model_args=model_args)
sim_sca.simulate_tree(n_genes=n_genes, n_extra_per_observed=n_extras, seed=seed)
sim_sca.observed_tree.create_adata()
sim_sca.ntssb.reset_node_parameters(node_hyperparams=sim_sca.model_args)
data, labels = sim_sca.simulate_data(n_cells=n_cells, copy=True, seed=seed)

import numpy as np
np.savetxt(simulated_data, data, delimiter=',')
np.savetxt(simulated_labels, labels, delimiter=',', fmt="%s")
np.savetxt(simulated_clones, sim_sca.observed_tree.adata.X, delimiter=',')
np.savetxt(simulated_clones_labels, np.array(sim_sca.observed_tree.adata.obs['node']), delimiter=',', fmt="%s")

import pickle
with open(simulated_observed_tree, 'wb') as f:
    pickle.dump(sim_sca.observed_tree, f)
