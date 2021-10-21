seed = int(snakemake.params['seed'])
n_clones = int(snakemake.params['n_clones'])
n_genes = int(snakemake.params['n_genes'])
n_cells = int(snakemake.params['n_cells'])
n_extras = int(snakemake.params['n_extras'])
n_factors = int(snakemake.params['n_factors'])
n_regions = int(snakemake.params['n_regions'])
min_nevents = int(snakemake.params['min_nevents'])
max_nevents_frac = float(snakemake.params['max_nevents_frac'])
frac_dosage = float(snakemake.params['frac_dosage'])
log_lib_size_mean = float(snakemake.params['log_lib_size_mean'])
log_lib_size_std = float(snakemake.params['log_lib_size_std'])

simulated_data = snakemake.output['simulated_data']
simulated_labels = snakemake.output['simulated_labels']
simulated_clones = snakemake.output['simulated_clones']
simulated_clones_labels = snakemake.output['simulated_clones_labels']
simulated_observed_tree = snakemake.output['simulated_observed_tree']

import scatrex
from scatrex import models
import numpy as np

# model_args = dict(log_lib_size_mean=10, num_global_noise_factors=n_factors, global_noise_factors_precisions_shape=10)
# sim_sca = scatrex.SCATrEx(model=models.cna, verbose=True, model_args=model_args)
# sim_sca.simulate_tree(n_genes=n_genes, n_extra_per_observed=n_extras, seed=seed)
# sim_sca.observed_tree.create_adata()
# sim_sca.ntssb.reset_node_parameters(node_hyperparams=sim_sca.model_args)
# data, labels = sim_sca.simulate_data(n_cells=n_cells, copy=True, seed=seed)

theta = 50
sim_sca = scatrex.SCATrEx(model=models.cna, verbose=True, model_args=dict(log_lib_size_mean=log_lib_size_mean, log_lib_size_std=log_lib_size_std,
                                                                          num_global_noise_factors=n_factors,
                                                                          global_noise_factors_precisions_shape=10.,
                                                                          unobserved_factors_kernel_concentration=1./theta,
                                                                          frac_dosage=frac_dosage,
                                                                          baseline_shape=.7))
observed_tree_args = dict(n_nodes=n_clones, node_weights=[1]*n_clones)
observed_tree_params = dict(n_regions=n_regions, min_cn=1, min_nevents=min_nevents, max_nevents_frac=max_nevents_frac)
sim_sca.simulate_tree(observed_tree=None, n_extra_per_observed=n_extras, n_genes=n_genes, seed=seed,
                        observed_tree_params=observed_tree_params, observed_tree_args=observed_tree_args)
sim_sca.observed_tree.create_adata()

clip = 3.5
for node in sim_sca.ntssb.get_nodes():
    node.unobserved_factors = np.clip(node.unobserved_factors, -clip, clip)

data, labels = sim_sca.simulate_data(n_cells=n_cells, copy=True, seed=seed)
sim_sca.plot_tree(counts=True)

# Remove genes with no expression in any cell
to_keep = np.where(np.sum(data, axis=0) > 0)[0]

data = data[:,to_keep]
for node in sim_sca.observed_tree.tree_dict:
    sim_sca.observed_tree.tree_dict[node]['params'] = sim_sca.observed_tree.tree_dict[node]['params'][to_keep]
sim_sca.observed_tree.adata = sim_sca.observed_tree.adata[:, to_keep]

np.savetxt(simulated_data, data, delimiter=',')
np.savetxt(simulated_labels, labels, delimiter=',', fmt="%s")
np.savetxt(simulated_clones, sim_sca.observed_tree.adata.X, delimiter=',')
np.savetxt(simulated_clones_labels, np.array(sim_sca.observed_tree.adata.obs['node']), delimiter=',', fmt="%s")

import pickle
sim_sca.observed_tree.adata = None # Pickling sliced AnnData would break it
with open(simulated_observed_tree, 'wb') as f:
    pickle.dump(sim_sca.observed_tree, f)
