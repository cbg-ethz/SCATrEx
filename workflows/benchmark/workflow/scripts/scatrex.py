import scatrex
from scatrex import models

import scanpy as sc
import numpy as np

from jax.config import config
config.update("jax_debug_nans", False)

simulated_data = snakemake.input['simulated_data']
simulated_clones = snakemake.input['simulated_clones']
simulated_clones_labels = snakemake.input['simulated_clones_labels']
simulated_observed_tree = snakemake.input['simulated_observed_tree']
n_tries = snakemake.params['n_tries']
output_file = snakemake.output['fname']

adata = sc.read_csv(simulated_data)
clones_adata = sc.read_csv(simulated_clones)
clones_adata.obs['node'] = np.loadtxt(simulated_clones_labels, delimiter=',', dtype='str')

sca_list = []
for i in range(n_tries):
    # Create a new object for the inference
    sca = scatrex.SCATrEx(model=models.cna, verbose=True)
    sca.add_data(adata)
    sca.set_observed_tree(simulated_observed_tree)

    # Learn SCATrEx tree
    move_weights = {'add':3,
                    'merge':6,
                    'prune_reattach':1,
                    'pivot_reattach':1,
                    'swap':1,
                    'add_reattach_pivot':.5,
                    'subtree_reattach':1,
                    'push_subtree':.5,
                    'extract_pivot':.5,
                    'perturb_node':.5,
                    'clean_node':.5,
                    'subtree_pivot_reattach':.5,
                    'reset_globals':.0,
                    'full':.0,
                    'globals':1}

    search_kwargs = {'n_iters': 500, 'n_iters_elbo': 500,
                    'move_weights': move_weights,
                    'local': True,
                    'factor_delay': 0,
                    'step_size': 0.01,
                    'posterior_delay': 0,
                    'mb_size': 200,
                    'num_samples': 1,
                    'window': 50,
                    'max_nodes': 5,
                    'add_rule_thres': .4,
                    'joint_init': False}
    sca.learn_tree(reset=True, search_kwargs=search_kwargs)
    sca_list.append(sca)

best_sca = sca_list[np.argmax([sca.ntssb.elbo for sca in sca_list])]
est_labels = np.array(best_sca.adata.obs['node'])

np.savetxt(output_file, est_labels, delimiter=',', fmt="%s")

# simulated_data = 'results/data/c3_g200_n200/e2_f0/r0_data.csv'#snakemake.input['simulated_data']
# simulated_clones = 'results/data/c3_g200_n200/e2_f0/r0_clones.csv'#snakemake.input['simulated_clones']
# simulated_clones_labels = 'results/data/c3_g200_n200/e2_f0/r0_obsnodes.csv' #snakemake.input['simulated_clones_labels']
# simulated_observed_tree = 'results/data/c3_g200_n200/e2_f0/r0_obstree.pickle'#snakemake.input['simulated_observed_tree']
# # output_file = #snakemake.output['fname']
#
# adata = sc.read_csv('workflows/benchmark/' + simulated_data)
# clones_adata = sc.read_csv('workflows/benchmark/' + simulated_clones)
# clones_adata.obs['node'] = np.loadtxt('workflows/benchmark/' + simulated_clones_labels, delimiter=',', dtype='str')
#
# adata.obs['nodes'] = np.loadtxt('workflows/benchmark/' + 'results/data/c3_g200_n200/e2_f0/r0_labels.csv', dtype='str')
# sc.pp.normalize_per_cell(adata)
# sc.pp.log1p(adata)
# sc.pl.heatmap(adata, var_names=adata.var_names, groupby='nodes')
