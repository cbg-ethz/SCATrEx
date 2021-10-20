import scatrex
from scatrex import models

import scanpy as sc
import numpy as np

simulated_data = snakemake.input['simulated_data']
simulated_clones = snakemake.input['simulated_clones']
simulated_clones_labels = snakemake.input['simulated_clones_labels']
simulated_observed_tree = snakemake.input['simulated_observed_tree']
n_tries = snakemake.params['n_tries']
output_file = snakemake.output['fname']

adata = sc.read_csv(simulated_data)
clones_adata = sc.read_csv(simulated_clones)
clones_adata.obs['node'] = np.loadtxt(simulated_clones_labels, delimiter=',', dtype='str')

args = dict(global_noise_factors_precisions_shape=2, num_global_noise_factors=6)

sca_list = []
for i in range(n_tries):
    # Create a new object for the inference
    sca = scatrex.SCATrEx(model=models.cna, verbose=True, model_args=args)
    sca.add_data(adata)
    sca.set_observed_tree(simulated_observed_tree)

    # Run clonealign
    sca.learn_clonemap(n_iters=1000, filter_genes=True, step_size=0.01)

    sca_list.append(sca)

best_sca = sca_list[np.argmax([sca.ntssb.elbo for sca in sca_list])]

best_sca.ntssb.plot_tree()
est_labels = np.array(best_sca.adata.obs['obs_node'])

np.savetxt(output_file, est_labels, delimiter=',', fmt="%s")
