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

# Run clonealign first
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
est_clones = np.array(best_sca.adata.obs['obs_node'])

# Run Leiden on each clone
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.regress_out(adata, 'total_counts')
sc.pp.scale(adata, max_value=10)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

est_labels = np.array(['----------'] * adata.shape[0])
for clone in np.unique(est_clones):
    clone_cells_idx = np.where(est_clones == clone)[0]
    sub_adata = adata[clone_cells_idx]
    print(sub_adata)
    try: # Strange error possible
        sc.tl.leiden(sub_adata)
        labs = np.array([clone + str(l) for l in sub_adata.obs['leiden']])
    except Exception:
        labs = np.array([clone] * sub_adata.shape[0])
    est_labels[clone_cells_idx] = labs

np.savetxt(output_file, est_labels, delimiter=',', fmt="%s")
