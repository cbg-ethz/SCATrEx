import scatrex
from scatrex import models

import scanpy as sc
import numpy as np

cnv_matrix_path = snakemake.input["cnv_matrix_path"]
clone_sizes_path = snakemake.input["clone_sizes_path"]
output_file = snakemake.output["fname"]
n_iters = snakemake.params["n_iters"]
n_factors = snakemake.params["n_factors"]


# Run SCATrEx
args = dict(
    num_global_noise_factors=n_factors,
    unobserved_factors_kernel_concentration=1e-3,
    unobserved_factors_kernel_rate=1.0,
    global_noise_factors_precisions_shape=2.0,
)
sca = scatrex.SCATrEx(model=models.cna, model_args=args)
sca.add_data(adata)
sca.set_observed_tree(new_observed_tree)


MOVE_WEIGHTS = {
    "add": 1,
    "merge": 4.0,
    "prune_reattach": 1.0,
    "pivot_reattach": 1.0,
    "swap": 1.0,
    "add_reattach_pivot": 1.0,
    "subtree_reattach": 0.5,
    "push_subtree": 1.0,
    "extract_pivot": 1.0,
    "subtree_pivot_reattach": 0.5,
    "perturb_node": 1,
    "perturb_globals": 1,
    "optimize_node": 1,
    "transfer_factor": 0.0,
    "transfer_unobserved": 0.0,
}
import logging

search_kwargs = {
    "n_iters": n_iters,
    "n_iters_elbo": 100,
    "move_weights": MOVE_WEIGHTS,
    "local": True,
    "factor_delay": 0,
    "step_size": 0.01,
    "posterior_delay": 0,
    "mb_size": 256,
    "num_samples": 1,
    "max_nodes": 5,
    "add_rule_thres": 1.0,
    "joint_init": True,
    "anneal": False,
    "restart_step": 100,
    "window": 50,
    "every": 10,
    "threshold": 1e-5,
    "alpha": 0.0,
    "random_seed": 1,
    "verbosity": logging.DEBUG,
}
sca.learn_tree(reset=True, batch_key="batc", search_kwargs=search_kwargs)
