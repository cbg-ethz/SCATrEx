#!/usr/bin/env python

"""Tests for `scatrex` package."""

import pytest


import scatrex
from scatrex import models

def test_simulation():
    """Run a complete simulation test"""
    model_args = dict(log_lib_size_mean=10, num_global_noise_factors=0)

    # Simulate
    sim_sca = scatrex.SCATrEx(model=models.cna, verbose=True, model_args=model_args)
    sim_sca.simulate_tree(observed_tree=None, n_extra_per_observed=1, seed=40)
    sim_sca.ntssb.reset_node_parameters(node_hyperparams=sim_sca.model_args)
    sim_sca.observed_tree.create_adata()
    sim_sca.simulate_data()
    sim_sca.normalize_data()
    sim_sca.project_data()

    # Create a new object for the inference
    sca = scatrex.SCATrEx(model=models.cna, verbose=True, model_args=model_args)
    sca.add_data(sim_sca.adata.raw.to_adata())
    sca.set_observed_tree(sim_sca.observed_tree)
    sca.normalize_data()
    sca.project_data()
    search_kwargs = {'n_iters': 100, 'n_iters_elbo': 100,
                    'local': True,
                    'moves': ['add', 'merge', 'pivot_reattach', 'swap', 'subtree_reattach', 'full']}
    sca.learn_tree(reset=True, search_kwargs=search_kwargs)

    import numpy as np
    from sklearn.metrics import accuracy_score, adjusted_rand_score
    true_hnode = np.array([assignment['subtree'].label for assignment in sim_sca.ntssb.assignments])
    inf_hnode = np.array([assignment['subtree'].label for assignment in sca.ntssb.assignments])
    assert accuracy_score(inf_hnode, true_hnode) > 1/3 # better than random
