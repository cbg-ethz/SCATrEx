import scatrex
from scatrex import models
import scanpy as sc
import numpy as np


def test_scatrex():
    seed = 1
    n_cells = 100
    n_genes = 50
    n_clones = 4
    n_extra_per_observed = 1

    sim_sca = scatrex.SCATrEx(model=models.cna)

    observed_tree_args = dict(
        n_nodes=n_clones, node_weights=np.array([0.25, 0.25, 0.25, 0.25])
    )
    observed_tree_params = dict(
        n_regions=20, min_cn=1, min_nevents=2, max_nevents_frac=0.3
    )
    sim_sca.simulate_tree(
        observed_tree=None,
        n_extra_per_observed=n_extra_per_observed,
        n_genes=n_genes,
        seed=seed,
        observed_tree_params=observed_tree_params,
        observed_tree_args=observed_tree_args,
    )
    sim_sca.observed_tree.create_adata()

    sim_sca.ntssb.reset_node_parameters(
        node_hyperparams=dict(
            log_lib_size_mean=8.1,
            log_lib_size_std=0.6,
            num_global_noise_factors=0,
            frac_dosage=0.95,
            baseline_shape=1.0,
        )
    )

    sim_sca.simulate_data(n_cells=n_cells, copy=False, seed=seed)
    sim_nodes = sim_sca.ntssb.get_nodes()

    # Check data was simulated as expected
    assert sim_sca.adata.shape[0] == n_cells
    assert sim_sca.adata.shape[1] == n_genes
    assert len(sim_sca.observed_tree.tree_dict.keys()) == n_clones + 1
    assert len(sim_nodes) == n_clones + n_clones * n_extra_per_observed

    args = dict(num_global_noise_factors=2)
    sca = scatrex.SCATrEx(model=models.cna, model_args=args)
    sca.model_args = args
    sca.add_data(sim_sca.adata.raw.to_adata())
    sca.set_observed_tree(sim_sca.observed_tree)
    sca.normalize_data()
    sca.project_data()

    move_weights = {
        "add": 1.0,
        "merge": 1.0,
        "prune_reattach": 1.0,
        "pivot_reattach": 1.0,
        "swap": 1.0,
        "add_reattach_pivot": 1.0,
        "subtree_reattach": 1.0,
        "push_subtree": 1.0,
        "extract_pivot": 1.0,
        "subtree_pivot_reattach": 1.0,
        "perturb_node": 1.0,
        "perturb_globals": 1.0,
        "optimize_node": 1.0,
        "transfer_factor": 1.0,
    }

    search_kwargs = {
        "n_iters": 50,
        "move_weights": move_weights,
        "n_iters_elbo": 1,
        "local": True,
        "factor_delay": 25,
        "step_size": 0.01,
        "posterior_delay": 10,
        "mb_size": 200,
        "num_samples": 1,
        "max_nodes": 5,
        "add_rule_thres": 1.0,
        "joint_init": True,
        "anneal": False,
        "restart_step": 100,
        "window": 50,
        "every": 10,
        "threshold": 1e-3,
        "alpha": 0.0,
        "random_seed": 1,
    }

    sca.learn_tree(reset=True, search_kwargs=search_kwargs)

    nodes = sca.ntssb.get_nodes()
    assert len(nodes) >= n_clones

    assert np.var(nodes[0].variational_parameters["globals"]["log_baseline_mean"] > 0)
    for node in nodes[1:]:
        assert np.var(
            node.variational_parameters["locals"]["unobserved_factors_mean"] > 0
        )

    # Check if all plots run
    # Run in interactive mode to disable blocking behavior of matplotlib
    import matplotlib as mpl

    mpl.rcParams["interactive"] = True

    sca.ntssb.plot_tree(counts=True, show_root=True)

    sca.plot_unobserved_parameters(estimated=True, figsize=(12, 3), step=2)
    sca.plot_unobserved_parameters(
        estimated=True, figsize=(12, 3), step=0.2, name="unobserved_factors_kernel"
    )

    sca.search.plot_traces(["elbo", "n_nodes"])
    sc.pl.heatmap(
        sca.adata,
        groupby="scatrex_node",
        var_names=sca.adata.var_names,
        layer="scatrex_noise",
        figsize=(16, 8),
    )
    sc.pl.heatmap(
        sca.adata,
        groupby="scatrex_node",
        var_names=sca.adata.var_names,
        layer="scatrex_cnvs",
        figsize=(16, 8),
    )
    sc.pl.heatmap(
        sca.adata,
        groupby="scatrex_node",
        var_names=sca.adata.var_names,
        layer="scatrex_xi",
        figsize=(16, 8),
    )

    sca.plot_proportions()

    # Comparing the CNVs with the cell state factors at each node highlights non-dosage-sensitive genes
    concordances = sca.get_concordances()
    cnv_state = sca.get_cnv_vs_state()
    nodes = sorted(list(cnv_state.keys()))

    sca.plot_cnv_vs_state(
        nodes,
        mapping=cnv_state,
        state_range=[-1.5, 1.5],
        figsize=(16, 2),
        alpha=0.8,
        concordances=concordances,
    )
    is_dosage_sensitive = np.array([True] * 100)
    is_dosage_sensitive[sim_sca.ntssb.root["node"].root["node"].inert_genes] = False

    # We can also rank the most discordant events across the whole tree
    (
        sorted_discordant_genes,
        sorted_concordances,
        sorted_nodes,
    ) = sca.get_discordant_genes(concordances)
    sca.plot_discordant_genes(
        sorted_discordant_genes,
        sorted_concordances,
        sorted_nodes=None,
        gene_annots=is_dosage_sensitive,
    )
    sca.plot_unobserved_parameters(estimated=True, gene="24", x_max=1.5, step=10)

    genes = np.arange(15, 35).astype(int).astype(str)
    sca.plot_unobserved_parameters(estimated=True, gene_names=genes, x_max=1.5, step=10)
    sca.plot_observed_parameters(gene_names=genes, show_names=True)

    sca.plot_tree(gene="17", genemode="unobserved", title="Cell state")
    sca.plot_tree(gene="17", genemode="observed", title="CNV")
