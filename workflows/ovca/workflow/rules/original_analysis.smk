localrules: remove_chr_14_segment

rule remove_chr_14_segment:
    input:
        aligned_observed_tree = 'results/original/aligned_original_input_tree.json',
        aligned_adata = 'results/original/aligned_original_adata.h5ad'
    output:
        heldout_observed_tree = 'results/original/heldout_original_input_tree.json',
        heldout_adata = 'results/original/heldout_adata.h5ad'
    run:
        # Get genes in chr 14 for which C has 3 copies
        clone_c = np.where(new_observed_tree.adata.obs['node'] == 'C')[0][0]
        chr_14_start = np.where(new_observed_tree.adata.var_names == sorted_chrs_dict['14'][0])[0]
        target_genes = new_observed_tree.adata.var_names[chr_14_start + np.where(new_observed_tree.adata[clone_c,sorted_chrs_dict['14']].X == 3)[1]]

        # Remove them
        new_observed_tree.subset_genes(sorted_genes_subs)
        adata = adata[:, sorted_genes_subs.ravel()]

rule run_original:
    output:
        fname = 'results/original/scatrex/runs/original_sca_{run_id}.pkl'
    input:
        observed_tree = rules.remove_chr_14_segment.output.heldout_observed_tree,
        adata = rules.remove_chr_14_segment.output.heldout_adata
    params:
        n_iters = config["scatrex"]["n_iters"],
        n_factors = config["scatrex"]["n_factors"],
    script:
        'scripts/run_scatrex.py'
