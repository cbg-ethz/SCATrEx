rule annotate_dna_data:
    input:
        ov2295_cnvs_path = 'results/data/ov2295_clone_cn.csv',
        ov2295_clone_clusters_path = 'results/data/ov2295_clone_clusters.csv',
    output:
        cnv_matrix = 'results/data/annotated_dna_data/cnv_matrix.csv',
        chrs = 'results/data/annotated_dna_data/chrs.csv',
        clone_sizes = 'results/data/annotated_dna_data/clone_sizes.csv'
    params:
        autosomes_only = config['autosomes_only']
    run:
        import scatrex
        import pandas as pd

        cnv_data = pd.read_csv(input.ov2295_cnvs_path)
        cnv_matrix, bins_df = scatrex.util.convert_tidy_to_matrix(cnv_data)
        bins_df = scatrex.util.annotate_bins(bins_df)
        cnv_matrix, chrs = scatrex.util.annotate_matrix(cnv_matrix, bins_df)

        cell_clone_assignments = pd.read_csv(input.ov2295_clone_clusters_path)
        clone_sizes = cell_clone_assignments['clone_id'].value_counts().sort_index()

        if params.autosomes_only:
            keep = np.where(np.logical_and(chrs != 'X', chrs != 'Y'))[0]
            cnv_matrix = cnv_matrix[cnv_matrix.columns[keep]]
            chrs = chrs[keep]

        cnv_matrix.to_csv(output.cnv_matrix)
        chrs.to_csv(output.bins_df)
        clone_sizes.to_csv(output.clone_sizes)


rule create_original_tree:
    output:
        fname = 'results/original/original_input_tree.json'
    input:
        cnv_matrix_path = rules.annotate_dna_data.output.cnv_matrix,
        clone_sizes_path = rules.annotate_dna_data.output.clone_sizes
    script:
        'scripts/make_cnv_tree.py'


rule filter_rna_data:
    output:
        fname = 'results/data/scrna/filtered_adata.h5ad',
    input:
        clonealign_data_path = directory('results/data/pbmc3k_filtered_gene_bc_matrices/'),
    script:
        'scripts/filter_rna_data.py'


rule align_data:
    output:
        aligned_observed_tree = 'results/{mode}/aligned_{mode}_input_tree.json',
        aligned_adata = 'results/{mode}/aligned_{mode}_adata.h5ad',
    input:
        observed_tree =  'results/{mode}/{mode}_input_tree.json',
        filtered_adata = rules.filter_rna_data.output.fname
    run:
        # Intersect with scDNA data and keep its order
        intersection = [x for x in observed_tree.adata.var_names if x in frozenset(adata.var_names)]
        # intersection = list(set(observed_tree.adata.var_names).intersection(set(adata.var_names)))
        int_chrs = pd.DataFrame(chrs[:,np.newaxis].T, columns=cnv_matrix.columns)[intersection]
        argsorted_genes = np.argsort(int_chrs.values.astype(int))
        sorted_genes = int_chrs.columns[argsorted_genes].ravel()
        sorted_chrs = int_chrs[sorted_genes]
        # Apply filter to scDNA data
        new_observed_tree = deepcopy(observed_tree)
        new_observed_tree.subset_genes(sorted_genes)
        new_observed_tree.adata.var['chr'] = sorted_chrs.T
        # init_observed_tree.subset_genes(sorted_genes)
        # init_observed_tree.adata.var['chr'] = sorted_chrs.T

        # Apply filter to scRNA data
        adata = adata[:, sorted_genes.ravel()]
        adata.var['chr'] = sorted_chrs.T
        sorted_chrs_dict = dict()
        for chr in range(1,23):
            genes = np.where(sorted_chrs.values.ravel()==str(chr))[0]
            sorted_chrs_dict[str(chr)] = sorted_chrs.columns[genes].tolist()

        # Clip CNVs to 8 as in the clonealign paper
        for node in new_observed_tree.tree_dict:
            new_observed_tree.tree_dict[node]['params'] = np.clip(new_observed_tree.tree_dict[node]['params'], 0, 8)

        new_observed_tree.create_adata(var_names=new_observed_tree.adata.var_names)

        # Remove weight from root
        new_observed_tree.tree_dict['root']["weight"] = 1e-300
