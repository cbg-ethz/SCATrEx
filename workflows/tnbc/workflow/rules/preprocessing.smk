rule annotate_dna_data:
    input:
        cna_tree = 'results/scicone/scicone_tree.pkl',
    output:
        annotated_dna_tree = 'results/scicone/annotated_scicone_tree.pkl',
    run:
        # Set gene names
        excluded_bins = np.loadtxt('/cluster/work/bewi/members/pedrof/data/SA501/cnv/excluded_bins.csv', delimiter=',')
        chr_stops = pd.read_csv('/cluster/work/bewi/members/pedrof/data/SA501/cnv/chr_stops.tsv', sep="\t", index_col=1)
        chr_stops_dict = dict(zip(chr_stops.index, chr_stops.values.ravel()))
        regions = np.loadtxt('/cluster/work/bewi/members/pedrof/SCICoNE_results/vancouver/sleek/breakpoint_detection/vancouver__segmented_regions.txt', delimiter=',')

        # Online
        region_gene_map = scicone.utils.get_region_gene_map(150000,
                                                            chr_stops_dict,
                                                            regions,
                                                            np.where(excluded_bins)[0])

        cna_tree.set_gene_event_dicts(region_gene_map)

        def set_node_gene_cnvs(self, region_gene_map):
            # Set root state
            all_genes = [x for xs in region_gene_map.values() for x in xs]
            is_duplicated = pd.DataFrame(index=all_genes, data=all_genes).duplicated(keep=False)
            is_duplicated = is_duplicated[~is_duplicated.index.duplicated(keep='first')]
            all_genes = list(pd.DataFrame(index=all_genes, data=all_genes).drop_duplicates(keep=False).values.ravel())
            n_genes = len(all_genes)
            self.node_dict['0']['gene_cnv'] = pd.DataFrame(data=np.ones((1,n_genes)), columns=all_genes)
            for region, state in enumerate(self.outputs['region_neutral_states']):
                genes = region_gene_map[region]
                # Remove duplicates
                genes = list(np.array(genes)[np.where(~is_duplicated.loc[genes])[0]])
                self.node_dict['0']['gene_cnv'][genes] = state

            for node_id in self.node_dict:
                if node_id != '0':
                    self.node_dict[node_id]['gene_cnv'] = None
                    parent_id = self.node_dict[node_id]['parent_id']
                    genes_event = pd.DataFrame(data=np.zeros((1,n_genes)), columns=all_genes)
                    for gene in self.node_dict[node_id]['gene_event_dict']:
                        if not is_duplicated.loc[gene]:
                            genes_event[gene] = int(self.node_dict[node_id]['gene_event_dict'][gene])
                    self.node_dict[node_id]['gene_cnv'] = self.node_dict[parent_id]['gene_cnv'] + genes_event

        set_node_gene_cnvs(cna_tree, region_gene_map)

        # Get chromosome of each gene
        server = Server("www.ensembl.org", use_cache=False)
        dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
        gene_coordinates = dataset.query(
            attributes=[
                "chromosome_name",
                "external_gene_name",
            ],
            filters={
                "chromosome_name": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    "X",
                    "Y",
                ]
            },
            use_attr_names=True,
        )
        # Drop duplicate genes
        gene_coordinates.drop_duplicates(subset="external_gene_name", ignore_index=True)
        gene_coordinates = gene_coordinates[~gene_coordinates.index.duplicated(keep='first')]
        chrs = gene_coordinates.loc[all_genes]

        # Make dict
        chrs_dict = dict()
        for chr in range(1,23):
            sorted_chrs_dict[str(chr)] = chrs.columns[np.where(chrs.values.ravel()==str(chr))[0]].tolist()
        chrs_dict['X'] = chrs.columns[np.where(chrs.values.ravel()=='X')[0]].tolist()
        chrs_dict['Y'] = chrs.columns[np.where(chrs.values.ravel()=='Y')[0]].tolist()

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
