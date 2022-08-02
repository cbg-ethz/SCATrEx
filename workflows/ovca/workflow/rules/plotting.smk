localrules: all

rule plot_original:
    input:
        fname_original_cnvmat = 'results/original/plots/original_cnv_matrix.pdf',
        fname_original_cnvtree_counts = 'results/original/plots/original_cnv_tree_counts.pdf',
        fname_original_cnvtree_nocounts = 'results/original/plots/original_cnv_tree_nocounts.pdf',
        fname_original_scatrextree_counts = 'results/original/plots/original_scatrex_tree_counts.pdf',
        fname_original_scatrextree_nocounts = 'results/original/plots/original_scatrex_tree_nocounts.pdf',
        # fname_original_stateevents = 'results/original/plots/original_stateevents.pdf',
        fname_original_cnv_exp = 'results/original/plots/original_cnv_exp.pdf',
        fname_original_chr14 = 'results/original/plots/original_chr14.pdf',
        fname_node_concordances_fname = 'results/original/plots/original_node_concordances.pdf',
        fname_top_discordances_fname = 'results/original/plots/original_top_discordances.pdf'
    output:
        fname = 'results/original/plots/done.txt',
    shell:
        "touch {output.fname}"

# Test-specific plots are subject to change
rule plot_tests:
    input:
        scatrex_tree = 'results/{test_mode}/scatrex/{test_mode}_best_sca.pkl'
    output:
        fname = 'results/{test_mode}/plots/done.txt',
    shell:
        "python ../scripts/{{test_mode}}_plots.py; \
        touch {output.fname}"

rule plot_cnv_matrix:
    input:
        observed_tree = 'results/{mode}/{mode}_input_tree.json'
    output:
        fname = 'results/{mode}/plots/{mode}_cnv_matrix.pdf'
    params:
        vmax = config['cnv_max'],
        axislabel_fontsize = config['plotting']['axislabel_fontsize'],
        cnv_matrix_width = config['plotting']['cnv_matrix_width'],
        cnv_matrix_height = config['plotting']['cnv_matrix_height'],
        dpi = config['plotting']['dpi']
    run:
        sc.set_figure_params(dpi=params.dpi)
        ax = sc.pl.heatmap(new_observed_tree.adata, groupby='node',
                       figsize=(params.cnv_matrix_width, cnv_matrix_height),
                       cmap=scatrex.models.cna.tree.get_cnv_cmap(vmax=params.vmax),
                       vmin=0, vmax=params.vmax, show=False)
        ax["groupby_ax"].set_ylabel("Cells", fontsize=params.axislabel_fontsize)
        ax["heatmap_ax"].set_xlabel("Genome", fontsize=params.axislabel_fontsize)
        plt.savefig(output.fname)

rule plot_cnv_tree:
    input:
        observed_tree = 'results/{mode}/{mode}_input_tree.json'
    output:
        fname_counts = 'results/{mode}/plots/{mode}_cnv_tree_counts.pdf',
        fname_nocounts = 'results/{mode}/plots/{mode}_cnv_tree_nocounts.pdf'
    params:
        nodelabel_fontsize = config['plotting']['nodelabel_fontsize'],
        nodesize_fontsize = config['plotting']['nodesize_fontsize']
    run:
        g_counts = observed_tree.plot_tree(counts=True, label_fontsize=params.nodelabel_fontsize,
                                            size_fontsize=params.nodesize_fontsize)
        g_nocounts = observed_tree.plot_tree(counts=False, label_fontsize=params.nodelabel_fontsize,
                                            size_fontsize=params.nodesize_fontsize)
        g_counts.render(output.fname_counts, cleanup=True)
        g_nocounts.render(output.fname_counts, cleanup=True)

rule plot_scatrex_tree:
    input:
        scatrex_tree = 'results/{mode}/scatrex/{mode}_best_sca.pkl'
    output:
        fname_counts = 'results/{mode}/plots/{mode}_scatrex_tree_counts.pdf',
        fname_nocounts = 'results/{mode}/plots/{mode}_scatrex_tree_nocounts.pdf',
    params:
        nodelabel_fontsize = config['plotting']['nodelabel_fontsize'],
        nodesize_fontsize = config['plotting']['nodesize_fontsize']
    run:
        g_counts = scatrex_tree.plot_tree(counts=True, label_fontsize=params.nodelabel_fontsize,
                                            size_fontsize=params.nodesize_fontsize)
        g_nocounts = scatrex_tree.plot_tree(counts=False, label_fontsize=params.nodelabel_fontsize,
                                            size_fontsize=params.nodesize_fontsize)
        g_counts.render(output.fname_counts, cleanup=True)
        g_nocounts.render(output.fname_counts, cleanup=True)

#rule plot_cell_states:

rule plot_cnv_exp:
    input:
        scatrex_tree = 'results/{mode}/scatrex/{mode}_best_sca.pkl'
    output:
        fname = 'results/{mode}/plots/{mode}_cnv_exp.pdf'
    params:
        vmax = config['cnv_max'],
        axislabel_fontsize = config['plotting']['axislabel_fontsize'],
        boxplot_width = config['plotting']['boxplot_width'],
        boxplot_height = config['plotting']['boxplot_height'],
        dpi = config['plotting']['dpi'],
    run:
        # The copy number state-specific expression distribution reflects good clonal assignments
        cnv_exp = sca.get_cnv_exp(max_level=params.vmax, method='scatrex')

        cnv_levels = list(cnv_exp.keys())
        cnv_levels_labels = [cnv_exp[l]['label'] for l in cnv_levels]
        exp_levels = [cnv_exp[l]['exp'] for l in cnv_levels]

        plt.figure(figsize=(params.boxplot_width, params.boxplot_height), dpi=params.dpi)
        box_plot = plt.boxplot(x=exp_levels, patch_artist=True)
        for median in box_plot['medians']:
            median.set_color('black')

        cm = scatrex.models.cna.tree.get_cnv_cmap(vmax=params.vmax)
        xx = np.array(range(1,1+len(cnv_levels)))
        for i, box in enumerate(box_plot['boxes']):
            color = cm.colors[int(cnv_levels[i])]
            box.set_facecolor(color)
            box.set_alpha(0.6)
            if cnv_levels[i] == 2:
                color = 'lightgray'
            plt.scatter(xx[i]+np.random.normal(0,0.07,size=len(exp_levels[i])), exp_levels[i], alpha=0.6,
                        color=color)
        plt.yscale('log')
        plt.ylabel('Mean log expression', fontsize=params.axislabel_fontsize)
        plt.xlabel('Copy number state', fontsize=params.axislabel_fontsize)
        plt.xticks(cnv_levels, labels=cnv_levels_labels)
        plt.savefig(output.fname)

rule plot_chr_14:
    input:
        scatrex_tree = 'results/original/scatrex/original_best_sca.pkl'
    output:
        fname = 'results/original/plots/original_chr14.pdf'
    params:
        vmax = config['cnv_max'],
        axislabel_fontsize = config['plotting']['axislabel_fontsize'],
        tick_fontsize = config['plotting']['tick_fontsize'],
        boxplot_width = config['plotting']['boxplot_width'],
        boxplot_height = config['plotting']['boxplot_height'],
        dpi = config['plotting']['dpi'],
    run:
        # Gene expression of held-out chr 14 in cells assigned to clone C versus D, E
        genes = sorted_chrs_dict['14']
        exp_avgs = []
        clones = ['C', 'D', 'E']
        cns = ['3', '2', '2']
        for clone in clones:
            cells_in_clone = sca.adata.obs.query(f"scatrex_obs_node == '{clone}'").index
            exp_avgs.append(np.mean(original_adata[cells_in_clone, target_genes].X, axis=0))

        plt.figure(figsize=(params.boxplot_width, params.boxplot_height), dpi=params.dpi)
        box_plot = plt.boxplot(x=exp_avgs, patch_artist=True)
        for median in box_plot['medians']:
            median.set_color('black')

        cm = scatrex.models.cna.tree.get_cnv_cmap(vmax=params.vmax)
        xx = np.array(range(1,1+len(clones)))
        for i, box in enumerate(box_plot['boxes']):
            color = cm.colors[int(cns[i])]
            box.set_facecolor(color)
            box.set_alpha(0.6)
            if cns[i] == '2':
                color = 'lightgray'
            plt.scatter(xx[i]+np.random.normal(0,0.07,size=len(exp_avgs[i])), exp_avgs[i], alpha=0.6,
                        color=color)

        plt.yscale('log')
        plt.ylabel('Mean log expression', fontsize=params.axislabel_fontsize)
        plt.xlabel('Clone', fontsize=params.axislabel_fontsize)
        plt.yticks(fontsize=params.tick_fontsize)
        plt.xticks(range(1,1+len(clones)), labels=clones, fontsize=params.tick_fontsize)
        plt.title("Chromosome 14", fontsize=params.axislabel_fontsize)
        # plt.legend(box_plot["boxes"][:-1], cns, title='Copy number', fontsize=14)
        plt.savefig(output.fname)

rule plot_node_concordances:
    input:
        scatrex_tree = 'results/original/scatrex/original_best_sca.pkl'
    output:
        node_concordances_fname = 'results/original/plots/original_node_concordances.pdf',
        top_discordances_fname = 'results/original/plots/original_top_discordances.pdf'
    params:
        vmax = config['cnv_max'],
        cellstate_cmap = config['plotting']['cellstate_cmap'],
        axislabel_fontsize = config['plotting']['axislabel_fontsize'],
        boxplot_width = config['plotting']['boxplot_width'],
        boxplot_height = config['plotting']['boxplot_height'],
        dpi = config['plotting']['dpi']
    run:
        # Show concordance of gene expression with CNVs per node
        concs = get_concordances(sca, key="scatrex_node")

        nodes = list(concs.keys())
        levels = list(concs.values())

        plt.figure(figsize=(params.boxplot_width, params.boxplot_height), dpi=params.dpi)
        box_plot = plt.boxplot(x=levels, patch_artist=True)
        for median in box_plot['medians']:
            median.set_color('black')

        xx = np.array(range(1,1+len(levels)))
        for i, box in enumerate(box_plot['boxes']):
            color = sca.ntssb.node_dict[nodes[i]]['node'].tssb.color
            box.set_alpha(0.6)
            box.set_facecolor(color)
            plt.scatter(xx[i]+np.random.normal(0,0.05,size=len(levels[i])), levels[i], alpha=0.6, color=color)
        plt.ylabel('Concordance', fontsize=params.axislabel_fontsize)
        plt.xlabel('Node', fontsize=params.axislabel_fontsize)
        plt.xticks(xx, labels=nodes, fontsize=params.tick_fontsize)
        plt.yticks(fontsize=params.tick_fontsize)
        plt.savefig(output.node_concordances_fname)

        # Top 5 cell state events in subclusters
        top_5_genes = dict()
        nodes = sca.ntssb.get_nodes()[1:]
        nodes = np.array(nodes)[np.argsort([n.label for n in nodes])]
        for node in nodes:
            if '-' in node.label:
                unobs = node.variational_parameters["locals"]["unobserved_factors_kernel_log_mean"]
                std = np.exp(node.variational_parameters["locals"]["unobserved_factors_kernel_log_std"])
                unobs_kern = np.exp(unobs + std**2 / 2)
                top_5_genes[node.label] = np.argsort(unobs_kern)[::-1][:5]
        all_top_5_genes = [item for sublist in top_5_genes.values() for item in sublist]
        mat = np.zeros((len(top_5_genes),len(list(top_5_genes.values())[0]) * len(top_5_genes)))
        for i, node in enumerate(top_5_genes):
            mat[i] = sca.ntssb.node_dict[node]['node'].variational_parameters["locals"]["unobserved_factors_mean"][all_top_5_genes]

        plt.figure(figsize=(3.5,6), dpi=params.dpi)
        plt.pcolormesh(mat.T, vmax=2.5, vmin=-2.5, cmap=params.cellstate_cmap)
        plt.xticks(np.arange(0.5,mat.shape[0]+0.5), labels=list(top_5_genes.keys()), fontsize=params.tick_fontsize)
        plt.yticks(np.arange(0.5,mat.shape[1]+0.5), labels=sca.adata.var_names[all_top_5_genes], fontsize=params.tick_fontsize)
        cb = plt.colorbar()
        cb.ax.set_ylabel('Cell state', rotation=270, fontsize=params.axislabel_fontsize)
        plt.savefig(output.top_discordances_fname)
