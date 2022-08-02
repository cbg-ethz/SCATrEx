localrules: create_test_pivot_tree

rule create_test_pivot_tree:
    output:
        fname = 'results/test_pivot/test_pivot_input_tree.json'
    input:
        original_tree = 'results/scicone/annotated_scicone_tree.pkl',
    run:
        remove_node(cna_tree, "65", region_gene_map)

rule run_tests:
    output:
        fname = 'results/test_pivot/scatrex/runs/test_pivot_sca_{run_id}.pkl'
    input:
        observed_tree = 'results/test_pivot/aligned_test_pivot_input_tree.json',
        adata = 'results/test_pivot/aligned_test_pivot_adata.h5ad',
    params:
        n_iters = config["scatrex"]["n_iters"],
        n_factors = config["scatrex"]["n_factors"],
    script:
        'scripts/run_scatrex.py'
