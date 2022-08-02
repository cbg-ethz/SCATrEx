localrules: create_test_pruned_tree, create_test_linear_tree, create_test_undetected_tree

rule create_test_pruned_tree:
    output:
        fname = 'results/test_pruned/test_pruned_input_tree.json'
    input:
        original_tree = rules.create_original_tree.output.fname
    run:
        # Remove E
        tree_dict = json.reads(input.original_tree)
        tree_dict['C']['size'] += tree_dict['E']['size']
        del modified_tree_dict['E']


rule create_test_linear_tree:
    output:
        fname = 'results/test_linear/test_linear_input_tree.json'
    input:
        original_tree = rules.create_original_tree.output.fname
    run:
        # Change branching into linear subtree
        tree_dict = json.reads(input.original_tree)
        tree_dict['E']['parent'] = 'D'

rule create_test_undetected_tree:
    output:
        fname = 'results/test_undetected/test_undetected_input_tree.json'
    input:
        original_tree = rules.create_original_tree.output.fname
    run:
        # Change branching into linear subtree
        tree_dict = json.reads(input.original_tree)
        tree_dict['E']['parent'] = 'D'

rule run_tests:
    output:
        fname = 'results/test_{mode}/scatrex/runs/test_{mode}_sca_{run_id}.pkl'
    input:
        observed_tree = 'results/test_{mode}/aligned_test_{mode}_input_tree.json',
        adata = 'results/test_{mode}/aligned_test_{mode}_adata.h5ad',
    params:
        n_iters = config["scatrex"]["n_iters"],
        n_factors = config["scatrex"]["n_factors"],
    script:
        'scripts/run_scatrex.py'
