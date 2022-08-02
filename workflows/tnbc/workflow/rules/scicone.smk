rule detect_cna_breakpoints:
    input:
        scdna_counts = 'results/data/scdna_counts.csv'
    output:
        segmented_data = 'results/scicone/segmented_data.csv',
        regions = 'results/scicone/segmented_regions.txt',
        region_sizes = 'results/scicone/segmented_region_sizes.txt',
    params:
        scicone_build_path = config["scicone"]["build_path"]
    script:
        "scripts/run_scicone_bps.py"

rule learn_cna_tree:
    input:
        segmented_data = 'results/scicone/segmented_data.csv',
        regions = 'results/scicone/segmented_regions.txt',
        region_sizes = 'results/scicone/segmented_region_sizes.txt',
    output:
        cna_tree = 'results/scicone/scicone_tree.pkl',
    params:
        n_iters = config["scicone"]["n_iters"],
        n_reps = config["scicone"]["n_reps"],
    script:
        "scripts/run_scicone_tree.py"
