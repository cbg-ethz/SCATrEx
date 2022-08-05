import scicone
import numpy as np

data_file = snakemake.input["segmented_data_file"]
region_sizes = snakemake.input["region_sizes"]
scicone_path = snakemake.params["scicone_path"]
n_iters = snakemake.params["n_iters"]
n_reps = snakemake.params["n_reps"]
n_tries = snakemake.params["n_tries"]
best_tree_pkl_path = snakemake.output["best_tree_pkl_path"]
best_tree_txt_path = snakemake.output["best_tree_txt_path"]

seed = 42

# Load data
data = np.loadtxt(data_file, delimiter=',')
segmented_region_sizes = np.loadtxt(segmented_region_sizes, delimiter=',')

# Run cluster tree
sci = scicone.SCICoNE(binary_path=scicone_path,
                        output_path="",
                        persistence=False,
                        )

inferred_tree = sci.learn_tree(data, segmented_region_sizes=segmented_region_sizes,
                         n_reps=n_reps, max_tries=n_tries, cluster_tree_n_iters=n_iters, full=False, seed=seed)

# Save best tree
with open(best_tree_txt_path, 'w') as f:
        f.write(inferred_tree.tree_str)

# Save pickled object too
with open(best_tree_pkl_path, 'w') as f:
        pickle.dump(inferred_tree, f)
