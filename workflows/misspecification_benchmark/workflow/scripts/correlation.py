import scanpy as sc
import numpy as np

simulated_data = snakemake.input["simulated_data"]
simulated_clones = snakemake.input["simulated_clones"]
simulated_clones_labels = snakemake.input["simulated_clones_labels"]
output_file = snakemake.output["fname"]

adata = sc.read_csv(simulated_data)
clones_adata = sc.read_csv(simulated_clones)
clones_adata.obs["node"] = np.loadtxt(
    simulated_clones_labels, delimiter=",", dtype="str"
)

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.regress_out(adata, "total_counts")
sc.pp.scale(adata, max_value=10)

from scipy.stats import spearmanr

est_labels = []
for cell in adata.X:
    corrs = []
    for clone in clones_adata.X:
        if np.var(clone) == 0:
            corrs.append(0.0)
        else:
            corrs.append(spearmanr(cell, clone)[0])
    est_labels.append(clones_adata.obs["node"][np.argmax(np.array(corrs))])
est_labels = np.array(est_labels)

np.savetxt(output_file, est_labels, delimiter=",", fmt="%s")
