import scanpy as sc
import numpy as np

simulated_data = snakemake.input['simulated_data']
output_file = snakemake.output['fname']

adata = sc.read_csv(simulated_data)

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.regress_out(adata, 'total_counts')
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata)

est_labels = np.array(adata.obs['leiden'])

np.savetxt(output_file, est_labels, delimiter=',', fmt="%s")
