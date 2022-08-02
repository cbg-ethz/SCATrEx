import os
import scanpy as sc

# TOV2295 (solid tumor)
tov_scrna = sc.read_10x_mtx(os.path.join(snakemake.input['clonealign_data_path'], "T_OV2295/10X/TOV2295/outs/filtered_gene_bc_matrices/hg19/"))
# OV2295 (ascites)
ov_scrna = sc.read_10x_mtx(os.path.join(snakemake.input['clonealign_data_path'], "T_OV2295/10X/OV2295n2/outs/filtered_gene_bc_matrices/hg19/"))

# Concatenate and filter
adata = tov_scrna.concatenate(ov_scrna, batch_categories=['TOV2295', 'OV2295'])
adata.X = adata.X.toarray()
adata.layers['counts'] = adata.X

# Filter cells as in clonealign paper
sc.pp.filter_cells(adata, min_counts=20_000)
sc.pp.filter_cells(adata, min_genes=3000)
sc.pp.filter_cells(adata, max_genes=7500)
sc.pp.filter_genes(adata, min_cells=3)

adata.raw = adata

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Keep HVGs
sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    subset=False
)
hvg = adata.var.highly_variable
adata = adata.raw.to_adata()[:,hvg]

adata.write(snakemake.output['fname'])
