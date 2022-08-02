import scanpy as sc

mat = io.mmread(f"{snakemake.input['clonealign_data_path']}/matrix.mtx").toarray()
barcodes = pd.read_csv(
    f"{snakemake.input['clonealign_data_path']}/barcodes.tsv", sep="\t", header=None
)
genes = pd.read_csv(
    f"{snakemake.input['clonealign_data_path']}/genes.tsv", sep="\t", header=None
)

# Concatenate and filter
adata = tov_scrna.concatenate(ov_scrna, batch_categories=["TOV2295", "OV2295"])
adata.X = adata.X.toarray()
adata.layers["counts"] = adata.X

# Filter cells as in clonealign paper
sc.pp.filter_cells(adata, min_counts=20_000)
sc.pp.filter_cells(adata, min_genes=3000)
sc.pp.filter_cells(adata, max_genes=7500)
sc.pp.filter_genes(adata, min_cells=3)

adata.raw = adata

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Keep HVGs
sc.pp.highly_variable_genes(adata, batch_key="batch", subset=False)
hvg = adata.var.highly_variable
adata = adata.raw.to_adata()[:, hvg]

adata.write(snakemake.output["fname"])
