import sys
import os
import glob

collectionName = snakemake.input["fname_list"][0]
for _ in range(4):  # go up to results/methods
    collectionName = os.path.dirname(collectionName)

outFileName = snakemake.output["fname"]

pattern = (
    collectionName + "/**/*scores.csv"
)  # all files matching this pattern are processed
fileList = glob.glob(pattern, recursive=True)

rows = []
for filename in fileList:
    # Parse filename
    l = filename.split("/")
    method = l[2]
    ll = l[3].split("_")
    n_clones = ll[0][1:]
    n_genes = ll[1][1:]
    n_cells = ll[2][1:]
    lll = l[4].split("_")
    n_extras = lll[0][1:]
    n_factors = lll[1][1:]
    llll = l[5].split("_")
    rep_id = llll[0][1:]

    # Parse scores
    with open(filename) as f:
        lines = f.read().splitlines()
        clone_accuracy, clone_ari, node_accuracy, node_ari = lines[1].split(",")

    rows.append(
        [
            method,
            n_clones,
            n_genes,
            n_cells,
            n_extras,
            n_factors,
            rep_id,
            clone_accuracy,
            clone_ari,
            node_accuracy,
            node_ari,
        ]
    )

columns = [
    "method",
    "n_clones",
    "n_genes",
    "n_cells",
    "n_extras",
    "n_factors",
    "rep_id",
    "clone_accuracy",
    "clone_ari",
    "node_accuracy",
    "node_ari",
]

import pandas as pd

scores = pd.DataFrame.from_records(rows, columns=columns)
scores = pd.melt(
    scores,
    id_vars=[
        "method",
        "n_clones",
        "n_genes",
        "n_cells",
        "n_extras",
        "n_factors",
        "rep_id",
    ],
    value_vars=["clone_accuracy", "clone_ari", "node_accuracy", "node_ari"],
    var_name="score",
)
scores.to_csv(outFileName, index=False)
