from pybiomart import Server
import pandas as pd


def convert_tidy_to_matrix(tidy_df, rows="single_cell_id", columns="copy_number"):
    # Takes a tidy dataframe specifying the CNVs of cells along genomic bins
    # and converts it to a cell by bin matrix
    cell_df = tidy_df.loc[tidy_df[rows] == tidy_df[rows][0]]
    bins_df = cell_df.drop(columns=[columns, rows], inplace=False)
    tidy_df["bin_id"] = np.tile(bins_df.index, tidy_df[rows].unique().size)
    matrix = tidy_df[[columns, rows, "bin_id"]].pivot_table(
        values=columns, index=rows, columns="bin_id"
    )

    return matrix, bins_df


def annotate_bins(bins_df):
    # Takes a dataframe of genomic regions and returns an ordered list of full genes in each region
    server = Server("www.ensembl.org", use_cache=False)
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    gene_coordinates = dataset.query(
        attributes=[
            "chromosome_name",
            "start_position",
            "end_position",
            "external_gene_name",
        ],
        filters={
            "chromosome_name": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                "X",
                "Y",
            ]
        },
        use_attr_names=True,
    )
    # Drop duplicate genes
    gene_coordinates.drop_duplicates(subset="external_gene_name", ignore_index=True)

    annotated_bins = bins_df.copy()
    annotated_bins["genes"] = [list() for _ in range(annotated_bins.shape[0])]

    bin_size = bins_df["end"][0] - bins_df["start"][0]
    for index, row in gene_coordinates.iterrows():
        gene = row["external_gene_name"]
        if pd.isna(gene):
            continue
        start_bin_in_chr = int(row["start_position"] / bin_size)
        stop_bin_in_chr = int(row["end_position"] / bin_size)
        chromosome = str(row["chromosome_name"])
        chr_start = np.where(bins_df["chr"] == chromosome)[0][0]
        start_bin = start_bin_in_chr + chr_start
        stop_bin = stop_bin_in_chr + chr_start

        if stop_bin < annotated_bins.shape[0]:
            if np.all(annotated_bins.iloc[start_bin:stop_bin].chr == chromosome):
                for bin in range(start_bin, stop_bin + 1):
                    annotated_bins.loc[bin, "genes"].append(gene)

    return annotated_bins


def annotate_matrix(matrix, annotated_bins):
    # Takes a dataframe of cells by bins and a dataframe with gene lists for each bin
    # and returns a dataframe of cells by genes
    df_list = []
    chrs = []
    for bin, row in annotated_bins.iterrows():
        genes = row["genes"]
        chr = row["chr"]
        if len(genes) > 0:
            df_list.append(
                pd.concat([matrix[bin]] * len(genes), axis=1, ignore_index=True).rename(
                    columns=dict(zip(range(len(genes)), genes))
                )
            )
            chrs.append([chr] * df_list[-1].shape[1])
    chrs = np.concatenate(chrs)
    df = pd.concat(df_list, axis=1)
    chrs = chrs[np.where(~df.columns.duplicated())[0]]
    df = df.loc[:, ~df.columns.duplicated()]
    return df, chrs

