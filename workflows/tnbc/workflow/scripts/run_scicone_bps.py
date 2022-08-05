import scicone
import numpy as np

data_file = snakemake.input["data_path"]
sp_vec = snakemake.input["sp_vec"]
scicone_path = snakemake.params["scicone_path"]
window_size = snakemake.params["window_size"]
threshold = snakemake.params["threshold"]
segmented_data_path = snakemake.output["segmented_data"]
regions_path = snakemake.output["regions"]
region_sizes_path = snakemake.output["region_sizes"]

sci = scicone.SCICoNE(
    binary_path=scicone_path,
    output_path="",
    persistence=False,
)

# Load data
data = np.loadtxt(data_file, delimiter=",")
sp_vec = np.loadtxt(sp_vec, delimiter=",")

# Run breakpoint detection
bps = sci.detect_breakpoints(
    data,
    window_size=window_size,
    threshold=threshold,
    sp=sp_vec,
    compute_lr=False,
    compute_sp=False,
)

# Make region-level data
segmented_data = sci.condense_regions(data, bps["segmented_region_sizes"])

# Save segmented data
np.savetxt(segmented_data_path, segmented_data, delimiter=",")
np.savetxt(regions_path, bps["segmented_regions"], delimiter=",")
np.savetxt(region_sizes_path, bps["segmented_region_sizes"], delimiter=",")
