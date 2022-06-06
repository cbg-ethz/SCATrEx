import matplotlib

CLONES_PAL = [
    "#7630A9",
    "#CC79A7",
    "#D55E00",
    "#E69F00",
    "#0072B2",
    "#56B4E9",
    "#56DDA0",
    "#49BD48",
    "#5E752B",
    "#D4D948",
    "#A8722F",
    "#A83D2F",
    "#CF1F1F",
]
BLUE_WHITE_RED = ["#2040C8", "white", "#EE241D"]
CNV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "cnv_cmap", BLUE_WHITE_RED
)
PATHWAY_CMAP = matplotlib.cm.viridis
PATHWAY_NORM = matplotlib.colors.Normalize(vmin=0.0, vmax=5.0)
PATHWAY_CMAPPER = matplotlib.cm.ScalarMappable(norm=PATHWAY_NORM, cmap=PATHWAY_CMAP)
