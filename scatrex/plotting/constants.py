import matplotlib
import string

CLONES_PAL = [
    "#7630A9B3",
    "#CC79A7B3",
    "#D55E00B3",
    "#E69F00B3",
    "#0072B2B3",
    "#56B4E9B3",
    "#56DDA0B3",
    "#49BD48B3",
    "#5E752BB3",
    "#D4D948B3",
    "#A8722FB3",
    "#A83D2FB3",
    "#CF1F1FB3",
]
BLUE_WHITE_RED = ["#2040C8", "white", "#EE241D"]
CNV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "cnv_cmap", BLUE_WHITE_RED
)
PATHWAY_CMAP = matplotlib.cm.viridis
PATHWAY_NORM = matplotlib.colors.Normalize(vmin=0.0, vmax=5.0)
PATHWAY_CMAPPER = matplotlib.cm.ScalarMappable(norm=PATHWAY_NORM, cmap=PATHWAY_CMAP)
LABEL_COLORS_DICT = dict(
    zip(list(string.ascii_uppercase)[: len(CLONES_PAL)], CLONES_PAL)
)
