import numpy as np
import matplotlib

from .node import CNANode
from ...utils.math_utils import *
from ...ntssb.observed_tree import *
from ...plotting import *
from ...utils.tree_utils import dict_to_tree

def get_cnv_cmap(vmax=4, vmid=2):
    # Extend amplification colors beyond 4
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cnvcmap", BLUE_WHITE_RED[:2], vmid + 1
    )

    l = []
    # Deletions
    for i in range(vmid):  # deletions
        rgb = cmap(i)
        l.append(matplotlib.colors.rgb2hex(rgb))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cnvcmap", BLUE_WHITE_RED[1:], (vmax - vmid) + 1
    )

    # Amplifications
    for i in range(0, cmap.N):
        rgb = cmap(i)
        l.append(matplotlib.colors.rgb2hex(rgb))
    cmap = matplotlib.colors.ListedColormap(l)

    return cmap


class CNATree(ObservedTree):
    def __init__(self, **kwargs):
        super(CNATree, self).__init__(**kwargs)
        self.node_constructor = CNANode
        self.cmap = get_cnv_cmap()
        self.sign_colors = {"-": "blue", "+": "red"}

    def sample_kernel(self, parent_params, min_nevents=1, max_nevents_frac=0.67, min_cn=0, seed=None, **kwargs):
        n_genes = parent_params.shape[0]
        i = 0
        if seed is None:
            seed = self.seed
        while True: # Rejection sampling
            cnvs = np.array(parent_params)

            i += 1
            # Sample number of regions to be affected
            rng = np.random.default_rng(seed=seed + i)
            n_r = rng.choice(
                np.arange(
                    min_nevents, np.max([int(max_nevents_frac * self.n_regions), 2])
                )
            )

            # Sample regions to be affected
            rng = np.random.default_rng(seed=seed + i + 1)
            affected_regions = rng.choice(
                np.arange(0, self.n_regions), size=n_r, replace=False
            )

            all_affected_genes = []
            for r in affected_regions:
                # Apply event to region
                if r > 0:
                    affected_genes = np.arange(self.region_stops[r - 1], self.region_stops[r])
                elif r == 0:
                    affected_genes = np.arange(0, self.region_stops[r])

                all_affected_genes.append(affected_genes)

                if np.any(parent_params[affected_genes]) == 0:
                    continue

                # Sample event sign
                rng = np.random.default_rng(seed=seed + i + 2 + r)
                s = rng.choice([-1, 1])

                if np.any(parent_params[affected_genes] < 2):
                    s = -1
                elif np.any(parent_params[affected_genes] > 2):
                    s = 1

                # Sample event magnitude
                rng = np.random.default_rng(seed=seed + i + 3 + r)
                m = np.max([1, rng.poisson(0.5)])

                # Record event
                clone_cn_events_genes = np.zeros((n_genes,))
                clone_cn_events_genes[affected_genes] = s * m

                cnvs[affected_genes] = (
                    parent_params[affected_genes]
                    + clone_cn_events_genes[affected_genes]
                )


            all_affected_genes = np.concatenate(all_affected_genes)
            if np.all(cnvs[all_affected_genes] >= min_cn):
                break
        
        return cnvs
    
    def sample_root(self, n_genes=50, n_regions=5, **kwargs):
        self.n_regions = np.max([n_regions, 3])
        # Define regions
        rng = np.random.default_rng(self.seed)
        self.region_stops = np.sort(
            rng.choice(np.arange(n_genes), size=n_regions, replace=False)
        )
        return 2*np.ones((n_genes,))

    def _add_node_params(
        self, n_genes=50, n_regions=5, min_nevents=1, max_nevents_frac=0.67, min_cn=0
    ):
        C = len(self.tree_dict.keys())
        n_regions = np.max([n_regions, 3])
        # Define regions
        rng = np.random.default_rng(self.seed)
        region_stops = np.sort(
            rng.choice(np.arange(n_genes), size=n_regions, replace=False)
        )

        # Trasverse the tree and generate events for each node
        for node in self.tree_dict:
            if self.tree_dict[node]["parent"] == "-1":
                self.tree_dict[node]["param"] = np.ones((n_genes,)) * 2
                self.tree_dict[node]["params_label"] = ""
                continue
            parent_params = np.array(
                self.tree_dict[self.tree_dict[node]["parent"]]["param"]
            )
            i = 0
            while True:
                i += 1
                self.tree_dict[node]["param"] = np.array(
                    self.tree_dict[self.tree_dict[node]["parent"]]["param"]
                )
                self.tree_dict[node]["params_label"] = ""

                # Sample number of regions to be affected
                rng = np.random.default_rng(seed=self.seed + i)
                n_r = rng.choice(
                    np.arange(
                        min_nevents, np.max([int(max_nevents_frac * n_regions), 2])
                    )
                )

                # Sample regions to be affected
                rng = np.random.default_rng(seed=self.seed + i + 1)
                affected_regions = rng.choice(
                    np.arange(0, n_regions), size=n_r, replace=False
                )

                all_affected_genes = []
                for r in affected_regions:
                    # Apply event to region
                    if r > 0:
                        affected_genes = np.arange(region_stops[r - 1], region_stops[r])
                    elif r == 0:
                        affected_genes = np.arange(0, region_stops[r])

                    all_affected_genes.append(affected_genes)

                    if np.any(parent_params[affected_genes]) == 0:
                        continue

                    # Sample event sign
                    rng = np.random.default_rng(seed=self.seed + i + 2 + r)
                    s = rng.choice([-1, 1])

                    if np.any(parent_params[affected_genes] < 2):
                        s = -1
                    elif np.any(parent_params[affected_genes] > 2):
                        s = 1

                    # Sample event magnitude
                    rng = np.random.default_rng(seed=self.seed + i + 3 + r)
                    m = np.max([1, rng.poisson(0.5)])

                    # Record event
                    clone_cn_events_genes = np.zeros((n_genes,))
                    clone_cn_events_genes[affected_genes] = s * m

                    self.tree_dict[node]["param"][affected_genes] = (
                        parent_params[affected_genes]
                        + clone_cn_events_genes[affected_genes]
                    )

                    # String to show in tree
                    sign = "-" if s < 0 else "+"
                    affected = ""
                    if len(affected_genes) > 5:
                        affected = f"{affected_genes[0]}...{affected_genes[-1]}"
                    else:
                        affected = ",".join(np.array(affected_genes).astype(str))
                    pref = ""
                    if self.tree_dict[node]["params_label"] != "":
                        pref = self.tree_dict[node]["params_label"] + "\n"
                    self.tree_dict[node]["params_label"] = (
                        pref
                        + f'<font color="{self.sign_colors[sign]}">{sign}{m}</font>: {affected}'
                    )
                    self.tree_dict[node]["params_label"] = (
                        pref + f"{sign}{m}: {affected}"
                    )

                all_affected_genes = np.concatenate(all_affected_genes)
                if np.all(self.tree_dict[node]["param"][all_affected_genes] >= min_cn):
                    break

        self.tree = dict_to_tree(self.tree_dict)

        self.create_adata()

    def get_affected_genes(self):
        return np.where(np.any(self.adata.X != 2, axis=0))[0]

    def set_neutral_nodes(self, thres=0.95, neutral_level=2):
        for node in self.tree_dict:
            self.tree_dict[node]["is_neutral"] = False
            cnvs = self.tree_dict[node]["param"].ravel()
            frac_neutral = np.sum(cnvs == neutral_level) / cnvs.size
            if frac_neutral > thres:
                self.tree_dict[node]["is_neutral"] = True

    def set_neutral_weights(self, weight=1e-6):
        for node in self.tree_dict:
            if self.tree_dict[node]["is_neutral"]:
                self.tree_dict[node]["weight"] = weight

    def plot_heatmap(self, var_names=None, cmap=None, **kwds):
        if var_names is None:
            var_names = self.adata.var_names
        if cmap is None:
            cmap = self.cmap
        kwds["vmax"] = 4 if "vmax" not in kwds else kwds["vmax"]
        kwds["vmin"] = 0 if "vmin" not in kwds else kwds["vmin"]

        if kwds["vmax"] > 4:
            cmap = get_cnv_cmap(vmax=kwds["vmax"])

        ax = sc.pl.heatmap(
            self.adata, var_names, groupby="node", cmap=cmap, show=False, **kwds
        )
        yticks = ax["groupby_ax"].get_yticks()
        ax["groupby_ax"].set_yticks(yticks - 0.5)
        node_labels = self.adata.obs["node"].values.tolist()
        ax["groupby_ax"].set_yticklabels(np.unique(node_labels))
        ax["groupby_ax"].get_yticks()
        plt.show()
