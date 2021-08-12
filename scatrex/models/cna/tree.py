import numpy as np
import matplotlib

from ...util import *
from ...ntssb.node import *
from ...ntssb.tree import *
from ...plotting import *

def get_cnv_cmap(vmax=4, vmid=2):
    # Extend amplification colors beyond 4
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cnvcmap", BLUE_WHITE_RED[:2], vmid+1)

    l = []
    # Deletions
    for i in range(vmid): # deletions
        rgb = cmap(i)
        l.append(matplotlib.colors.rgb2hex(rgb))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cnvcmap", BLUE_WHITE_RED[1:], (vmax-vmid)+1)

    # Amplifications
    for i in range(0, cmap.N):
        rgb = cmap(i)
        l.append(matplotlib.colors.rgb2hex(rgb))
    cmap = matplotlib.colors.ListedColormap(l)

    return cmap

class ObservedTree(Tree):
    def __init__(self, **kwargs):
        super(ObservedTree, self).__init__(**kwargs)
        self.cmap = get_cnv_cmap()
        self.sign_colors = {'-': 'blue', '+': 'red'}

    def add_node_params(self, n_genes=50, n_regions=5):
        C = len(self.tree_dict.keys())
        n_regions = np.max([n_regions, 3])
        # Define regions
        region_stops = np.sort(np.random.choice(np.arange(n_genes), size=n_regions, replace=False))

        # Trasverse the tree and generate events for each node
        for node in self.tree_dict:
            if self.tree_dict[node]['parent'] == '-1':
                self.tree_dict[node]['params'] = np.ones((n_genes,)) * 2
                self.tree_dict[node]['params_label'] = ''
                continue
            self.tree_dict[node]['params'] = np.array(self.tree_dict[self.tree_dict[node]['parent']]['params'])
            self.tree_dict[node]['params_label'] = ''
            parent_params = np.array(self.tree_dict[self.tree_dict[node]['parent']]['params'])

            # Sample number of regions to be affected
            n_r = np.random.choice(np.arange(1, np.max([int(2/3 * n_regions), 2])))

            # Sample regions to be affected
            affected_regions = np.random.choice(np.arange(0, n_regions), size=n_r, replace=False)

            for r in affected_regions:
                # Apply event to region
                if r > 0:
                    affected_genes = np.arange(region_stops[r-1], region_stops[r])
                elif r == 0:
                    affected_genes = np.arange(0, region_stops[r])

                if np.any(parent_params[affected_genes]) == 0:
                    continue

                while True:
                    # Sample event sign
                    s = np.random.choice([-1, 1])

                    if np.any(parent_params[affected_genes] < 2):
                        s = -1
                    elif np.any(parent_params[affected_genes] > 2):
                        s = 1

                    # Sample event magnitude
                    m = np.max([1, np.random.poisson(0.5)])

                    # Record event
                    clone_cn_events_genes = np.zeros((n_genes,))
                    clone_cn_events_genes[affected_genes] = s*m

                    if np.all(parent_params[affected_genes] + clone_cn_events_genes[affected_genes] >= 0):
                        break

                self.tree_dict[node]['params'][affected_genes] = parent_params[affected_genes] + clone_cn_events_genes[affected_genes]

                # String to show in tree
                sign = '-' if s < 0 else '+'
                affected = ""
                if len(affected_genes) > 5:
                    affected = f'{affected_genes[0]}...{affected_genes[-1]}'
                else:
                    affected = ','.join(np.array(affected_genes).astype(str))
                pref = ''
                if self.tree_dict[node]['params_label'] != '':
                    pref = self.tree_dict[node]['params_label'] + '\n'
                self.tree_dict[node]['params_label'] = pref + f'<font color="{self.sign_colors[sign]}">{sign}{m}</font>: {affected}'
                self.tree_dict[node]['params_label'] = pref + f'{sign}{m}: {affected}'

    def plot_heatmap(self, var_names=None, cmap=None, **kwds):
        if var_names is None:
            var_names = self.adata.var_names
        if cmap is None:
            cmap = self.cmap
        kwds['vmax'] = 4 if 'vmax' not in kwds else kwds['vmax']
        kwds['vmin'] = 0 if 'vmin' not in kwds else kwds['vmin']

        ax = sc.pl.heatmap(self.adata, var_names, groupby='node', cmap=cmap, show=False, **kwds)
        yticks = ax['groupby_ax'].get_yticks()
        ax['groupby_ax'].set_yticks(yticks - 0.5)
        node_labels = self.adata.obs['node'].values.tolist()
        ax['groupby_ax'].set_yticklabels(np.unique(node_labels))
        ax['groupby_ax'].get_yticks()
        plt.show()
