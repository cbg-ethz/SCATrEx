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

        # Define regions
        region_stops = np.arange(0, n_genes+1, int(n_genes/n_regions))[1:]

        # Trasverse the tree and generate events for each node
        for node in self.tree_dict:
            if self.tree_dict[node]['parent'] == '-1':
                self.tree_dict[node]['params'] = np.ones((n_genes,)) * 2
                self.tree_dict[node]['params_label'] = ''
                continue

            # Sample region to be affected
            r = np.random.choice(np.arange(0, n_regions))

            if r > 0:
                affected_genes = np.arange(region_stops[r-1], region_stops[r])
            elif r == 0:
                affected_genes = np.arange(0, region_stops[r])

            # Sample event sign
            s = np.random.choice([-1, 1])

            # Sample event magnitude
            m = np.max([1, np.random.poisson(0.5)])

            # Record event
            clone_cn_events_genes = np.zeros((n_genes,))
            clone_cn_events_genes[affected_genes] = s*m

            events_genes = np.zeros((n_genes,))
            self.tree_dict[node]['params'] = self.tree_dict[self.tree_dict[node]['parent']]['params'] + clone_cn_events_genes

            # String to show in tree
            sign = '-' if s < 0 else '+'
            affected = ""
            if len(affected_genes) > 5:
                affected = f'{affected_genes[0]}...{affected_genes[-1]}'
            else:
                affected = ','.join(affected_genes)
            self.tree_dict[node]['params_label'] = f'<font color="{self.sign_colors[sign]}">{sign}{m}</font>: {affected}'
            self.tree_dict[node]['params_label'] = f'{sign}{m}: {affected}'
