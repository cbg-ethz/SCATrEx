import string
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import matplotlib.pyplot as plt
from graphviz import Digraph

from ..plotting import constants

class Tree(ABC):
    def __init__(self, n_nodes=3, dp_alpha_subtree=1.0, alpha_decay_subtree=0.9,
                        dp_gamma_subtree=1.0, dp_alpha_parent_edge=1.0, alpha_decay_parent_edge=0.9,
                        eta=0.):


        self.tree_dict = dict()
        self.n_nodes = n_nodes
        self.dp_alpha_subtree = dp_alpha_subtree
        self.alpha_decay_subtree = alpha_decay_subtree
        self.dp_gamma_subtree = dp_gamma_subtree
        self.dp_alpha_parent_edge = dp_alpha_parent_edge
        self.alpha_decay_parent_edge = alpha_decay_parent_edge
        self.eta = eta
        self.adata = None
        self.cmap = None

    def get_size(self):
        return len(self.tree_dict.keys())

    def get_params(self):
        params = []
        for node in self.tree_dict:
            params.append(self.tree_dict[node]['params'])
        return np.array(params, dtype=np.float)

    def add_tree_parameters(self, change_name=True):

        nodes = list(self.tree_dict.keys())
        if change_name:
            alphabet = list(string.ascii_uppercase)
            new_names = dict(zip(nodes, alphabet[:len(nodes)]))
            new_names['-1'] = '-1'
        sizes = None
        try:
            sizes = [self.tree_dict[node]['size'] for node in nodes]
        except KeyError:
            pass
        for i, node in enumerate(nodes):
            self.tree_dict[node]['children'] = []
            self.tree_dict[node]['dp_alpha_subtree'] = self.dp_alpha_subtree
            self.tree_dict[node]['alpha_decay_subtree'] = self.alpha_decay_subtree
            self.tree_dict[node]['dp_gamma_subtree'] = self.dp_gamma_subtree
            self.tree_dict[node]['dp_alpha_parent_edge'] = self.dp_alpha_parent_edge
            self.tree_dict[node]['alpha_decay_parent_edge'] = self.alpha_decay_parent_edge
            self.tree_dict[node]['eta'] = self.eta
            self.tree_dict[node]['weight'] = 1.0/len(list(self.tree_dict.keys()))
            self.tree_dict[node]['size'] = 1.0
            if sizes:
                self.tree_dict[node]['weight'] = sizes[i]/sum(sizes)
                self.tree_dict[node]['size'] = sizes[i]
            if change_name:
                self.tree_dict[node]['parent'] = new_names[self.tree_dict[node]['parent']]
                self.tree_dict[alphabet[i]] = self.tree_dict[node]
                del self.tree_dict[node]

        for i in self.tree_dict:
            for j in self.tree_dict:
                if self.tree_dict[j]['parent'] == i:
                    self.tree_dict[i]['children'].append(j)

    def generate_tree(self):
        alphabet = list(string.ascii_uppercase)
        self.tree_dict = dict(A=dict(parent='-1', children=[],
                            params=None, dp_alpha_subtree=self.dp_alpha_subtree,
                            alpha_decay_subtree=self.alpha_decay_subtree,
                            dp_gamma_subtree=self.dp_gamma_subtree,
                            dp_alpha_parent_edge=self.dp_alpha_parent_edge,
                            alpha_decay_parent_edge=self.alpha_decay_parent_edge,
                            eta=self.eta,
                            weight=1.0/self.n_nodes,
                            size=1,
                            color=constants.CLONES_PAL[0],
                            label='A'))
        for c in range(1, self.n_nodes):
            parent = alphabet[np.random.choice(np.arange(0, c))]
            self.tree_dict[alphabet[c]]= dict(parent=parent, children=[],
                            params=None, dp_alpha_subtree=self.dp_alpha_subtree,
                            alpha_decay_subtree=self.alpha_decay_subtree,
                            dp_gamma_subtree=self.dp_gamma_subtree,
                            dp_alpha_parent_edge=self.dp_alpha_parent_edge,
                            alpha_decay_parent_edge=self.alpha_decay_parent_edge,
                            eta=self.eta,
                            weight=1.0/self.n_nodes,
                            size=1,
                            color=constants.CLONES_PAL[c],
                            label=alphabet[c])

        for i in self.tree_dict:
            for j in self.tree_dict:
                if self.tree_dict[j]['parent'] == i:
                    self.tree_dict[i]['children'].append(j)

    def get_sum_weights_subtree(self, label):
        if "weight" not in self.tree_dict['A'].keys():
            raise KeyError("No weights were specified in the input tree.")

        sum = self.tree_dict[label]['weight']
        def descend(label, total):
            for child in self.tree_dict[label]['children']:
                total = total + self.tree_dict[child]['weight']
                total = descend(child, total)
            return total

        sum = descend(label, sum)

        return sum

    def plot_tree(self, fillcolor=None, labels=False):
        u = Digraph()
        start = 0
        end = self.n_nodes
        for node in self.tree_dict:
            parent = self.tree_dict[node]['parent']
            if parent == '-1':
                continue

            style = None
            # Add colors
            try:
                parent_fillcolor = self.tree_dict[parent]['color']
                node_fillcolor = self.tree_dict[node]['color']
            except:
                parent_fillcolor = fillcolor
                node_fillcolor = fillcolor
            if parent_fillcolor is not None or node_fillcolor is not None:
                style = 'filled'

            # Add labels
            parent_label, node_label = parent, node
            if labels:
                try:
                    parent_label = parent_label + '\n\n' + self.tree_dict[parent]['params_label']
                    node_label = node_label + '\n\n' + self.tree_dict[node]['params_label']
                except:
                    pass

            u.node(parent, parent_label, fillcolor=parent_fillcolor, style=style)
            u.node(node, node_label, fillcolor=node_fillcolor, style=style)
            u.edge(parent, node, arrowhead='none')

        return u

    def get_node_ancestors(self, label):
        ancestors = [label]
        parent = self.tree_dict[label]['parent']
        while parent != "-1":
            ancestors.append(parent)
            parent = self.tree_dict[parent]['parent']

        ancestors = ancestors[::-1]

        return ancestors

    def create_adata(self, var_names=None):
        params = []
        for node in self.tree_dict:
            params.append(self.tree_dict[node]['params'])
        params = pd.DataFrame(params)
        params.index = pd.Series(list(self.tree_dict.keys()), dtype="category")
        if var_names is not None:
            params.columns = var_names
        self.adata = AnnData(params)
        self.adata.obs['node'] = params.index
        self.adata.uns['node_colors'] = [self.tree_dict[node]['color'] for node in self.tree_dict]
        self.adata.uns['node_sizes'] = np.array([self.tree_dict[node]['size'] for node in self.tree_dict])
        self.adata.var['bulk'] = np.mean(self.adata.X * self.adata.uns['node_sizes'].reshape(-1,1), axis=0)

    def plot_heatmap(self, var_names=None, cmap=None, **kwds):
        if var_names is None:
            var_names = self.adata.var_names
        if cmap is None:
            cmap = self.cmap
        kwds['vmax'] = 4 if kwds['vmax'] is None else kwds['vmax']
        kwds['vmin'] = 0 if kwds['vmin'] is None else kwds['vmin']

        ax = sc.pl.heatmap(self.adata, var_names, groupby='node', cmap=cmap, show=False, **kwds)
        yticks = ax['groupby_ax'].get_yticks()
        ax['groupby_ax'].set_yticks(yticks - 0.5)
        ax['groupby_ax'].set_yticklabels(['A', 'B', 'C'])
        ax['groupby_ax'].get_yticks()
        plt.show()

    def read_tree_from_dict(self, tree_dict, input_params_key='params', input_label_key='label', input_parent_key='parent', input_sizes_key='size'):
        self.tree_dict = dict()
        alphabet = list(string.ascii_uppercase)

        sizes = None
        try:
            sizes = [tree_dict[node][input_sizes_key] for node in tree_dict]
        except KeyError:
            pass

        for node in tree_dict:
            self.tree_dict[alphabet[int(tree_dict[node][input_label_key])]] = dict(
                parent=tree_dict[node][input_parent_key],
                children=[],
                params=np.array(tree_dict[node][input_params_key]),
                dp_alpha_subtree=self.dp_alpha_subtree,
                alpha_decay_subtree=self.alpha_decay_subtree,
                dp_gamma_subtree=self.dp_gamma_subtree,
                dp_alpha_parent_edge=self.dp_alpha_parent_edge,
                alpha_decay_parent_edge=self.alpha_decay_parent_edge,
                eta=self.eta,
                weight=1.0/len(list(tree_dict.keys())),
                size=1,
                color=constants.CLONES_PAL[int(tree_dict[node][input_label_key])],
                label=tree_dict[node][input_label_key],
            )
            if sizes:
                self.tree_dict[alphabet[int(tree_dict[node][input_label_key])]]['size'] = int(tree_dict[node][input_sizes_key])
                self.tree_dict[alphabet[int(tree_dict[node][input_label_key])]]['weight'] = tree_dict[node][input_sizes_key]/sum(sizes)

        for i in self.tree_dict:
            for j in self.tree_dict:
                if self.tree_dict[j]['parent'] == i:
                    self.tree_dict[i]['children'].append(j)

    @abstractmethod
    def add_node_params(self):
        return
