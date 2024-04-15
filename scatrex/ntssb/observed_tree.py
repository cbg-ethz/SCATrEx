import string
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import matplotlib.pyplot as plt
from graphviz import Digraph

from .node import AbstractNode
from ..plotting import constants
from ..utils.tree_utils import tree_to_dict, dict_to_tree, subsample_tree, condense_tree


class ObservedTree(ABC):
    def __init__(
        self,
        n_nodes=3,
        dp_alpha_subtree=1.0,
        alpha_decay_subtree=1.0,
        dp_gamma_subtree=1.0,
        dp_alpha_parent_edge=1.0,
        alpha_decay_parent_edge=1.0,
        eta=1.0,
        node_weights=None,
        seed=42,
        add_root=True,
        **kwargs,
    ):
        self.node_constructor = AbstractNode
        self.seed = seed
        self.tree_dict = dict() # flat
        self.tree = dict() # recursive
        self.n_nodes = n_nodes
        self.dp_alpha_subtree = dp_alpha_subtree
        self.alpha_decay_subtree = alpha_decay_subtree
        self.dp_gamma_subtree = dp_gamma_subtree
        self.dp_alpha_parent_edge = dp_alpha_parent_edge
        self.alpha_decay_parent_edge = alpha_decay_parent_edge
        self.eta = eta
        self.adata = None
        self.cmap = None
        self.node_weights = node_weights
        self.add_root = add_root
        if self.node_weights is None:
            rng = np.random.default_rng(seed=self.seed)
            self.node_weights = rng.dirichlet([100.0] * self.n_nodes)

    def condense(self, min_weight=.1, inplace=False):
        """
        Traverse the tree from the bottom up and merge nodes until all nodes have at least min_weight
        """
        if inplace:
            condense_tree(self.tree, min_weight=min_weight)
            self.tree_dict = tree_to_dict(self.tree)
            self.tree = dict_to_tree(self.tree_dict, root_name=self.tree['label'])
            self.n_nodes = len(self.tree_dict.keys())
        else:
            new_tree = deepcopy(self.tree)
            condense_tree(new_tree, min_weight=min_weight)
            new_tree_dict = tree_to_dict(new_tree)
            new_tree = dict_to_tree(new_tree_dict, root_name=new_tree['label'])
            n_nodes = len(new_tree_dict.keys())
            new_obj = self.__class__(n_nodes=n_nodes, seed=self.seed)
            new_obj.tree = new_tree
            new_obj.tree_dict = new_tree_dict
            return new_obj

    def subsample(self, keep_prob=0.5, force=True, inplace=False):
        """
        Randomly choose a fraction of nodes to keep in the tree
        """
        init_n_nodes = self.n_nodes
        seed = self.seed
        while True:
            new_tree = deepcopy(self.tree)
            subsample_tree(new_tree, keep_prob=keep_prob, seed=seed)
            new_tree_dict = tree_to_dict(new_tree)
            new_tree = dict_to_tree(new_tree_dict, root_name=new_tree['label'])
            n_nodes = len(new_tree_dict.keys())
            if force:
                if n_nodes == int(init_n_nodes*keep_prob):
                    break
                else:
                    seed += 1
            else:
                break

        if inplace:
            self.tree = new_tree
            self.tree_dict = new_tree_dict
            self.n_nodes = n_nodes
        else:
            new_obj = self.__class__(n_nodes=n_nodes, seed=self.seed)
            new_obj.tree = new_tree
            new_obj.tree_dict = new_tree_dict
            return new_obj

    def get_size(self):
        return len(self.tree_dict.keys())

    def get_param_size(self):
        return self.tree["param"].size

    def get_params(self):
        params = []
        for node in self.tree_dict:
            params.append(self.tree_dict[node]["param"])
        return np.array(params, dtype=np.float)

    def change_names(self, keep="root"):
        nodes = list(self.tree_dict.keys())
        alphabet = list(string.ascii_uppercase)

        tochange = [node for node in nodes if node != keep]
        new_names = alphabet[: len(tochange)]
        new_names = dict(zip(tochange, new_names))
        new_names["-1"] = "-1"
        new_names[keep] = keep

        for i, node in enumerate(tochange):
            self.tree_dict[node]["parent"] = new_names[self.tree_dict[node]["parent"]]
            self.tree_dict[node]["children"] = []
            self.tree_dict[alphabet[i]] = self.tree_dict[node]
            self.tree_dict[alphabet[i]]["label"] = alphabet[i]
            if new_names[node] != node:
                del self.tree_dict[node]

        for i in self.tree_dict:
            self.tree_dict[i]["children"] = []

        for i in self.tree_dict:
            for j in self.tree_dict:
                if self.tree_dict[j]["parent"] == i:
                    self.tree_dict[i]["children"].append(j)

        root_name = keep
        for node in self.tree_dict:
            if self.tree_dict[node]['parent'] == '-1':
                root_name = node

        self.tree = dict_to_tree(self.tree_dict, root_name=root_name)
        self.tree_dict = tree_to_dict(self.tree)

    def set_colors(self, root_node='root'):
        idx = 0
        if root_node in self.tree_dict:
            self.tree_dict[root_node]["color"] = "lightgray"
            idx += 1
        for i, node in enumerate(list(self.tree_dict.keys())[idx:]):
            try:
                self.tree_dict[node]["color"] = constants.LABEL_COLORS_DICT[node]
            except:
                self.tree_dict[node]["color"] = constants.CLONES_PAL[i]
        
        root_name = root_node
        for node in self.tree_dict:
            if self.tree_dict[node]['parent'] == '-1':
                root_name = node
        
        self.tree = dict_to_tree(self.tree_dict, root_name=root_name)
        self.tree_dict = tree_to_dict(self.tree)

    def add_tree_parameters(self, change_name=True):

        nodes = list(self.tree_dict.keys())
        if change_name:
            alphabet = list(string.ascii_uppercase)
            new_names = dict(zip(nodes, alphabet[: len(nodes)]))
            new_names["-1"] = "-1"
        sizes = None
        try:
            sizes = [self.tree_dict[node]["size"] for node in nodes]
        except KeyError:
            pass
        for i, node in enumerate(nodes):
            self.tree_dict[node]["children"] = []
            self.tree_dict[node]["dp_alpha_subtree"] = self.dp_alpha_subtree
            self.tree_dict[node]["alpha_decay_subtree"] = self.alpha_decay_subtree
            self.tree_dict[node]["dp_gamma_subtree"] = self.dp_gamma_subtree
            self.tree_dict[node]["dp_alpha_parent_edge"] = self.dp_alpha_parent_edge
            self.tree_dict[node][
                "alpha_decay_parent_edge"
            ] = self.alpha_decay_parent_edge
            self.tree_dict[node]["eta"] = self.eta
            self.tree_dict[node]["weight"] = 1.0 / len(list(self.tree_dict.keys()))
            self.tree_dict[node]["size"] = 1.0
            if sizes:
                self.tree_dict[node]["weight"] = sizes[i] / sum(sizes)
                self.tree_dict[node]["size"] = sizes[i]
            if change_name:
                self.tree_dict[node]["parent"] = new_names[
                    self.tree_dict[node]["parent"]
                ]
                self.tree_dict[alphabet[i]] = self.tree_dict[node]
                del self.tree_dict[node]

        for i in self.tree_dict:
            for j in self.tree_dict:
                if self.tree_dict[j]["parent"] == i:
                    self.tree_dict[i]["children"].append(j)

    def generate_tree(self):
        alphabet = list(string.ascii_uppercase)
        # Add healthy node
        if self.add_root:
            self.tree_dict = dict(
                root=dict(
                    parent="-1",
                    children=[],
                    param=None,
                    dp_alpha_subtree=self.dp_alpha_subtree,
                    alpha_decay_subtree=self.alpha_decay_subtree,
                    dp_gamma_subtree=self.dp_gamma_subtree,
                    dp_alpha_parent_edge=self.dp_alpha_parent_edge,
                    alpha_decay_parent_edge=self.alpha_decay_parent_edge,
                    eta=self.eta,
                    weight=0,
                    size=int(0),
                    color="lightgray",
                    label="root",
                )
            )
            mrca_parent = "root"
        else:
            mrca_parent = "-1"
        # Add MRCA node
        self.tree_dict["A"] = dict(
            parent=mrca_parent,
            children=[],
            param=None,
            dp_alpha_subtree=self.dp_alpha_subtree,
            alpha_decay_subtree=self.alpha_decay_subtree,
            dp_gamma_subtree=self.dp_gamma_subtree,
            dp_alpha_parent_edge=self.dp_alpha_parent_edge,
            alpha_decay_parent_edge=self.alpha_decay_parent_edge,
            eta=self.eta,
            weight=self.node_weights[0],
            size=int(self.node_weights[0] * 100),
            color=constants.CLONES_PAL[0],
            label="A",
        )
        for c in range(1, self.n_nodes):
            rng = np.random.default_rng(seed=self.seed+c)
            parent = alphabet[rng.choice(np.arange(0, c))]
            self.tree_dict[alphabet[c]] = dict(
                parent=parent,
                children=[],
                param=None,
                dp_alpha_subtree=self.dp_alpha_subtree,
                alpha_decay_subtree=self.alpha_decay_subtree,
                dp_gamma_subtree=self.dp_gamma_subtree,
                dp_alpha_parent_edge=self.dp_alpha_parent_edge,
                alpha_decay_parent_edge=self.alpha_decay_parent_edge,
                eta=self.eta,
                weight=self.node_weights[c],
                size=int(self.node_weights[c] * 100),
                color=constants.CLONES_PAL[c],
                label=alphabet[c],
            )

        for i in self.tree_dict:
            for j in self.tree_dict:
                if self.tree_dict[j]["parent"] == i:
                    self.tree_dict[i]["children"].append(j)

        for node in self.tree_dict:
            if self.tree_dict[node]['parent'] == '-1':
                root_name = node

        self.tree = dict_to_tree(self.tree_dict, root_name=root_name)

    def get_sum_weights_subtree(self, label):
        if "weight" not in self.tree_dict["A"].keys():
            raise KeyError("No weights were specified in the input tree.")

        sum = self.tree_dict[label]["weight"]

        def descend(label, total):
            for child in self.tree_dict[label]["children"]:
                total = total + self.tree_dict[child]["weight"]
                total = descend(child, total)
            return total

        sum = descend(label, sum)

        return sum

    def plot_tree(
        self,
        fillcolor=None,
        labels=False,
        counts=True,
        label_fontsize=24,
        size_fontsize=12,
    ):
        u = Digraph()
        start = 0
        end = self.n_nodes
        for node in self.tree_dict:
            parent = self.tree_dict[node]["parent"]
            if parent == "-1":
                continue

            style = None
            # Add colors
            try:
                parent_fillcolor = self.tree_dict[parent]["color"]
                node_fillcolor = self.tree_dict[node]["color"]
            except:
                parent_fillcolor = fillcolor
                node_fillcolor = fillcolor
            if parent_fillcolor is not None or node_fillcolor is not None:
                style = "filled"

            # Add labels
            parent_label, node_label = parent, node
            if labels:
                try:
                    parent_label = (
                        parent_label + "\n\n" + self.tree_dict[parent]["params_label"]
                    )
                    node_label = (
                        node_label + "\n\n" + self.tree_dict[node]["params_label"]
                    )
                except:
                    pass

            parent_label, node_label = parent, node
            parent_label = f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{parent_label}</B></FONT>'
            node_label = f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{node_label}</B></FONT>'

            if counts:
                try:
                    parent_label = (
                        parent_label
                        + "<br/><br/>"
                        + f'<FONT FACE="Arial">{str(self.tree_dict[parent]["size"])} cells</FONT>'
                    )
                    node_label = (
                        node_label
                        + "<br/><br/>"
                        + f'<FONT FACE="Arial">{str(self.tree_dict[node]["size"])} cells</FONT>'
                    )
                except:
                    pass

            u.node(
                parent,
                "<" + parent_label + ">",
                fillcolor=parent_fillcolor,
                style=style,
            )
            u.node(node, "<" + node_label + ">", fillcolor=node_fillcolor, style=style)
            u.edge(parent, node, arrowhead="none")

        return u

    def get_node_ancestors(self, label):
        ancestors = [label]
        parent = self.tree_dict[label]["parent"]
        while parent != "-1":
            ancestors.append(parent)
            parent = self.tree_dict[parent]["parent"]

        ancestors = ancestors[::-1]

        return ancestors

    def create_adata(self, var_names=None):
        params = []
        params_labels = []
        for node in self.tree_dict:
            if self.tree_dict[node]["size"] != 0:
                params_labels.append(
                    [self.tree_dict[node]["label"]] * self.tree_dict[node]["size"]
                )
                params.append(
                    np.vstack(
                        [self.tree_dict[node]["param"]] * self.tree_dict[node]["size"]
                    )
                )
        params = pd.DataFrame(np.vstack(params))
        params_labels = np.concatenate(params_labels).tolist()
        if var_names is not None:
            params.columns = var_names
        self.adata = AnnData(params)
        self.adata.obs["node"] = params_labels
        self.adata.uns["node_colors"] = [
            self.tree_dict[node]["color"]
            for node in self.tree_dict
            if self.tree_dict[node]["size"] != 0
        ]
        self.adata.uns["node_sizes"] = np.array(
            [
                self.tree_dict[node]["size"]
                for node in self.tree_dict
                if self.tree_dict[node]["size"] != 0
            ]
        )
        self.adata.var["bulk"] = np.mean(self.adata.X, axis=0)

    def plot_heatmap(self, var_names=None, cmap=None, **kwds):
        if var_names is None:
            var_names = self.adata.var_names
        if cmap is None:
            cmap = self.cmap
        kwds["vmax"] = 4 if "vmax" not in kwds else kwds["vmax"]
        kwds["vmin"] = 0 if "vmin" not in kwds else kwds["vmin"]

        if kwds["vmax"] > 4:
            cmap = None

        ax = sc.pl.heatmap(
            self.adata, var_names, groupby="node", cmap=cmap, show=False, **kwds
        )
        yticks = ax["groupby_ax"].get_yticks()
        ax["groupby_ax"].set_yticks(yticks - 0.5)
        node_labels = self.adata.obs["node"].values.tolist()
        ax["groupby_ax"].set_yticklabels(np.unique(node_labels))
        ax["groupby_ax"].get_yticks()
        plt.show()

    def read_tree_from_dict(
        self,
        tree_dict,
        input_params_key="param",
        input_label_key="label",
        input_parent_key="parent",
        input_sizes_key="size",
        root_parent="NULL",
        use_labels=False,
    ):
        self.tree_dict = dict()
        self.n_nodes = len(self.tree_dict.keys())
        fixed_color = None
        if self.n_nodes > len(constants.CLONES_PAL):
            fixed_color = "lightgray"

        sizes = None
        try:
            sizes = [tree_dict[node][input_sizes_key] for node in tree_dict]
        except KeyError:
            pass
        colors = None
        try:
            colors = [tree_dict[node]["color"] for node in tree_dict]
        except KeyError:
            pass

        for idx, node in enumerate(tree_dict):
            parent_id = tree_dict[node][input_parent_key]
            if parent_id == root_parent:
                parent_id = "-1"
            elif use_labels:
                parent_id = tree_dict[parent_id][input_label_key]
            if fixed_color is not None:
                color = fixed_color
            else:
                if use_labels:
                    color = (
                        "lightgray"
                        if parent_id == "-1"
                        else constants.LABEL_COLORS_DICT[
                            tree_dict[node][input_label_key]
                        ]
                    )
                else:
                    color = (
                        constants.CLONES_PAL[idx]
                        if "color" not in tree_dict[node]
                        else tree_dict[node]["color"]
                    )
            label = (
                node
                if input_label_key not in tree_dict[node]
                else tree_dict[node][input_label_key]
            )
            size = 0.0
            weight = 0.0
            new_key = node
            if use_labels:
                new_key = tree_dict[node][input_label_key]
            self.tree_dict[new_key] = dict(
                parent=parent_id,
                children=[],
                params=np.array(tree_dict[node][input_params_key]).ravel(),
                dp_alpha_subtree=self.dp_alpha_subtree,
                alpha_decay_subtree=self.alpha_decay_subtree,
                dp_gamma_subtree=self.dp_gamma_subtree,
                dp_alpha_parent_edge=self.dp_alpha_parent_edge,
                alpha_decay_parent_edge=self.alpha_decay_parent_edge,
                eta=self.eta,
                weight=weight,
                size=size,
                color=color,
                label=label,
            )
            if sizes:
                self.tree_dict[new_key]["size"] = int(tree_dict[node][input_sizes_key])
                self.tree_dict[new_key]["weight"] = (
                    tree_dict[node][input_sizes_key] / sum(sizes) + 1e-10
                )

        for i in self.tree_dict:
            for j in self.tree_dict:
                if self.tree_dict[j]["parent"] == i:
                    self.tree_dict[i]["children"].append(j)
        
        for node in self.tree_dict:
            if self.tree_dict[node]['parent'] == '-1':
                root_name = node

        self.tree = dict_to_tree(self.tree_dict, root_name=root_name)

    def root(self):
        nodes = list(self.tree_dict.keys())
        for node in nodes:
            if self.tree_dict[node]["parent"] not in nodes:
                return node

    def mrca(self):
        if "root" in self.tree_dict:
            if len(self.tree_dict["root"]["children"]) > 1:
                return self.root()
            else:
                for node in self.tree_dict:
                    if self.tree_dict[node]["parent"] == "root":
                        return node
        else:
            return self.root()

    def update_weights(self, uniform=False):
        total = np.sum([self.tree_dict[node]["size"] for node in self.tree_dict])
        for node in self.tree_dict:
            self.tree_dict[node]["size"] = int(self.tree_dict[node]["size"])
            self.tree_dict[node]["weight"] = (
                self.tree_dict[node]["size"] / total + 1e-10
            )
            if uniform:
                self.tree_dict[node]["weight"] = 1.0 / len(list(self.tree_dict.keys()))

    def subset_genes(self, gene_list):
        for node in self.tree_dict:
            self.tree_dict[node]["param"] = pd.DataFrame(
                self.tree_dict[node]["param"][:, np.newaxis].T,
                columns=self.adata.var_names,
            )[gene_list].values.ravel()
        self.adata = self.adata[:, gene_list]

    @abstractmethod
    def sample_root(self):
        return
    
    @abstractmethod
    def sample_kernel(self):
        return
    
    def update_tree(self):
        for node in self.tree_dict:
            if self.tree_dict[node]['parent'] == '-1':
                root_name = node
        self.tree = dict_to_tree(self.tree_dict, root_name=root_name)

    def update_dict(self):
        self.tree_dict = tree_to_dict(self.tree)        
    
    def param_distance(self, paramA, paramB):
        return np.sqrt(np.sum((paramA-paramB)**2))

    def add_node_params(
        self, n_genes=2, min_dist=0.2, **params
    ):
        def descend(root, idx=0, depth=1):
            for i, child in enumerate(root['children']):
                seed = self.seed+idx
                accepted = False
                while not accepted:
                    child["param"] = self.sample_kernel(root["param"], seed=seed, depth=depth, **params)
                    dist_to_parent = self.param_distance(root["param"], child["param"])
                    # Reject sample if too close to any other child
                    dists = []
                    for j, child2 in enumerate(root['children']):
                        if j < i:
                            dists.append(self.param_distance(child["param"], child2["param"]))
                    if np.all(np.array(dists) >= min_dist*dist_to_parent):
                        accepted = True
                    else:
                        seed += 1

                idx = descend(child, idx+1, depth=depth+1)
            return idx

        # Set root param
        self.tree["param"] = self.sample_root(n_genes=n_genes, seed=self.seed, **params)

        # Set node params recursively 
        descend(self.tree)

        # Update tree_dict too
        self.tree_dict = tree_to_dict(self.tree)

        self.create_adata()
