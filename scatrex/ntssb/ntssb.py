"""
This module contains the NestedTSSB class.
"""

from functools import partial
from copy import deepcopy

from graphviz import Digraph
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

import numpy as np
from numpy import *

import jax
from jax import jit, grad, vmap
from jax import random
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.nn as jnn

from ..utils.math_utils import *
from ..callbacks import elbos_callback
from .tssb import TSSB
from ..plotting import tree_colors, plot_full_tree

import logging

logger = logging.getLogger(__name__)


class NTSSB(object):
    """
    Takes as input a dictionary of {node_label: parent_label} pairs.
    Contains various TSSB objects:
        - a truncated TSSB with fixed structure and no room for more nodes
        - a TSSB for each node in the truncated TSSB
        - a truncated TSSB for each edge in the main tree that has the same
          structure as the corresponding subtree but different nu-stick lengths,
          which control the probability of each node being the pivot that
          connects to the next subtree.
    Each node in an observed node's TSSB maintains a variable indicating its
    child node in the observed tree.

    We keep two levels of assignments: the observed node level and its subtree.
    We first assign to the observed node based on the fit to its root node,
    and then explore nodes inside its subtree. The nodes of the first level
    are the same objects as the root nodes of the second level -- when the latter
    are updated (e.g. by a change in pivot), so are the former.
    """

    min_dp_alpha = 0.001
    max_dp_alpha = 10.0
    min_dp_gamma = 0.001
    max_dp_gamma = 10.0
    min_alpha_decay = 0.001
    max_alpha_decay = 0.80

    def __init__(
        self,
        input_tree,
        dp_alpha=1.0,
        dp_gamma=1.0,
        alpha_decay=1.0,
        min_depth=0,
        max_depth=15,
        fixed_weights_pivot_sampling=True,
        use_weights=True,
        weights_concentration=10.,
        min_weight=1e-6,
        verbosity=logging.INFO,
        node_hyperparams=dict(),
        seed=42,
    ):
        if input_tree is None:
            raise Exception("Input tree must be specified.")

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.dp_alpha = dp_alpha  # smaller dp_alpha => larger nu => less nodes
        self.dp_gamma = dp_gamma  # smaller dp_gamma => larger psi => less nodes
        self.alpha_decay = alpha_decay # smaller alpha_decay => larger decay with depth => less nodes

        self.seed = seed

        self.input_tree = input_tree
        self.input_tree_dict = self.input_tree.tree_dict
        self.node_constructor = self.input_tree.node_constructor
        self.node_hyperparams = node_hyperparams

        self.fixed_weights_pivot_sampling = fixed_weights_pivot_sampling

        self.assignments = []

        self.elbo = -np.inf
        self.ll = -np.inf
        self.kl = -np.inf
        self.node_kl = -np.inf
        self.global_kl = -np.inf
        self.cell_kl = -np.inf
        self.data = None
        self.num_data = None
        self.covariates = None
        self.num_batches = 1
        self.batch_size = None

        self.max_nodes = (
            len(self.input_tree_dict.keys()) * 1
        )  # upper bound on number of nodes
        self.n_nodes = len(self.input_tree_dict.keys())
        self.n_total_nodes = self.n_nodes

        self.obs_cmap = self.input_tree.cmap
        self.exp_cmap = matplotlib.cm.viridis
        self.gene_node_colormaps = dict()

        logger.setLevel(verbosity)

        self.reset_tree(use_weights=use_weights, weights_concentration=weights_concentration, min_weight=min_weight)

        self.set_pivot_priors()

        self.variational_parameters = {
            'LSE_c': [], # normalizing constant for cell-TSSB assignments
                                       }

    # ========= Functions to initialize tree. =========
    def reset_tree(self, use_weights=False, weights_concentration=10., min_weight=1e-6):
        if use_weights and "weight" not in self.input_tree_dict["A"].keys():
            raise KeyError("No weights were specified in the input tree.")

        # Clear tree
        self.assignments = []

        # Traverse tree in depth first and create TSSBs
        def descend(input_root, idx=1, depth=0):
            alpha_nu = 1.
            beta_nu = (self.alpha_decay**depth) * self.dp_alpha

            children_roots = []
            sticks = []
            psi_priors = []
            for i, child in enumerate(input_root['children']):
                child_root, idx = descend(child, idx, depth+1)
                children_roots.append(child_root)

                rng = np.random.default_rng(int(self.seed+idx*1e6))
                stick = boundbeta(1, self.dp_gamma, rng)
                if i >= len(input_root['children']) - 1:
                    stick = 1.0
                psi_prior = {"alpha_psi": 1., "beta_psi": self.dp_gamma}
                if use_weights:
                    stick = self.input_tree.get_sum_weights_subtree(child_root["label"])
                    if i < len(input_root["children"]) - 1:
                        sum = 0
                        for j, c in enumerate(input_root["children"][i:]):
                            sum = sum + self.input_tree.get_sum_weights_subtree(c["label"])
                        stick = stick / sum
                    else:
                        stick = 1.0
                    psi_prior["alpha_psi"] = stick * (weights_concentration - 2) + 1 
                    psi_prior["beta_psi"] = (1-stick) * (weights_concentration - 2) + 1
                psi_priors.append(psi_prior)
                sticks.append(stick)
            if len(sticks) == 0:
                sticks = empty((0, 1))
            else:
                sticks = vstack(sticks)

            label = input_root["label"]

            # Create node
            local_seed = int(self.seed+idx*1e6)
            node = self.node_constructor(
                input_root["param"],
                label=label,
                seed=local_seed,
                **self.node_hyperparams,
            )                

            # Create TSSB with pointers to children root nodes
            rng = np.random.default_rng(local_seed)

            children_nodes = [c["node"].root["node"] for c in children_roots]
            tssb = TSSB(
                    node,
                    label,
                    ntssb=self,
                    children_root_nodes=children_nodes,
                    dp_alpha=input_root["dp_alpha_subtree"],
                    alpha_decay=input_root["alpha_decay_subtree"],
                    dp_gamma=input_root["dp_gamma_subtree"],
                    eta=input_root["eta"],
                    color=input_root["color"],
                    seed=local_seed,
                )
            input_root["subtree"] = tssb

            # Create root dict
            if depth >= self.min_depth:
                main = boundbeta(1.0, (self.alpha_decay ** depth) * self.dp_alpha, rng) 
            else: # if depth < min_depth, no data can be added to this node (main stick is nu)
                main = 0. 
            if use_weights:
                main = input_root["weight"]
                subtree_weights_sum = self.input_tree.get_sum_weights_subtree(label)
                main = main / subtree_weights_sum
                input_root["subtree"].weight = input_root["weight"]
            if len(input_root["children"]) < 1:
                main = 1.0  # stop at leaf node
            
            if use_weights:
                alpha_nu = main * (weights_concentration - 2) + 1 
                beta_nu = (1-main) * (weights_concentration - 2) + 1 

            root_dict =  {
                    "node": tssb,
                    "main": main,
                    "sticks": sticks, 
                    "children": children_roots,
                    "label": input_root["label"],
                    "super_parent": None, # maybe remove
                    "pivot_node": None, # maybe remove
                    "pivot_tssb": None, # maybe remove
                    "color": input_root["color"],
                    "alpha_nu":  alpha_nu,
                    "beta_nu": beta_nu,
                    "psi_priors": psi_priors,
                }

            return root_dict, idx+1

        self.root, _ = descend(self.input_tree.tree)

        def descend(root):
            for child in root['children']:
                child["node"]._parent = root["node"]
                descend(child)
        
        # Set parents
        descend(self.root)

        # And add weights keys
        self.set_weights()

    def set_tssb_params(self, dp_alpha=1., alpha_decay=1., dp_gamma=1.):
        def descend(root):
            root['node'].dp_alpha = dp_alpha
            root['node'].alpha_decay = alpha_decay
            root['node'].dp_gamma = dp_gamma
            for child in root['children']:
                descend(child)
        descend(self.root)

    def set_node_hyperparams(self, **kwargs):
        def descend(root):
            root['node'].set_node_hyperparams(**kwargs)
            for child in root['children']:
                descend(child)
        descend(self.root)

    def reset_variational_kernels(self, **kwargs):
        def descend(root):
            root['node'].reset_variational_kernels(**kwargs)
            for child in root['children']:
                descend(child)
        descend(self.root)

    def sample_variational_distributions(self, **kwargs):
        def descend(root):
            root['node'].sample_variational_distributions(**kwargs)
            for child in root['children']:
                descend(child)
        descend(self.root)

    def set_learned_parameters(self):
        def descend(root):
            root['node'].set_learned_parameters()
            for child in root['children']:
                descend(child)
        descend(self.root)        

    def reset_sufficient_statistics(self):
        def descend(super_tree):
            super_tree["node"].reset_sufficient_statistics(num_batches=self.num_batches)
            for child in super_tree["children"]:
                descend(child)
        descend(self.root)

    def reset_variational_parameters(self, **kwargs):
        # Reset node parameters
        def descend(super_tree, alpha_psi=1., beta_psi=1.):
            alpha_nu = super_tree['alpha_nu']
            beta_nu = super_tree['beta_nu']
            super_tree["node"].reset_variational_parameters(alpha_nu=alpha_nu, beta_nu=beta_nu,
                                                            alpha_psi=alpha_psi,beta_psi=beta_psi, 
                                                            **kwargs)
            c_norm = jnp.array(super_tree["node"].variational_parameters['q_c'])
            for i, child in enumerate(super_tree["children"]):         
                alpha_psi = super_tree['psi_priors'][i]["alpha_psi"]
                beta_psi = super_tree['psi_priors'][i]["beta_psi"]
                c_norm += descend(child, alpha_psi=alpha_psi, beta_psi=beta_psi)
            return c_norm
        
        c_norm = descend(self.root)

        # Apply normalization
        def descend(root, alpha_psi=1., beta_psi=1.):
            alpha_nu = root['alpha_nu']
            beta_nu = root['beta_nu']
            root["node"].reset_variational_parameters(alpha_nu=alpha_nu, beta_nu=beta_nu,
                                                        alpha_psi=alpha_psi,beta_psi=beta_psi,
                                                        **kwargs)
            root["node"].variational_parameters['q_c'] = root["node"].variational_parameters['q_c'] / c_norm
            for i, child in enumerate(root["children"]):
                alpha_psi = root['psi_priors'][i]["alpha_psi"]
                beta_psi = root['psi_priors'][i]["beta_psi"]
                descend(child, alpha_psi=alpha_psi, beta_psi=beta_psi)
        descend(self.root)

    def init_root_kernels(self, **kwargs):
        def descend(super_tree):
            for child in super_tree["children"]:
                child["node"].root["node"].init_kernel(**kwargs)
                descend(child)
        descend(self.root)
        
    def reset_node_parameters(
        self, **node_hyperparams
    ):  
        # Reset node parameters
        def descend(super_tree):
            super_tree["node"].reset_node_parameters(**node_hyperparams)
            for child in super_tree["children"]:
                descend(child)

        descend(self.root)

    def remake_observed_params(self):
        def descend(super_tree):
            self.input_tree.tree_dict[super_tree["label"]]["param"] = super_tree["node"].root["node"].params
            super_tree["node"].root["node"].observed_parameters = self.input_tree.tree_dict[super_tree["label"]]["param"]
            for child in super_tree["children"]:
                descend(child)

        descend(self.root)
        self.input_tree.update_tree()

    def set_radial_positions(self):
        """
        Create a radial layout from the full NTSSB and set the node means
        as their positions in the layout. 

        Make sure the outer params correspond to longer branches than internal ones.
        """
        import networkx as nx
        self.create_augmented_tree_dict() # create self.node_dict

        G = nx.DiGraph()
        for node in self.node_dict:
            G.add_node(G, node)
            if self.node_dict[node]['parent'] != '-1':
                parent = self.node_dict[node]['parent']
                G.add_edge(parent, node)
        pos = nx.nx_pydot.graphviz_layout(G, prog="twopi")

        self.set_node_means(pos) # to sample observations from nodes in these positions

    def set_node_means(self, pos):
        for node in self.node_dict:
            self.node_dict[node].set_node_mean(pos[node])

    def sync_subtrees(self):
        subtrees = self.get_subtrees()
        for subtree in subtrees:
            subtree.ntssb = self

    def get_node(self, u, key=None, uniform=False, include_leaves=True, variational=True):
        # See in which subtree it lands
        if uniform:
            key, subkey = jax.random.split(key)
            subtree, _, u = self.find_node_uniform(subkey, include_leaves=True)
        else:
            if variational:
                subtree, _, u = self.find_node_variational(u)
            else:
                subtree, _, u = self.find_node(u)

        # See in which node it lands
        if uniform:
            key, subkey = jax.random.split(key)
            _, _, root = subtree.find_node_uniform(subkey, include_leaves=include_leaves)
        else:
            _, _, root = subtree.find_node(u, include_leaves=include_leaves)

        return root

    def sample_assignments(self, num_data):
        self.num_data = num_data

        node_assignments = []
        obs_node_assignments = []
        
        self.assignments = []
        self.subtree_assignments = []
        subtrees = self.get_subtrees()
        for tssb in subtrees:
            tssb.assignments = []
            tssb.remove_data()
        
        # Draw sticks
        rng = np.random.default_rng(self.seed)
        u_vector = rng.random(size=num_data)
        for n in range(num_data):
            u = u_vector[n]
            # See in which subtree it lands
            subtree, _, u = self.find_node(u)

            # See in which node it lands
            node, _, _ = subtree.find_node(u)

            self.assignments.append(node)
            self.subtree_assignments.append(subtree)

            node_assignments.append(node.label)
            obs_node_assignments.append(subtree.label)

            subtree.assignments.append(node)
            subtree.add_datum(n)
            node.add_datum(n)
        
        return node_assignments, obs_node_assignments
    
    def simulate_data(self):
        self.data = np.zeros((self.num_data, self.input_tree.get_param_size()))

        # Reset root node parameters to set data-dependent variables if applicable
        self.root["node"].root["node"].reset_data_parameters()
        
        # Sample observations
        def super_descend(super_root):
            descend(super_root['node'].root)    
            for super_child in super_root['children']:
                super_descend(super_child)

        def descend(root):
            attached_cells = np.array(list(root['node'].data))
            if len(attached_cells) > 0:
                self.data[attached_cells] = root['node'].sample_observations()
            for child in root['children']:
                descend(child)
                
        super_descend(self.root)
        self.data = jnp.array(self.data)

        return self.data

    def add_data(self, data, covariates=None):
        self.data = jnp.array(data)
        self.num_data = data.shape[0]
        if covariates is None:
            self.covariates = np.zeros((self.num_data, 0))
        else:
            self.covariates = covariates
        assert self.covariates.shape[0] == self.num_data

        logger.debug(f"Adding data of shape {data.shape} to NTSSB")

        try:
            # Reset root node parameters to set data-dependent variables if applicable
            self.root["node"].root["node"].reset_data_parameters()
        except AttributeError:
            pass

        # Reset node variational parameters to use this data size
        self.reset_variational_parameters()

    def clear_data(self):
        def descend(root):
            for index, child in enumerate(root["children"]):
                descend(child)
            root["node"].clear_data()

        self.root["node"].clear_data()
        descend(self.root)
        self.assignments = []

    def truncate_subtrees(self):
        def descend(root):
            for index, child in enumerate(root["children"]):
                descend(child)
            root["node"].truncate()

        descend(self.root)

    def set_pivot_dp_alphas(self, tree_dict):
        def descend(super_tree, label, depth=0):
            for i, child in enumerate(tree_dict[label]["children"]):
                super_tree["children"][i]["pivot_tssb"].dp_alpha = tree_dict[child][
                    "dp_alpha_parent_edge"
                ]
                descend(super_tree["children"][i], child, depth + 1)

        descend(self.root, "A")

    def create_new_tree(self, n_extra_per_observed=1):
        # Clear current tree (including subtrees)
        self.reset_tree(
            True
        )
        self.set_weights()

        def get_distance(nodeA, nodeB):
            return np.sqrt(np.sum((nodeA.get_mean() - nodeB.get_mean())**2))

        # Add nodes and set pivots
        def descend(super_tree):
            if super_tree['weight'] != 0: # Add nodes only if it has some mass
                n_nodes = 0
                while n_nodes < n_extra_per_observed:
                    _, _, nodes_roots = super_tree['node'].get_mixture(get_roots=True)
                    # Uniformly choose a node from the subtree
                    rng = np.random.default_rng(super_tree['node'].seed + n_nodes)
                    snode = rng.choice(nodes_roots)
                    super_tree['node'].add_node(snode)
                    n_nodes = n_nodes + 1
                super_tree['node'].reset_node_parameters(**self.node_hyperparams) # adjust parameters to avoid overlapping subnodes
            for i, child in enumerate(super_tree["children"]):
                weights, nodes = super_tree["node"].get_fixed_weights(
                    eta=child["node"].eta
                )
                weights = np.array([w/get_distance(child['node'].root['node'], n) for w, n in zip(weights, nodes)])
                weights = weights / np.sum(weights)
                # rng = np.random.default_rng(super_tree['node'].seed + i)
                # pivot_node = rng.choice(nodes, p=weights)
                pivot_node = nodes[np.argmax(weights)]
                child["pivot_node"] = pivot_node
                child["node"].root["node"].set_parent(pivot_node)

                descend(child)
            super_tree['node'].truncate()
            super_tree['node'].set_weights()
            super_tree['node'].set_pivot_priors()

        descend(self.root)
        self.plot_tree(super_only=False)  # update names

    def sample_new_tree(self, num_data, use_weights=False):
        self.num_data = num_data

        # Clear current tree (including subtrees)
        self.reset_tree(
            use_weights,
        )

        # Break sticks to assign data
        for n in range(num_data):
            # First choose the subtree
            u = rand()
            subtree, _, u = self.find_node(u)

            # Now choose the node
            node, _, _ = subtree.find_node(u)
            node.add_datum(n)
            subtree.assignments.append(node)

            self.assignments.append(node)

        # Remove empty leaves from subtrees
        def descend(super_tree):
            for child in super_tree["children"]:
                child["node"].cull_tree()
                descend(child)

        self.root["node"].cull_tree()
        descend(self.root)

        # And now recursively sample the pivots
        def descend(super_tree):
            for child in super_tree["children"]:
                # We have new subtrees, so update the pivot trees before
                # choosing pivot nodes
                weights, nodes = super_tree["node"].get_fixed_weights(
                    eta=child["node"].eta
                )
                pivot_node = np.random.choice(nodes, p=weights)
                child["pivot_node"] = pivot_node
                child["node"].root["node"].set_parent(pivot_node)

                descend(child)

        descend(self.root)

    def get_param_dict(self):
        """
        Go from a dictionary where each node is a TSSB to a dictionary where each node is a dictionary,
        with `params` and `weight` keys 
        """

        param_dict = {
                "node": self.root['node'].get_param_dict(),
                "weight": self.root['weight'],
                "children": [],
                "obs_param": self.root['node'].root['node'].get_observed_parameters(),
                "label": self.root['label'],
                "color": self.root['color'],
                "size": len(self.root['node']._data),
        }
        def descend(root, root_new):
            for child in root["children"]:
                child_new = {
                        "node": child['node'].get_param_dict(),
                        "weight": child['weight'],
                        "children": [],
                        "obs_param": child['node'].root['node'].get_observed_parameters(),
                        "label": child['label'],
                        "color": child['color'],
                        "size": len(child['node']._data)
                    }
                root_new['children'].append(child_new)
                descend(child, root_new['children'][-1])
        
        descend(self.root, param_dict)
        return param_dict

    # ========= Functions to sample tree parameters. =========
    def sample_pivot_main_sticks(self):
        def descend(super_tree):
            for child in super_tree["children"]:
                child["pivot_tssb"].sample_main_sticks(truncate=True)
                descend(child)

        descend(self.root)

    def sample_pivot_sticks(self, balanced=True):
        def descend(super_tree):
            for child in super_tree["children"]:
                child["pivot_tssb"].sample_sticks(truncate=True, balanced=balanced)
                descend(child)

        descend(self.root)

    def sample_pivots(self):
        def descend(super_tree):
            for child in super_tree["children"]:
                nodes, weights = super_tree["node"].get_fixed_weights()
                pivot_node = np.random.choice(nodes, p=weights)
                child["pivot_node"] = pivot_node
                descend(child)

        descend(self.root)

    # ========= Functions to access tree features. =========
    def get_pivot_tree(self, subtree_label):
        def descend(root):
            if root["label"] == subtree_label:
                return root["pivot_node"], root["pivot_tssb"]
            for child in root["children"]:
                out = descend(child)
                if out is not None:
                    return out

        pivot_node, pivot_tssb = descend(self.root)

        return pivot_node, pivot_tssb

    def get_pivot_reldepth(self, subtree_label):
        # Returns the depth of the parent subtree's node that connects to the
        # root of `subtree`

        # First we need to find this node in the tree to get the a pointer to the
        # pivot node in its parent
        pivot_node, pivot_tssb = self.get_pivot_tree(subtree_label)
        if pivot_node is None or pivot_tssb is None:
            raise Exception(f"Did not find pivot of node {subtree_label}")

        def descend(root, depth=1):
            if root["node"] == pivot_node:
                return depth
            for child in root["children"]:
                n = descend(child, depth + 1)
                if n is not None:
                    return n

        depth = descend(pivot_tssb.root)

        # Depth of parent subtree
        total_depth = pivot_tssb.get_n_levels()

        return depth / total_depth

    def get_node_weights(self, eta=0):
        subtree_weights, subtrees = self.get_mixture()
        nodes = []
        weights = []

        for i, subt in enumerate(subtrees):
            node_weights, subtree_nodes = subt.get_fixed_weights(eta=eta)
            nodes.append(subtree_nodes)
            weights.append(np.array(node_weights).ravel() * subtree_weights[i])

        nodes = np.concatenate(nodes)
        weights = np.concatenate(weights)
        weights = weights / np.sum(weights)

        return nodes, weights

    def get_node_mixture(self):
        # Get node weights and their log likelihoods
        subtree_weights, subtrees = self.get_mixture()
        nodes = []
        weights = []

        for i, subt in enumerate(subtrees):
            node_weights, subtree_nodes = subt.get_variational_mixture()
            nodes.append(subtree_nodes)
            weights.append(np.array(node_weights).ravel() * subtree_weights[i])

        nodes = np.concatenate(nodes)
        weights = np.concatenate(weights).astype(float)
        weights = weights / np.sum(weights)

        return nodes, weights

    def get_tree_roots(self):
        def descend(root):
            sr = [root]
            for child in root["children"]:
                cr = descend(child)
                sr.extend(cr)
            return sr
        return descend(self.root)

    def get_mixture(self, reset_names=False):
        if reset_names:
            self.set_node_names()

        def descend(root, mass):
            weight = [mass * root["main"]]
            subtree = [root["node"]]
            edges = sticks_to_edges(root["sticks"])
            weights = diff(hstack([0.0, edges]))

            for i, child in enumerate(root["children"]):
                (child_weights, child_subtrees) = descend(
                    child, mass * (1.0 - root["main"]) * weights[i]
                )
                weight.extend(child_weights)
                subtree.extend(child_subtrees)
            return (weight, subtree)

        return descend(self.root, 1.0)

    def set_weights(self):
        def descend(root, mass):
            root['weight'] = mass * root["main"]
            edges = sticks_to_edges(root["sticks"])
            weights = diff(hstack([0.0, edges]))
            for i, child in enumerate(root["children"]):
                descend(child, mass * (1.0 - root["main"]) * weights[i])
        return descend(self.root, 1.0)
    
    def set_expected_weights(self):
        def descend(root):
            logprior = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            if root['node'].parent() is not None:
                logprior += root['node'].parent().variational_parameters['sum_E_log_1_nu']
                logprior += root['node'].variational_parameters['E_log_phi']
            root['weight'] = jnp.exp(logprior)
            root['node'].set_expected_weights()
            for child in root['children']:
                descend(child)
        descend(self.root)

    def set_pivot_priors(self):
        def descend(root):
            for child in root['children']:
                root['node'].set_pivot_priors()
                descend(child)
        descend(self.root)

    def get_tree_data_sizes(self, normalized=False):
        trees = self.get_trees()
        sizes = []

        for tree in trees:
            sizes.append(tree.num_data())

        sizes = np.array(sizes)
        if normalized:
            sizes = sizes / np.sum(sizes)
        return np.array(trees), sizes

    def get_node_data_sizes(self, normalized=False):
        nodes = self.get_nodes()
        sizes = []

        for node in nodes:
            sizes.append(len(node.data))

        sizes = np.array(sizes)
        if normalized:
            sizes = sizes / np.sum(sizes)
        return np.array(nodes), sizes

    def get_subtrees(self, get_roots=False):
        def descend(root):
            subtree = [root["node"]]
            roots = [root]

            for i, child in enumerate(root["children"]):
                if get_roots:
                    (child_subtrees, child_roots) = descend(child)
                    roots.extend(child_roots)
                else:
                    child_subtrees = descend(child)
                subtree.extend(child_subtrees)
            if get_roots:
                return (subtree, roots)
            return subtree

        out = descend(self.root)
        if get_roots:
            return list(zip(out[0], out[1]))
        else:
            return out

    def get_subtree_leaves(self):
        def descend(root, l):
            for i, child in enumerate(root["children"]):
                descend(child, l)
            if len(root["children"]) == 0:
                l.append(root)

        li = []
        descend(self.root, li)
        return li

    def _get_nodes(self, get_roots=False):
        # Go to each subtree
        subtree_weights, subtrees = self.get_mixture()
        n = []
        for i, subtree in enumerate(subtrees):
            if get_roots:
                node_weights, nodes, roots = subtree.get_mixture(get_roots=True)
            else:
                node_weights, nodes = subtree.get_mixture(get_roots=False)
            for j, node in enumerate(nodes):
                if get_roots:
                    n.append([node, roots[j]])
                else:
                    n.append(node)

        return n

    def get_node_roots(self):
        def sub_descend(root):
            sr = [root]
            for child in root["children"]:
                cr = sub_descend(child)
                sr.extend(cr)
            return sr
        def descend(super_root):
            roots = []
            sub_roots = sub_descend(super_root["node"].root)
            roots.extend(sub_roots)
            for super_child in super_root["children"]:
                children_roots = descend(super_child)
                roots.extend(children_roots)
            return roots
        return descend(self.root)


    def get_nodes(self):
        def descend(root):
            nodes = [root['node']]
            for child in root['children']:
                nodes.extend(descend(child))
            return nodes
        
        def super_descend(root):
            nodes = descend(root['node'].root)
            for child in root['children']:
                nodes.extend(super_descend(child))
            return nodes

        return super_descend(self.root)

    def get_trees(self):
        def super_descend(root):
            nodes = [root['node']]
            for child in root['children']:
                nodes.extend(super_descend(child))
            return nodes

        return super_descend(self.root)

    def get_width_distribution(self):
        def descend(root, depth, width_vec):
            width_vec[depth - 1] = width_vec[depth - 1] + 1
            if len(root["children"]) == 0:
                return width_vec
            for i, child in enumerate(root["children"]):
                descend(child, depth + 1, width_vec)
            return width_vec

        width_vec = zeros(self.max_depth)
        return descend(self.root, 1, width_vec)

    def get_weight_distribtuion(self):
        def descend(root, mass, depth, mass_vec):
            edges = sticks_to_edges(root["sticks"])
            weights = diff(hstack([0.0, edges]))
            for i, child in enumerate(root["children"]):
                mass_vec[depth] = mass_vec[depth] + mass * weights[i] * child["main"]
                mass_vec = descend(
                    child,
                    mass * (1.0 - child["main"]) * weights[i],
                    depth + 1,
                    mass_vec,
                )
            return mass_vec

        mass_vec = zeros(self.max_depth)
        edges = sticks_to_edges(self.root["sticks"])
        weights = diff(hstack([0.0, edges]))
        if len(weights) > 0:
            mass_vec[0] = weights[0] * self.root["main"]
            return descend(self.root, 1.0 - mass_vec[0], 1, mass_vec)
        else:
            return mass_vec

    def deepest_node_depth(self):
        def descend(root, depth):
            if len(root["children"]) == 0:
                return depth
            deepest = depth
            for i, child in enumerate(root["children"]):
                hdepth = descend(child, depth + 1)
                if deepest < hdepth:
                    deepest = hdepth
            return deepest

        return descend(self.root, 1)

    # Represent TSSB tree with node clones ordered by label, assuming they have one
    def get_clone_sorted_root_dict(self):
        ordered_dict = self.root.copy()

        sorted_idxs = []
        labels = [int(node["label"]) for node in self.root["children"]]
        for i in range(len(self.root["children"])):
            min_idx = sorted_idxs[i - 1] if i > 0 else 0
            for j in range(len(self.root["children"])):
                if labels[j] < labels[min_idx] and j not in sorted_idxs:
                    min_idx = j
            sorted_idxs.append(min_idx)
            labels[min_idx] = np.inf

        new_children = []
        for k in sorted_idxs:
            child = ordered_dict["children"][k]
            new_children.append(child)
        ordered_dict["children"] = new_children

        return ordered_dict
    
    def find_node_uniform(self, key, include_leaves=True):
        def descend(root, key, depth=0):
            if depth >= self.max_depth:
                return (root["node"], [], root)
            elif len(root["children"]) == 0:
                return (root["node"], [], root)
            else:
                key, subkey = jax.random.split(key)
                n_children = len(root["children"])
                if jax.random.bernoulli(subkey, p=1./(n_children+1)):
                    return (root["node"], [], root)
                else:
                    key, subkey = jax.random.split(key)
                    index = jax.random.choice(subkey, len(root["children"]))

                    # Perhaps stop before continuing to a leaf
                    if not include_leaves and len(root["children"][index]["children"]) == 0:
                        return (root["node"], [], root)
                    
                    (node, path, root) = descend(root["children"][index], key, depth + 1)

                    path.insert(0, index)

                    return (node, path, root)

        return descend(self.root, key)

    def find_node_variational(self, u):
        """This function breaks sticks in a tree where each node is a subtree."""

        def descend(root, u, depth=0):
            if depth >= self.max_depth:
                # print >>sys.stderr, "WARNING: Reached maximum depth."
                return (root["node"], [], u)
            else:
                main = np.exp(E_log_beta(root['node'].variational_parameters['delta_1'],root['node'].variational_parameters['delta_2']))
                if u < main:
                    return (root["node"], [], u / main)
                else:
                    # Rescale the uniform variate to the remaining interval.
                    u = (u - main) / (1.0 - main)

                    # Don't need to break sticks
                    children_sticks = np.array(np.exp([E_log_beta(root["children"][i]['node'].variational_parameters['sigma_1'],root["children"][i]['node'].variational_parameters['sigma_2']) for i in range(len(root['children']))]))
                    children_sticks[-1] = 1.
                    edges = 1.0 - cumprod(1.0 - children_sticks)
                    index = sum(u > edges)
                    edges = hstack([0.0, edges])
                    u = (u - edges[index]) / (edges[index + 1] - edges[index])

                    (node, path, u_out) = descend(root["children"][index], u, depth + 1)

                    path.insert(0, index)

                    return (node, path, u_out)

        return descend(self.root, u)    

    def find_node(self, u):
        """This function breaks sticks in a tree where each node is a subtree."""

        def descend(root, u, depth=0):
            if depth >= self.max_depth:
                # print >>sys.stderr, "WARNING: Reached maximum depth."
                return (root["node"], [], u)
            elif u < root["main"]:
                return (root["node"], [], u / root["main"])
            else:
                # Rescale the uniform variate to the remaining interval.
                u = (u - root["main"]) / (1.0 - root["main"])

                # Don't need to break sticks

                edges = 1.0 - cumprod(1.0 - root["sticks"])
                index = sum(u > edges)
                edges = hstack([0.0, edges])
                u = (u - edges[index]) / (edges[index + 1] - edges[index])

                (node, path, u_out) = descend(root["children"][index], u, depth + 1)

                path.insert(0, index)

                return (node, path, u_out)

        return descend(self.root, u)

    # ========= Functions to evaluate quality of tree. =========

    def importance_sampling(self, n_samples=1000):
        """
        Returns posterior samples and importance weights using the variational
        approximation as proposal.
        """
        raise NotImplementedError

    def psis_diagnostic(self, importance_samples):
        """
        Returns the Pareto-smoothed Importance Sampling diagnostic for the
        variational approximation.
        """
        raise NotImplementedError

    def vbis_estimate(self, importance_samples=None, n_samples=1000):
        """
        Returns the VBIS estimate of the marginal likelihood using the fitted
        variational approximation.
        """
        if not importance_samples:
            importance_samples = self.importance_sampling(n_samples=n_samples)
        raise NotImplementedError

    # ========= Functions to update tree parameters given data. =========

    def Eq_log_p_nu(self, dp_alpha, nu_sticks_alpha, nu_sticks_beta):
        l = 0
        aux = digamma(nu_sticks_beta) - digamma(nu_sticks_alpha + nu_sticks_beta)
        l = l + (dp_alpha - 1) * aux - betaln(1, dp_alpha)
        return l

    def Eq_log_q_nu(self, nu_sticks_alpha, nu_sticks_beta):
        l = 0
        aux = digamma(nu_sticks_alpha + nu_sticks_beta)
        aux1 = digamma(nu_sticks_alpha) - aux
        aux2 = digamma(nu_sticks_beta) - aux
        l = (
            l
            + (nu_sticks_alpha - 1) * aux1
            + (nu_sticks_beta - 1) * aux2
            - betaln(nu_sticks_alpha, nu_sticks_beta)
        )
        return l

    def Eq_log_p_psi(self, dp_gamma, psi_sticks_alpha, psi_sticks_beta):
        l = 0
        aux = digamma(psi_sticks_beta) - digamma(psi_sticks_alpha + psi_sticks_beta)
        l = l + (dp_gamma - 1) * aux - betaln(1, dp_gamma)
        return l

    def Eq_log_q_psi(self, psi_sticks_alpha, psi_sticks_beta):
        l = 0
        aux = digamma(psi_sticks_alpha + psi_sticks_beta)
        aux1 = digamma(psi_sticks_alpha) - aux
        aux2 = digamma(psi_sticks_beta) - aux
        l = (
            l
            + (psi_sticks_alpha - 1) * aux1
            + (psi_sticks_beta - 1) * aux2
            - betaln(psi_sticks_alpha, psi_sticks_beta)
        )
        return l

    def Eq_log_p_tau(self, tau_alpha, tau_beta):
        l = 0
        aux = digamma(tau_beta) - digamma(tau_alpha + tau_beta)
        l = l + (2 - 1) * aux - betaln(1, 2)
        return l

    def Eq_log_q_tau(self, tau_alpha, tau_beta):
        l = 0
        aux = digamma(tau_alpha + tau_beta)
        aux1 = digamma(tau_alpha) - aux
        aux2 = digamma(tau_beta) - aux
        l = (
            l
            + (tau_alpha - 1) * aux1
            + (tau_beta - 1) * aux2
            - betaln(tau_alpha, tau_beta)
        )
        return l

    def make_batches(self, batch_size=None, seed=42):
        if batch_size is None:
            batch_size = self.num_data
        
        if batch_size > self.num_data:
            batch_size = self.num_data

        self.batch_size = batch_size

        rng = np.random.RandomState(seed)
        perm = rng.permutation(self.num_data)

        num_complete_batches, leftover = divmod(self.num_data, self.batch_size)
        self.num_batches = num_complete_batches + bool(leftover)

        self.batch_indices = []
        for i in range(self.num_batches):
            batch_idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            self.batch_indices.append(batch_idx)

        self.reset_sufficient_statistics()

    def get_top_node_obs(self, q=70):
        """
        Get data which is very well explained by the node they attach to
        """
        def sub_descend(root):
            # Get cells attached to this node
            idx = np.where(self.assignments == root['node'])[0]
            top_obs = root['node'].get_top_obs(q=q, idx=idx)
            for child in root['children']:
                top_obs = np.concatenate([top_obs,sub_descend(child)])
            return top_obs

        def descend(root):
            top_obs = sub_descend(root['node'].root)
            for child in root['children']:
                top_obs = np.concatenate([top_obs, descend(child)])
            return top_obs
        
        top_obs = descend(self.root)
        top_obs = np.unique(top_obs).astype(int)
        return top_obs

    def compute_elbo(self, memoized=True, batch_idx=None, **kwargs):
        if memoized:
            return self.compute_elbo_suff()
        else:
            return self.compute_elbo_batch(batch_idx=batch_idx)

    def compute_elbo_batch(self, batch_idx=None):
        """
        Compute the ELBO of the model in a tree traversal, abstracting away the likelihood and kernel specific functions
        for the model. The seed is used for MC sampling from the variational distributions for which Eq[logp] is generally not analytically
        available (which is the likelihood and the kernel distribution).

        If batch_idx is not None, return an estimate of the ELBO based on just the subset of data in batch_idx.
        Otherwise, use sufficient statistics.
        """
        if batch_idx is None:
            idx = jnp.arange(self.num_data)
        else:
            idx = self.batch_indices[batch_idx]
        def descend(root, depth=0, local_contrib=0, global_contrib=0, psi_priors=None):
            # Traverse inner TSSB
            subtree_ll_contrib, subtree_ass_contrib, subtree_node_contrib = root['node'].compute_elbo(idx)
            ll_contrib = subtree_ll_contrib * root['node'].variational_parameters['q_c'][idx]

            # Assignments
            ## E[log p(c|nu,psi)]
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            eq_logp_c = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            if root['node'].parent() is not None:
                eq_logp_c += root['node'].parent().variational_parameters['sum_E_log_1_nu']
                eq_logp_c += root['node'].variational_parameters['E_log_phi']
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu + root['node'].parent().variational_parameters['sum_E_log_1_nu']
            else:
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu
            ## E[log q(c)]
            eq_logq_c = jax.lax.select(root['node'].variational_parameters['q_c'][idx] != 0, 
                        root['node'].variational_parameters['q_c'][idx] * jnp.log(root['node'].variational_parameters['q_c'][idx]), 
                        root['node'].variational_parameters['q_c'][idx])
            ass_contrib = eq_logp_c*root['node'].variational_parameters['q_c'][idx] - eq_logq_c + \
                            subtree_ass_contrib * root['node'].variational_parameters['q_c'][idx]

            # Sticks
            E_log_nu = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl = (root['beta_nu'] - root['node'].variational_parameters['delta_2']) * E_log_1_nu
            nu_kl -= (root['node'].variational_parameters['delta_1'] - root['alpha_nu']) * E_log_nu
            nu_kl += logbeta_func(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl -= logbeta_func(root['alpha_nu'], root['beta_nu'])
            psi_kl = 0.
            if depth != 0:
                E_log_psi = E_log_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                E_log_1_psi = E_log_1_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl = (psi_priors['beta_psi'] - root['node'].variational_parameters['sigma_2']) * E_log_1_psi
                psi_kl -= (root['node'].variational_parameters['sigma_1'] - psi_priors['alpha_psi']) * E_log_psi
                psi_kl += logbeta_func(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl -= logbeta_func(psi_priors['alpha_psi'], psi_priors['beta_psi'])
            stick_contrib = nu_kl + psi_kl

            self.n_total_nodes += root['node'].n_nodes
            sum_E_log_1_psi = 0.
            for i, child in enumerate(root['children']):
                # Auxiliary quantities
                ## Branches
                E_log_psi = E_log_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                child['node'].variational_parameters['E_log_phi'] = E_log_psi + sum_E_log_1_psi
                E_log_1_psi = E_log_1_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                sum_E_log_1_psi += E_log_1_psi

                # Go down
                local_contrib, global_contrib = descend(child, depth=depth+1, local_contrib=local_contrib, global_contrib=global_contrib, psi_priors=root['psi_priors'][i])

            local_contrib += ll_contrib + ass_contrib
            global_contrib += subtree_node_contrib + stick_contrib
            return local_contrib, global_contrib
        
        self.n_total_nodes = 0
        local_contrib, global_contrib = descend(self.root)

        # Add tree-independent contributions
        global_contrib += self.num_data/len(idx) * (self.root['node'].root['node'].compute_local_priors(idx) + self.root['node'].root['node'].compute_local_entropies(idx))
        global_contrib += self.root['node'].root['node'].compute_global_priors() + self.root['node'].root['node'].compute_global_entropies()
        
        elbo = self.num_data/len(idx) * np.sum(local_contrib) + global_contrib
        self.elbo = elbo
        return elbo

    def compute_elbo_suff(self):
        def descend(root, depth=0, local_contrib=0, global_contrib=0, psi_priors=None):
            # Traverse inner TSSB
            ll_contrib, subtree_ass_contrib, subtree_node_contrib = root['node'].compute_elbo_suff()

            # Assignments
            ## E[log p(c|nu,psi)]
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            eq_logp_c = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            if root['node'].parent() is not None:
                eq_logp_c += root['node'].parent().variational_parameters['sum_E_log_1_nu']
                eq_logp_c += root['node'].variational_parameters['E_log_phi']
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu + root['node'].parent().variational_parameters['sum_E_log_1_nu']
            else:
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu
            ass_contrib = eq_logp_c*root['node'].suff_stats['mass']['total'] + root['node'].suff_stats['ent']['total'] + subtree_ass_contrib

            # Sticks
            E_log_nu = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl = (root['beta_nu'] - root['node'].variational_parameters['delta_2']) * E_log_1_nu
            nu_kl -= (root['node'].variational_parameters['delta_1'] - root['alpha_nu']) * E_log_nu
            nu_kl += logbeta_func(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl -= logbeta_func(root['alpha_nu'], root['beta_nu'])
            psi_kl = 0.
            if depth != 0:
                E_log_psi = E_log_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                E_log_1_psi = E_log_1_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl = (psi_priors['beta_psi'] - root['node'].variational_parameters['sigma_2']) * E_log_1_psi
                psi_kl -= (root['node'].variational_parameters['sigma_1'] - psi_priors['alpha_psi']) * E_log_psi
                psi_kl += logbeta_func(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl -= logbeta_func(psi_priors['alpha_psi'], psi_priors['beta_psi'])
            stick_contrib = nu_kl + psi_kl

            self.n_total_nodes += root['node'].n_nodes
            sum_E_log_1_psi = 0.
            for i, child in enumerate(root['children']):
                # Auxiliary quantities
                ## Branches
                E_log_psi = E_log_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                child['node'].variational_parameters['E_log_phi'] = E_log_psi + sum_E_log_1_psi
                E_log_1_psi = E_log_1_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                sum_E_log_1_psi += E_log_1_psi

                # Go down
                local_contrib, global_contrib = descend(child, depth=depth+1, local_contrib=local_contrib, global_contrib=global_contrib, psi_priors=root['psi_priors'][i])

            local_contrib += ll_contrib + ass_contrib
            global_contrib += subtree_node_contrib + stick_contrib
            return local_contrib, global_contrib
        
        self.n_total_nodes = 0
        local_contrib, global_contrib = descend(self.root)

        # Add tree-independent contributions
        global_contrib += self.root['node'].root['node'].local_suff_stats['locals_kl']['total']
        global_contrib += self.root['node'].root['node'].compute_global_priors() + self.root['node'].root['node'].compute_global_entropies()
        
        elbo = local_contrib + global_contrib
        self.elbo = elbo
        return elbo

    def learn_model(self, n_epochs, seed=42, memoized=True, update_roots=True, update_globals=True, adaptive=True, return_trace=False, 
                     locals_names=None, globals_names=None, **kwargs):
        key = jax.random.PRNGKey(seed)
        elbos = []
        states = None
        local_states = None
        if adaptive:
            local_states = self.root['node'].root['node'].initialize_local_opt_states(param_names=locals_names)
            states = self.root['node'].root['node'].initialize_global_opt_states(param_names=globals_names)
        it = 0
        idx = None
        for i in range(n_epochs):
            for batch_idx in range(self.num_batches):
                key, subkey = jax.random.split(key)
                if update_globals:
                    states = self.update_global_params(subkey, idx=idx, batch_idx=batch_idx, adaptive=adaptive, states=states, i=it, param_names=globals_names, **kwargs)                
                local_states = self.update_local_params(subkey, batch_idx=batch_idx, adaptive=adaptive, states=local_states, i=it, 
                                                        param_names=locals_names, update_globals=update_globals, **kwargs)
                if memoized:
                    self.update_sufficient_statistics(batch_idx=batch_idx)    
                self.update_node_params(subkey, i=it, adaptive=adaptive, memoized=memoized, **kwargs)
                if update_roots:
                    self.update_root_node_params(subkey, memoized=memoized, batch_idx=batch_idx, adaptive=adaptive, i=it, **kwargs)                
                self.update_pivot_probs()                    
                it += 1
                if return_trace:
                    elbos.append(self.compute_elbo(memoized=memoized, batch_idx=batch_idx))
        
        if return_trace:
            return elbos

    def learn_roots(self, n_epochs, seed=42, adaptive=True, return_trace=False, **kwargs):
        key = jax.random.PRNGKey(seed)
        elbos = []
        it = 0
        for i in range(n_epochs):
            for batch_idx in range(self.num_batches):
                key, subkey = jax.random.split(key)
                self.update_root_node_params(subkey, batch_idx=batch_idx, adaptive=adaptive, i=it, **kwargs)
                if return_trace:
                    elbos.append(self.compute_elbo(batch_idx=batch_idx, **kwargs))
                it += 1
        
        if return_trace:
            return elbos

    def learn_globals(self, n_epochs, globals_names=None, locals_names=None, ass_anneal=1., ent_anneal=1., update_ass=True, update_locals=True, update_roots=False, subset=None, adaptive=True, seed=42, return_trace=False, **kwargs):
        key = jax.random.PRNGKey(seed)
        elbos = []
        states = None
        local_states = None
        if adaptive:
            local_states = self.root['node'].root['node'].initialize_local_opt_states(param_names=locals_names)
            states = self.root['node'].root['node'].initialize_global_opt_states(param_names=globals_names)
        it = 0
        idx = None
        for i in range(n_epochs):
            for batch_idx in range(self.num_batches):
                key, subkey = jax.random.split(key)
                if subset is not None:
                    idx = jnp.array(list(set(self.batch_indices[batch_idx]).intersection(set(subset))))
                local_states = self.update_local_params(subkey, idx=idx, batch_idx=batch_idx, ass_anneal=ass_anneal, ent_anneal=ent_anneal, 
                                                        update_ass=update_ass, update_globals=update_locals, adaptive=adaptive, states=local_states, i=it, 
                                                        param_names=locals_names, **kwargs)
                states = self.update_global_params(subkey, idx=idx, batch_idx=batch_idx, adaptive=adaptive, states=states, i=it, param_names=globals_names, **kwargs)
                if update_roots:
                    self.update_root_node_params(subkey, memoized=False, adaptive=adaptive, i=it, **kwargs)
                it += 1
                if return_trace:
                    elbos.append(self.compute_elbo_batch(batch_idx=batch_idx))
    
        if return_trace:
            return elbos  

    def learn_params(self, n_epochs, seed=42, memoized=True, adaptive=True, update_roots=False, return_trace=False, **kwargs):    
        key = jax.random.PRNGKey(seed)
        elbos = []
        it = 0
        for i in range(n_epochs):
            for batch_idx in range(self.num_batches):
                key, subkey = jax.random.split(key)
                self.update_local_params(subkey, batch_idx=batch_idx, update_globals=False, **kwargs)
                if memoized:
                    self.update_sufficient_statistics(batch_idx=batch_idx)    
                if update_roots:
                    self.update_root_node_params(subkey, memoized=memoized, adaptive=adaptive, i=it, **kwargs)
                self.update_node_params(subkey, i=it, adaptive=adaptive, memoized=memoized, **kwargs)
                self.update_pivot_probs()                    
                it += 1
                if return_trace:
                    elbos.append(self.compute_elbo(memoized=memoized, batch_idx=batch_idx))
        
        if return_trace:
            return elbos

    def update_sufficient_statistics(self, batch_idx=None):
        """
        Go to each node and update its sufficient statistics. Set the suff stats of batch_idx,
        and update the total suff stats
        """
        def descend(root):
            root['node'].update_sufficient_statistics(batch_idx=batch_idx)
            for child in root['children']:
                descend(child)
        
        descend(self.root)

    def update_local_params(self, key, ass_anneal=1., ent_anneal=1., idx=None, batch_idx=None, states=None, adaptive=False, i=0, step_size=0.0001, mc_samples=10, update_ass=True, update_outer_ass=True, update_globals=True, **kwargs):
        """
        This performs a tree traversal to update the sample to node attachment probabilities and other sample-specific parameters
        """
        if idx is None:
            if batch_idx is None:
                idx = jnp.arange(self.num_data)
            else:
                idx = self.batch_indices[batch_idx]

        take_gradients = False
        if update_globals:
            take_gradients = True

        def descend(root, local_grads=None):
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            logprior = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            if root['node'].parent() is not None:
                logprior += root['node'].parent().variational_parameters['sum_E_log_1_nu']
                logprior += root['node'].variational_parameters['E_log_phi']
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu + root['node'].parent().variational_parameters['sum_E_log_1_nu']
            else:
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu

            # Traverse inner TSSB
            ll_node_sum, ent_node_sum, local_grads_down = root['node'].update_local_params(idx, update_ass=update_ass, take_gradients=take_gradients)
            new_log_probs = ass_anneal*(ll_node_sum + logprior + ent_node_sum)
            if update_ass and update_outer_ass:
                root['node'].variational_parameters['q_c'] = root['node'].variational_parameters['q_c'].at[idx].set(new_log_probs)
            if local_grads_down is not None:
                if local_grads is None:
                    local_grads = list(local_grads_down)
                else:
                    for ii, grads in enumerate(list(local_grads_down)):
                        local_grads[ii] += grads
            logqs = [new_log_probs]
            sum_E_log_1_psi = 0.
            for child in root['children']:
                E_log_psi = E_log_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                child['node'].variational_parameters['E_log_phi'] = E_log_psi + sum_E_log_1_psi
                E_log_1_psi = E_log_1_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                sum_E_log_1_psi += E_log_1_psi

                # Go down
                child_log_probs, child_local_grads = descend(child, local_grads=local_grads)
                logqs.extend(child_log_probs)
                if child_local_grads is not None:
                    for ii, grads in enumerate(list(child_local_grads)):
                        local_grads[ii] += grads
            
            return logqs, local_grads # batch-sized

        if update_globals:
            # Take MC sample and its gradient wrt parameters
            key, sample_grad = self.root['node'].root['node'].local_sample_and_grad(idx, key, n_samples=mc_samples)
            locals_curr_sample, locals_params_grad = sample_grad
            self.root['node'].root['node'].set_local_sample(locals_curr_sample, idx=idx)

        # Traverse tree
        logqs, locals_sample_grad = descend(self.root)

        if update_globals:
            locals_prior_grads = self.root['node'].root['node'].compute_locals_prior_grad(locals_curr_sample)
            for ii, grads in enumerate(locals_prior_grads):
                locals_sample_grad[ii] += grads

            # Gradient of entropy wrt parameters
            locals_entropies_grad = self.root['node'].root['node'].compute_locals_entropy_grad(idx)

            # Take gradient step
            if adaptive:
                states = self.root['node'].root['node'].update_local_params_adaptive(idx, locals_params_grad, locals_sample_grad, locals_entropies_grad, ent_anneal=ent_anneal, step_size=step_size,
                        states=states, i=i, **kwargs)
            else:
                self.root['node'].root['node'].update_local_params(idx, locals_params_grad, locals_sample_grad, locals_entropies_grad, ent_anneal=ent_anneal, step_size=step_size, **kwargs)

            # Resample and store 
            key, sample_grad = self.root['node'].root['node'].local_sample_and_grad(idx, key, n_samples=mc_samples)
            locals_curr_sample, _ = sample_grad
            self.root['node'].root['node'].set_local_sample(locals_curr_sample, idx=idx)

        if update_ass and update_outer_ass:
            # Compute LSE
            logqs = jnp.array(logqs).T
            self.variational_parameters['LSE_c'] = jax.scipy.special.logsumexp(logqs, axis=1)
            # Set probs 
            def descend(root):
                newvals = jnp.exp(root['node'].variational_parameters['q_c'][idx] - self.variational_parameters['LSE_c'])
                root['node'].variational_parameters['q_c'] = root['node'].variational_parameters['q_c'].at[idx].set(newvals)
                for child in root['children']:
                    descend(child)
            descend(self.root)
            
        return states

    def update_global_params(self, key, idx=None, batch_idx=None, mc_samples=10, adaptive=False, step_size=0.0001, states=None, i=0, **kwargs):
        """
        This performs a tree traversal to update the global parameters.
        The global parameters are updated using stochastic mini-batch VI.
        """
        if idx is None:
            if batch_idx is None:
                idx = jnp.arange(self.num_data)
            else:
                idx = self.batch_indices[batch_idx]
        def descend(root, globals_grads=None):
            globals_grads_down = root['node'].get_global_grads(idx)
            if globals_grads_down is not None:
                if globals_grads is None:
                    globals_grads = list(globals_grads_down)
                else:
                    for ii, grads in enumerate(list(globals_grads_down)):
                        globals_grads[ii] += grads
            for child in root['children']:
                child_globals_grads = descend(child, globals_grads=globals_grads)
                if child_globals_grads is not None:
                    for ii, grads in enumerate(list(child_globals_grads)):
                        globals_grads[ii] += grads
            return globals_grads

        # Take MC sample and its gradient wrt parameters
        key, sample_grad = self.root['node'].root['node'].global_sample_and_grad(key, n_samples=mc_samples)
        globals_curr_sample, globals_params_grad = sample_grad
        self.root['node'].root['node'].set_global_sample(globals_curr_sample)

        # Get gradient of loss of data likelihood weighted by assignment probability to each node wrt current sample of global params
        globals_sample_grad = descend(self.root)

        # Scale gradient by batch size
        for ii in range(len(globals_sample_grad)):
            globals_sample_grad[ii] *= self.num_data/len(idx)

        # Gradient of prior wrt sample
        globals_prior_grads = self.root['node'].root['node'].compute_globals_prior_grad(globals_curr_sample)
        len_globals_in_ll = len(globals_sample_grad)
        # Add the priors
        for ii in range(len_globals_in_ll):
            globals_sample_grad[ii] += globals_prior_grads[ii]

        # Extend to hierarchical parameters
        for grads in globals_prior_grads[len_globals_in_ll:]:
            globals_sample_grad.append(grads)

        # Gradient of entropy wrt parameters
        globals_entropies_grad = self.root['node'].root['node'].compute_globals_entropy_grad()

        # Take gradient step
        if adaptive:
            states = self.root['node'].root['node'].update_global_params_adaptive(globals_params_grad, globals_sample_grad, 
                                                            globals_entropies_grad, step_size=step_size, states=states, i=i, **kwargs)
        else:
            states = self.root['node'].root['node'].update_global_params(globals_params_grad, globals_sample_grad, 
                                                                globals_entropies_grad, step_size=step_size, **kwargs)

        # Resample and store 
        key, sample_grad = self.root['node'].root['node'].global_sample_and_grad(key, n_samples=mc_samples)
        globals_curr_sample, _ = sample_grad
        self.root['node'].root['node'].set_global_sample(globals_curr_sample)

        if adaptive:
            return states

    def update_node_params(self, key, memoized=True, **kwargs):
        """
        This performs a tree traversal to update the stick parameters and the kernel parameters.
        The node parameters are updated using stochastic memoized VI.
        """
        def descend(root, depth=0):
            mass_down = 0
            for i, child in enumerate(root["children"][::-1]):
                j = len(root['children'])-1-i
                child_mass = descend(child, depth + 1)
                child['node'].variational_parameters['sigma_1'] = root['psi_priors'][j]["alpha_psi"] + child_mass
                child['node'].variational_parameters['sigma_2'] = root['psi_priors'][j]["beta_psi"] + mass_down
                mass_down += child_mass

            # Traverse inner TSSB
            root['node'].update_node_params(key, memoized=memoized, **kwargs)

            if memoized:
                mass_here = root['node'].suff_stats['mass']['total']
            else:
                mass_here = jnp.sum(root['node'].variational_parameters['q_c'])
            root['node'].variational_parameters['delta_1'] = root['alpha_nu'] + mass_here
            root['node'].variational_parameters['delta_2'] = root['beta_nu'] + mass_down

            return mass_here + mass_down
        
        descend(self.root)

    def update_root_node_params(self, key, memoized=True, adaptive=True, i=0, **kwargs):
        """
        This performs a tree traversal to update the kernel parameters of root nodes
        The node parameters are updated using stochastic memoized VI.

        Go inside TSSB1, compute the gradient of TSSB2's root wrt TSSB1's nodes, return their sum
        Go inside TSSB2, compute the gradient of TSSB2's root wrt TSSB3's root, add to previous gradient and update
        Repeat
        """
        def descend(root, depth=0):
            ll_grads = []
            locals_grads = []
            children_grads = []
            for child in root["children"]:
                child_ll_grad, child_locals_grads, child_children_grads = descend(child, depth + 1)
                ll_grads.append(child_ll_grad)
                locals_grads.append(child_locals_grads)
                children_grads.append(child_children_grads)

            if len(root["children"]) > 0:
                # Compute gradient of children roots wrt possible parents here
                parent_grads = root['node'].compute_children_root_node_grads(**kwargs)

                # Update parameters of each child root
                for ii, child in enumerate(root["children"]):
                    ll_grad = ll_grads[ii]
                    local_grad = locals_grads[ii]
                    children_grad = children_grads[ii]
                    parent_grad = parent_grads[ii]
                    child['node'].update_root_node_params(key, ll_grad, local_grad, children_grad, parent_grad, adaptive=adaptive, i=i, **kwargs)

            if depth > 0: # root of root TSSB has no parameters
                # Sample root node and compute gradients wrt children (including child roots)
                # direction_grads = [direction_params_grad, direction_params_entropy_grad, direction_sample_grad]
                # state_grads = [state_params_grad, state_params_entropy_grad, state_sample_grad]
                ll_grad, locals_grads, children_grads = root['node'].sample_grad_root_node(key, memoized=memoized, **kwargs)                

                return ll_grad, locals_grads, children_grads
        
        descend(self.root)        

    def update_pivot_probs(self):
        def descend(root):
            if len(root["children"]) > 0:
                root["node"].update_pivot_probs()
            for child in root["children"]:
                descend(child)
        descend(self.root)

    def assign_samples(self):
        def descend(root):
            nodes_probs = [root['node'].variational_parameters['q_z'].ravel() * root['node'].tssb.variational_parameters['q_c'].ravel()]
            nodes = [root['node']]
            for child in root["children"]:
                cnodes_probs, cnodes = descend(child)
                nodes_probs.extend(cnodes_probs)
                nodes.extend(cnodes)
            return nodes_probs, nodes
        def super_descend(super_root):
            nodes_probs_lists = []
            nodes_lists = []
            nodes_probs, nodes = descend(super_root["node"].root)
            nodes_probs_lists = [nodes_probs]
            nodes_lists = [nodes]
            for super_child in super_root["children"]:
                children_nodes_probs, children_nodes = super_descend(super_child)
                nodes_probs_lists.extend(children_nodes_probs)
                nodes_lists.extend(children_nodes)
            return nodes_probs_lists, nodes_lists
        
        nodes_probs, nodes = super_descend(self.root)
        nodes = [x for xs in nodes for x in xs]
        nodes_probs = [x for xs in nodes_probs for x in xs]
        nodes_probs = np.array(nodes_probs).T
        self.assignments = np.array(nodes)[np.argmax(nodes_probs, axis=1)]
        for node in nodes:
            node.remove_data()
            node.add_data(np.where(self.assignments == node)[0])
            node.tssb._data.update(np.where(self.assignments == node)[0])

    def assign_pivots(self):
        def descend(root):
            pivot_probs_nodes = [root['node'].get_pivot_probabilities(i) for i in range(len(root['children']))]
            for i, child in enumerate(root['children']):
                pivot_probs = [l for l in pivot_probs_nodes[i][0]]
                pivot_nodes = [l for l in pivot_probs_nodes[i][1]]
                child['pivot_node'] = pivot_nodes[np.argmax(pivot_probs)]
                child['node'].root['node'].set_parent(child['pivot_node'])
                descend(child)
        descend(self.root)                   

    def remove_pivots(self):
        def descend(root):
            for i, child in enumerate(root['children']):
                child['pivot_node'] = None
                child['node'].root['node'].set_parent(None)
                descend(child)
        descend(self.root)    

    # ========= Functions to update tree structure. =========

    def prune_subtrees(self):
        # Remove all nodes except observed ones
        def descend(root):
            def sub_descend(sub_root):
                for sub_child in sub_root['children']:
                    sub_descend(sub_child)
                    root['node'].merge_nodes(sub_root, sub_child, sub_root)
            sub_descend(root['node'].root)
            for child in root['children']:
                descend(child)
        descend(self.root)

    def birth(self, source, seed=None):
        def sub_descend(root, target_root=None):
            if source == root['node'].label: # node label
                target_root = root
            for child in root['children']:
                if target_root is not None:
                    return target_root
                else:
                    target_root = sub_descend(child, target_root=target_root)
            return target_root

        def descend(root):
            if root['node'].label in source: # TSSB label
                target_root = sub_descend(root['node'].root)
                root['node'].add_node(target_root, seed=seed)
            for child in root['children']:
                descend(child)

        descend(self.root)

    def merge(self, source, target):
        def sub_descend(root, parent_root=None, source_root=None, target_root=None):
            if target == root['node'].label: # node label
                target_root = root
            for child in root['children']:
                if source == child['node'].label: # node label
                    source_root = child
                if target == child['node'].label: # node label
                    target_root = child
                if source_root is not None and target_root is not None and parent_root is None:
                    parent_root = root if source_root['node'].parent().label == root['node'].label else target_root
                    return parent_root, source_root, target_root
                else:
                    parent_root, source_root, target_root = sub_descend(child, parent_root=parent_root, source_root=source_root, target_root=target_root)
            return parent_root, source_root, target_root

        def descend(root):
            if root['node'].label in source: # TSSB label
                parent_root, source_root, target_root = sub_descend(root['node'].root)
                root['node'].merge_nodes(parent_root, source_root, target_root)
            for child in root['children']:
                descend(child)

        descend(self.root)


    def swap_nodes(self, parent_node, child_node):
        # Put parameters of child in parent and of parent in child
        parent_params = deepcopy(parent_node.variational_parameters)
        parent_suffstats = deepcopy(parent_node.suff_stats)
        child_params = deepcopy(child_node.variational_parameters)
        child_suffstats = deepcopy(child_node.suff_stats)
        parent_node.variational_parameters = child_params
        parent_node.suff_stats = child_suffstats
        child_node.variational_parameters = parent_params
        child_node.suff_stats = parent_suffstats

    def swap_root(self, parent, child):
        def descend(root, depth=0):
            if depth > 0:
                if root['node'].label == parent: # TSSB label
                    for ch in root['node'].root['children']:
                        print(ch['node'].label)
                        if ch['node'].label == child:
                            self.swap_nodes(root['node'].root['node'], ch['node'])
            for ch in root['children']:
                descend(ch, depth+1)    

        descend(self.root)

    def remove_last_leaf_node(self, parent_label):
        nodes = self._get_nodes(get_roots=True)
        node_labels = np.array([node[0].label for node in nodes])
        node_idx = np.where(node_labels == parent_label)[0][0]
        root = nodes[node_idx][1]

        # Remove last child
        root["sticks"] = root["sticks"][:-2]
        root["children"][-1]["node"].kill()
        del root["children"][-1]["node"]
        root["children"] = root["children"][:-2]

    def prune_reattach(self, node, new_parent):
        nodes = self._get_nodes(get_roots=True)
        roots = [n[1] for n in nodes]
        nodes = np.array([n[0] for n in nodes])
        if isinstance(node, str) or isinstance(new_parent, str):
            self.plot_tree(super_only=False)
            node_labels = np.array([n.label for n in nodes])
        if isinstance(node, str):
            node_idx = np.where(node_labels == node)[0][0]
            node = nodes[node_idx]
        else:
            node_idx = np.where(nodes == node)[0][0]
        if isinstance(new_parent, str):
            new_parent_idx = np.where(node_labels == new_parent)[0][0]
            new_parent = nodes[new_parent_idx]
        else:
            new_parent_idx = np.where(nodes == new_parent)[0][0]

        assert node.tssb == new_parent.tssb

        node_root = roots[node_idx]
        new_parent_root = roots[new_parent_idx]
        prev_parent_idx = np.where(np.array(nodes) == node.parent())[0][0]

        # Move subtree
        node.set_parent(new_parent)

        # Update dict: copy dict into new parent
        new_parent_root["children"].append(node_root)
        new_parent_root["sticks"] = np.vstack([new_parent_root["sticks"], 1.0])

        # Remove dict from previous parent
        childnodes = np.array([n["node"] for n in roots[prev_parent_idx]["children"]])
        tokeep = np.where(childnodes != roots[node_idx]["node"])[0].astype(int).ravel()
        roots[prev_parent_idx]["sticks"] = roots[prev_parent_idx]["sticks"][tokeep]
        roots[prev_parent_idx]["children"] = list(
            np.array(roots[prev_parent_idx]["children"])[tokeep]
        )

        # Reset kernel variational parameters
        n_genes = node.cnvs.size
        # node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = np.log(node.unobserved_factors_kernel_concentration_caller())*np.ones((n_genes,))
        # node.variational_parameters['locals']['unobserved_factors_kernel_log_std'] += .5

    def extract_pivot(self, node):
        """
        extract_pivot(B):
        A-0 -> B -> B-0 to A-0 -> A-0-0 -> B -> B-0
        Put unobserved factors of B also in A-0-0
        """
        if isinstance(node, str):
            self.plot_tree(super_only=False)
            nodes = self.get_nodes(None)
            node_labels = np.array([node.label for node in nodes])
            node = nodes[np.where(node_labels == node)[0][0]]

        if node.parent() is None:
            raise ValueError("Can't pull from root tree")
        if not node.is_observed:
            raise ValueError("Can't pull unobserved node")

        parent_node = node.parent()

        # Add node below parent
        new_node_root = self.add_node_to(parent_node, return_parent_root=False)
        new_node = new_node_root["node"]
        paramsB = np.array(
            node.variational_parameters["locals"]["unobserved_factors_mean"]
        )
        paramsB_std = np.array(
            node.variational_parameters["locals"]["unobserved_factors_log_std"]
        )
        paramsB_k = np.array(
            node.variational_parameters["locals"]["unobserved_factors_kernel_log_shape"]
        )
        paramsB_k_std = np.array(
            node.variational_parameters["locals"]["unobserved_factors_kernel_log_rate"]
        )

        # Set new node's parameters equal to the previous parameters of node
        new_node.variational_parameters["locals"]["unobserved_factors_mean"] = np.array(
            paramsB
        )
        new_node.variational_parameters["locals"][
            "unobserved_factors_log_std"
        ] = np.array(paramsB_std)
        new_node.variational_parameters["locals"][
            "unobserved_factors_kernel_log_shape"
        ] = np.array(paramsB_k)
        new_node.variational_parameters["locals"][
            "unobserved_factors_kernel_log_rate"
        ] = np.array(paramsB_k_std)
        new_node.set_mean(variational=True)

        # Increase the kernel of node a bit to help the useless events disappear
        # node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] *= 1.

        # Make room for the child
        new_node.parent().variational_parameters["locals"]["nu_log_mean"] = np.array(
            0.0
        )
        new_node.parent().variational_parameters["locals"]["nu_log_std"] = np.array(0.0)

        # Update pivot
        self.pivot_reattach_to(node.tssb, new_node)
        node.set_mean(variational=True)

        # Open up subtree root's parameters
        n_genes = node.cnvs.size

        # Remove kernel from most affected genes in node -- they are now explained by the top
        affected_genes = np.where(
            np.abs(node.variational_parameters["locals"]["unobserved_factors_mean"])
            > 0.5
        )[0]
        node.variational_parameters["locals"]["unobserved_factors_kernel_log_shape"][
            affected_genes
        ] = np.log(node.unobserved_factors_kernel_concentration_caller())

        # node.variational_parameters['locals']['unobserved_factors_log_std'] += .5
        # node.variational_parameters['locals']['unobserved_factors_kernel_log_std'] += .5

        return new_node_root

    def path_to_node(self, node):
        path = []
        path.append(node)
        parent = node.parent()
        while parent is not None:
            path.append(parent)
            parent = parent.parent()
        return path[::-1][:]

    def get_mrca(self, nodeA, nodeB):
        pathA = np.array(self.path_to_node(nodeA))
        pathB = np.array(self.path_to_node(nodeB))
        path = []
        # Get MRCA
        i = -1
        for node in pathA:
            if node in pathB:
                i += 1
            else:
                break
        mrca = pathA[i]
        return mrca

    def path_between_nodes(self, nodeA, nodeB):
        pathA = np.array(self.path_to_node(nodeA))
        pathB = np.array(self.path_to_node(nodeB))
        path = []
        # Get MRCA
        i = -1
        for node in pathA:
            if node in pathB:
                i += 1
            else:
                break
        mrca = pathA[i]
        pathA = np.array(pathA[::-1])
        # Get path from A to MRCA
        path = path + list(pathA[: np.where(pathA == mrca)[0][0]])
        # Get path from MRCA to B
        path = path + list(pathB[np.where(pathB == mrca)[0][0] :])
        return path

    def subtree_reattach_to(self, node, target_clone, optimal_init=True):
        # Get the node and its parent root
        nodes = self._get_nodes(get_roots=True)
        roots = [node[1] for node in nodes]
        nodes = [node[0] for node in nodes]
        node_labels = np.array([node.label for node in nodes])
        if isinstance(node, str):
            nodeA_idx = np.where(np.array(node_labels) == node)[0][0]
        else:
            nodeA_idx = np.where(np.array(nodes) == node)[0][0]
        nodeA_parent_idx = np.where(np.array(nodes) == nodes[nodeA_idx].parent())[0][0]
        init_mean = nodes[nodeA_idx].node_mean
        init_cnvs = nodes[nodeA_idx].cnvs

        def descend(root):
            ns = [root["node"]]
            for child in root["children"]:
                child_node = descend(child)
                ns.extend(child_node)
            return ns

        nodes_below_nodeA = descend(roots[nodeA_idx])

        # Get the target clone subtree
        subtrees = self.get_subtrees(get_roots=True)
        subtrees_tssbs = [subtree[0] for subtree in subtrees]  # the objects
        subtrees = [subtree[1] for subtree in subtrees]  # the roots
        subtree_labels = np.array([subtree["node"].label for subtree in subtrees])
        if isinstance(target_clone, str):
            subtree_idx = np.where(np.array(subtree_labels) == target_clone)[0][0]
        else:
            subtree_idx = np.where(np.array(subtrees_tssbs) == target_clone)[0][0]
        target_subtree = subtrees[subtree_idx]
        subtreeA = subtrees[
            np.where(
                np.array([s["node"].label for s in subtrees])
                == nodes[nodeA_idx].tssb.label
            )[0][0]
        ]

        # Check if there is a pivot here
        pivot_changed = False
        for subtree_child in subtreeA["children"]:
            for n in nodes_below_nodeA:
                if subtree_child["pivot_node"] == n:
                    pivot_changed = True
                    subtree_child["node"].root["node"].set_parent(
                        subtreeA["node"].root["node"]
                    )
                    subtree_child["node"].root["node"].set_mean(variational=True)
                    subtree_child["pivot_node"] = subtreeA["node"].root["node"]
                    break

        # Move subtree
        roots[nodeA_idx]["node"].tssb = target_subtree["node"]
        roots[nodeA_idx]["node"].cnv = np.array(
            target_subtree["node"].root["node"].cnvs
        )
        roots[nodeA_idx]["node"].observed_parameters = np.array(
            target_subtree["node"].root["node"].observed_parameters
        )
        roots[nodeA_idx]["node"].set_parent(target_subtree["node"].root["node"])
        roots[nodeA_idx]["node"].set_mean(variational=True)
        for n in nodes_below_nodeA:
            n.tssb = target_subtree["node"]
            n.cnvs = np.array(target_subtree["node"].root["node"].cnvs)
            n.observed_parameters = np.array(
                target_subtree["node"].root["node"].observed_parameters
            )
        target_subtree["node"].root["children"].append(roots[nodeA_idx])
        target_subtree["node"].root["sticks"] = np.vstack(
            [target_subtree["node"].root["sticks"], 1.0]
        )

        childnodes = np.array([n["node"] for n in roots[nodeA_parent_idx]["children"]])
        tokeep = np.where(childnodes != roots[nodeA_idx]["node"])[0].astype(int).ravel()
        roots[nodeA_parent_idx]["sticks"] = roots[nodeA_parent_idx]["sticks"][tokeep]
        roots[nodeA_parent_idx]["children"] = list(
            np.array(roots[nodeA_parent_idx]["children"])[tokeep]
        )

        if optimal_init:
            for node in nodes_below_nodeA:
                node.variational_parameters["locals"]["unobserved_factors_mean"] = (
                    node.variational_parameters["locals"]["unobserved_factors_mean"]
                    - np.log(node.cnvs / 2)
                    + np.log(init_cnvs / 2)
                )

            parent_affected_genes = np.where(
                np.abs(
                    roots[nodeA_idx]["node"]
                    .parent()
                    .variational_parameters["locals"]["unobserved_factors_mean"]
                )
                > 0.5
            )[0]
            for node in nodes_below_nodeA:
                node.variational_parameters["locals"][
                    "unobserved_factors_kernel_log_shape"
                ][parent_affected_genes] = -1

            if roots[nodeA_idx]["node"].parent().parent() is not None:
                roots[nodeA_idx]["node"].variational_parameters["locals"][
                    "unobserved_factors_mean"
                ][parent_affected_genes] = (
                    roots[nodeA_idx]["node"]
                    .parent()
                    .variational_parameters["locals"]["unobserved_factors_mean"][
                        parent_affected_genes
                    ]
                )

            genes = np.where(roots[nodeA_idx]["node"].cnvs != 2)[0]
            roots[nodeA_idx]["node"].variational_parameters["locals"][
                "unobserved_factors_kernel_log_shape"
            ][genes] -= 1

            genes = np.where(init_cnvs != 2)[0]
            roots[nodeA_idx]["node"].variational_parameters["locals"][
                "unobserved_factors_kernel_log_shape"
            ][genes] -= 1

            # Need to accomodate events this node has that were previously inherited
            affected_genes = np.where(
                np.abs(
                    roots[nodeA_idx]["node"].variational_parameters["locals"][
                        "unobserved_factors_mean"
                    ]
                )
                > 0.5
            )[0]
            roots[nodeA_idx]["node"].variational_parameters["locals"][
                "unobserved_factors_kernel_log_shape"
            ][affected_genes] = -1

        # Reset variational parameters: all log_std and unobserved factors kernel
        # n_genes = target_subtree['node'].root['node'].cnvs.size
        # roots[nodeA_idx]['node'].variational_parameters['locals']['unobserved_factors_mean'] = np.array(target_subtree['node'].root['node'].variational_parameters['locals']['unobserved_factors_mean'])
        # roots[nodeA_idx]['node'].variational_parameters['locals']['unobserved_factors_log_std'] += .5#*np.ones((n_genes,))
        # roots[nodeA_idx]['node'].variational_parameters['locals']['unobserved_factors_kernel_log_mean'] += 1.#= np.log(roots[nodeA_idx]['node'].unobserved_factors_kernel_concentration_caller())*np.ones((n_genes,))
        # roots[nodeA_idx]['node'].variational_parameters['locals']['unobserved_factors_kernel_log_std'] += .5#*np.ones((n_genes,))

        # Set new unobserved factors to explain the same data as before (i.e. keep mean equal)
        # baseline = jnp.append(1, jnp.exp(self.root['node'].root['node'].log_baseline_caller()))
        # total_rna = jnp.sum(baseline * roots[nodeA_idx]['node'].cnvs/2 * jnp.exp(roots[nodeA_idx]['node'].unobserved_factors_mean))
        # roots[nodeA_idx]['node'].unobserved_factors_mean = np.log(init_mean * total_rna / (baseline * roots[nodeA_idx]['node'].cnvs/2))

        return pivot_changed

    def resample_pivots(self, verbose=False):
        """
        For each subtree, sample the pivot from the upstream tree with prob
        prob(pivot) * likelihod(subtree|pivot)
        where prob(pivot) is the probability of choosing that pivot
        and likelihood(subtree|pivot) is the likelihood of each node in the
        subtree given that the root node is parametererized by that pivot.

        Ideally, likelihod(subtree|pivot) would marginalize over all possible
        unobserved factors that spring from the new root. However, for a
        non-conjugate model, this would involve an expensive sampling loop
        to compute the probability of each pivot, so we just resort to Gibbs
        sampling each pivot.
        """

        def descend(super_tree):
            for child in super_tree["children"]:
                nodes, weights = super_tree["node"].get_fixed_weights()
                # Re-weight the node pivot probabilities by the likelihood
                # of the resulting subtree's root parameter
                reweights = [
                    child["node"]
                    .root["node"]
                    .unobserved_factors_ll(node.unobserved_factors)
                    + np.log(weights[i])
                    for i, node in enumerate(nodes)
                ]
                reweights = np.array(reweights)
                probs = np.exp(reweights - logsumexp(reweights))
                pivot_node = np.random.choice(nodes, p=probs)
                logger.debug(
                    f"Pivot of {child['node'].root['node'].label}: {child['pivot_node'].label}->{pivot_node.label}"
                )
                child["pivot_node"] = pivot_node
                if child["node"].root["node"].parent() != pivot_node:
                    child["node"].root["node"].set_parent(
                        pivot_node, reset=True
                    )  # update parameters

                descend(child)

        descend(self.root)

    def cull_subtrees(self, verbose=False, resample_sticks=True):
        culled = []

        def descend(super_tree):
            for child in super_tree["children"]:
                descend(child)
            culled.append(
                super_tree["node"].cull_tree(
                    verbose=verbose, resample_sticks=resample_sticks
                )
            )

        descend(self.root)
        return culled

    def cull_tree(self):
        """
        If a leaf node has no data assigned to it, remove it
        """

        def descend(root):
            counts = list(map(lambda child: descend(child), root["children"]))
            keep = len(trim_zeros(counts, "b"))

            for child in root["children"][keep:]:
                child["node"].kill()
                del child["node"]

            root["sticks"] = root["sticks"][:keep]
            root["children"] = root["children"][:keep]

            return (
                sum(counts)
                + root["node"].num_local_data()
                + (not root["node"].is_subcluster)
            )  # if it is a subcluster, returns 0.
            # i.e., all non-subclusters will return > 0

        descend(self.root)

    # ========= Functions to evaluate tree. =========

    def complete_data_log_likelihood(self):
        # Go to each subtree
        subtree_weights, subtrees = self.get_mixture()
        llhs = []
        for i, subtree in enumerate(subtrees):
            node_weights, nodes = subtree.get_mixture()
            for j, node in enumerate(nodes):
                if node.num_local_data():
                    llhs.append(
                        node.num_local_data()
                        * log(node_weights[j])
                        * log(subtree_weights[i])
                        + node.data_log_likelihood()
                    )
        llh = sum(array(llhs))

        return llh

    def complete_data_log_likelihood_nomix(self):
        weights, nodes = self.get_mixture()
        llhs = []
        lln = []
        for i, node in enumerate(nodes):
            if node.num_local_data():
                llhs.append(node.data_log_likelihood())
                lln.append(node.num_local_data() * log(weights[i]))
                # lln.append(weights[i])
        return (sum(array(lln)), sum(array(llhs)))

    def unnormalized_posterior_with_hypers(self):
        weights, nodes = self.get_mixture()
        llhs = []
        for i, node in enumerate(nodes):
            if node.num_local_data():
                llhs.append(
                    node.num_local_data() * log(weights[i])
                    + node.data_log_likelihood()
                    + node.parameter_log_prior()
                )
        llh = sum(array(llhs))

        def alpha_descend(root, depth=0):
            llh = (
                betapdfln(
                    root["main"], 1.0, (self.alpha_decay**depth) * self.dp_alpha
                )
                if self.min_depth <= depth
                else 0.0
            )
            for child in root["children"]:
                llh += alpha_descend(child, depth + 1)
            return llh

        weights_log_prob = alpha_descend(self.root)

        def gamma_descend(root):
            llh = 0
            for i, child in enumerate(root["children"]):
                llh += betapdfln(root["sticks"][i], 1.0, self.dp_gamma)
                llh += gamma_descend(child)
            return llh

        sticks_log_prob = gamma_descend(self.root)

        return llh + weights_log_prob + sticks_log_prob

    def unnormalized_posterior(self, verbose=False, compound=False):
        # Go to each subtree
        subtree_weights, subtrees = self.get_mixture()
        llhs = []
        for i, subtree in enumerate(subtrees):
            node_weights, nodes, roots, depths = subtree.get_mixture(
                get_roots=True, get_depths=True, truncate=True
            )
            for j, node in enumerate(nodes):
                if node.num_local_data():
                    llhs.append(
                        node.num_local_data()
                        * log(node_weights[j] * subtree_weights[i])
                        + node.data_log_likelihood()
                        + subtree.sticks_node_logprior(roots[j], depths[j])
                        + node.logprior()
                    )
                    logger.debug(
                        f"{node.label}: {node.num_local_data()*log(node_weights[j]*subtree_weights[i])}\t{node.data_log_likelihood()}\t{subtree.sticks_node_logprior(roots[j], depths[j])}\t{node.logprior()}"
                    )
        llh = sum(array(llhs))

        return llh

    def remove_empty_nodes(self):
        def descend(root):
            for index, child in enumerate(root["children"]):
                descend(child)
            root["node"].remove_empty_nodes()

        descend(self.root)

    # ========= Functions to update tree metadata. =========

    def label_nodes(self, counts=False, names=False):
        if not counts or names is True:
            self.label_nodes_names()
        elif not names or counts is True:
            self.label_nodes_counts()

    def set_node_names(self):
        def descend(root):
            root['node'].set_node_names(root_name=root['label'])
            for child in root["children"]:
                descend(child)

        descend(self.root)

    def set_tree_names(self, root_name="X"):
        self.root["label"] = str(root_name)
        self.root["node"].label = str(root_name)

        def descend(root, name):
            for i, child in enumerate(root["children"]):
                child_name = "%s-%d" % (name, i)

                root["children"][i]["label"] = child_name
                root["children"][i]["node"].label = child_name

                descend(child, child_name)

        descend(self.root, root_name)

    def tssb_dict2graphviz(
        self,
        g=None,
        counts=False,
        root_fillcolor=None,
        events=False,
        color_subclusters=False,
        show_labels=True,
        gene=None,
        genemode="raw",
        fontcolor="black",
        label_fontsize=24,
        size_fontsize=12,
    ):
        if g is None:
            g = Digraph()
            g.attr(fontcolor=fontcolor)

        if gene is not None:
            name_exp_dict = dict(
                [
                    (n.label, nodeavg[gene])
                    for n, nodeavg in zip(*self.get_avg_subtree_exp())
                ]
            )
            cmap = self.exp_cmap
            arr = np.array(list(name_exp_dict.values()))
            norm = matplotlib.colors.Normalize(vmin=min(arr), vmax=max(arr))
            if genemode == "observed":
                name_exp_dict = dict(
                    [
                        (n.label, nodeavg[gene])
                        for n, nodeavg in zip(*self.get_subtree_obs())
                    ]
                )
                cmap = self.cnv_cmap
                arr = np.array(list(name_exp_dict.values()))
                norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
            mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            name_color_dict = dict()
            for name in name_exp_dict:
                color = matplotlib.colors.to_hex(mapper.to_rgba(name_exp_dict[name]))
                name_color_dict[name] = color

        root_label = ""
        if show_labels:
            root_label = f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{self.root["label"].replace("-", "")}</B></FONT>'
        if counts:
            root_label = f'<FONT POINT-SIZE="{size_fontsize}" FACE="Arial">{str(self.root["node"].num_data())} cells</FONT>'
        if show_labels and counts:
            root_label = (
                f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{self.root["label"].replace("-", "")}</B></FONT>'
                + "<br/><br/>"
                + f'<FONT FACE="Arial">{str(self.root["node"].num_data())} cells</FONT>'
            )
        if events:
            root_label = self.root["node"].event_str
        if show_labels and events:
            root_label = (
                f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{self.root["label"].replace("-", "")}</B></FONT>'
                + "<br/><br/>"
                + self.root["node"].event_str
            )

        style = "filled"
        if root_fillcolor is None:
            root_fillcolor = self.root["node"].color
        if gene is not None:
            fillcolor = name_color_dict[str(self.root["label"])]
        g.node(
            str(self.root["label"]),
            "<" + str(root_label) + ">",
            fillcolor=root_fillcolor,
            style=style,
        )

        edge_color = "black"

        def descend(root, g):
            name = root["label"]
            for i, child in enumerate(root["children"]):
                child_name = child["label"]
                child_label = ""
                if show_labels:
                    child_label = f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{child_name.replace("-", "")}</B></FONT>'

                if counts:
                    child_label = f'<FONT POINT-SIZE="{size_fontsize}" FACE="Arial">{str(root["children"][i]["node"].num_data())} cells</FONT>'

                if show_labels and counts:
                    child_label = (
                        f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{child_name.replace("-", "")}</B></FONT>'
                        + "<br/><br/>"
                        + f'<FONT FACE="Arial">{str(root["children"][i]["node"].num_data())} cells</FONT>'
                    )

                if events:
                    child_label = root["children"][i]["node"].event_str

                if show_labels and events:
                    child_label = (
                        f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{child_name.replace("-", "")}</B></FONT>'
                        + "<br/><br/>"
                        + self.root["node"].event_str
                    )

                fillcolor = child["node"].color
                if gene is not None:
                    fillcolor = name_color_dict[str(child_name)]
                g.node(
                    str(child_name),
                    "<" + str(child_label) + ">",
                    fillcolor=fillcolor,
                    style=style,
                )

                g.edge(str(name), str(child_name), color=edge_color)

                g = descend(child, g)

            return g

        g = descend(self.root, g)
        return g

    def get_node_unobs(self):
        nodes = self.get_nodes()
        unobs = []
        estimated = (
            np.var(nodes[1].variational_parameters["kernel"]["state"]['mean'])
            != 0
        )
        if estimated:
            logger.debug("Getting the learned unobserved factors.")
        for node in nodes:
            unobs_factors = node.params[0]
            unobs.append(unobs_factors)
        return nodes, unobs

    def get_node_unobs_affected_genes(self):
        nodes = self.get_nodes()
        unobs = []
        estimated = (
            np.var(
                np.exp(nodes[1].variational_parameters["kernel"][
                    "direction"
                ]['log_alpha'])
            )
            != 0
        )
        if estimated:
            logger.debug("Getting the learned unobserved factors.")
        for node in nodes:
            unobs_factors = (
                node.unobserved_factors
                if not estimated
                else node.variational_parameters["locals"]["unobserved_factors_mean"]
            )
            unobs_factors_kernel = (
                node.unobserved_factors_kernel
                if not estimated
                else np.exp(node.variational_parameters["kernel"]['direction'][
                    "log_alpha"
                ] - node.variational_parameters["kernel"]['direction'][
                    "log_beta"
                ])
            )
            unobs.append(unobs_factors)
        return nodes, unobs

    def get_node_obs(self):
        nodes = self.get_nodes()
        obs = []
        for node in nodes:
            obs.append(node.observed_parameters)
        return nodes, obs

    def get_avg_node_exp(self, norm=True):
        nodes = self.get_nodes()
        data = self.data
        if norm:
            try:
                data = self.normalized_data
            except AttributeError:
                pass
        avgs = []
        for node in nodes:
            idx = np.array(list(node.data))
            if len(idx) > 0:
                avgs.append(np.mean(data[idx], axis=0))
            else:
                avgs.append(np.zeros(data.shape[1]))
        return nodes, avgs

    def get_avg_subtree_exp(self, norm=True):
        subtrees = self.get_subtrees()
        data = self.normalized_data if norm else self.data
        avgs = []
        for subtree in subtrees:
            _, nodes = subtree.get_mixture()
            idx = []
            for node in nodes:
                idx.append(list(node.data))
            idx = np.array(list(set(np.concatenate(idx))))
            avgs.append(np.mean(data[idx], axis=0))
        return subtrees, avgs

    def get_subtree_obs(self):
        subtrees = self.get_subtrees()
        obs = []
        for subtree in subtrees:
            obs.append(subtree.root["node"].cnvs)
        return subtrees, obs

    def initialize_gene_node_colormaps(
        self, node_obs=None, node_avg_exp=None, gene_specific=False, vmin=-0.5, vmax=0.5,
    ):
        nodes, vals = self.get_node_unobs()
        vals = np.array(vals)
        cmap = self.exp_cmap
        if gene_specific:
            mappers = []
            for gene in range(vals[0].shape[0]):
                gene_min, gene_max = np.nanmin(vals[:, gene]), np.nanmax(vals[:, gene])
                norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
                mappers.append(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
            self.gene_node_colormaps["unobserved"] = dict()
            self.gene_node_colormaps["unobserved"]["vals"] = dict(
                zip([node.label for node in nodes], vals)
            )
            self.gene_node_colormaps["unobserved"]["mapper"] = mappers
        else:
            global_min, global_max = np.nanmin(vals), np.nanmax(vals)
            cmap = self.exp_cmap
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            self.gene_node_colormaps["unobserved"] = dict()
            self.gene_node_colormaps["unobserved"]["vals"] = dict(
                zip([node.label for node in nodes], vals)
            )
            self.gene_node_colormaps["unobserved"]["mapper"] = mapper

        if node_obs:
            nodes_labels = list(node_obs.keys())
            vals = list(node_obs.values())
        else:
            nodes, vals = self.get_node_obs()
            nodes_labels = [node.label for node in nodes]
        cmap = self.obs_cmap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=cmap.N - 1)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        self.gene_node_colormaps["observed"] = dict()
        self.gene_node_colormaps["observed"]["vals"] = dict(zip(nodes_labels, vals))
        self.gene_node_colormaps["observed"]["mapper"] = mapper

        if node_avg_exp:
            nodes_labels = list(node_avg_exp.keys())
            vals = list(node_avg_exp.values())
        else:
            nodes, vals = self.get_avg_node_exp()
            nodes_labels = [node.label for node in nodes]
        vals = np.array(vals)
        cmap = self.exp_cmap
        if gene_specific:
            mappers = []
            for gene in range(vals[0].shape[0]):
                gene_min, gene_max = np.nanmin(vals[:, gene]), np.nanmax(vals[:, gene])
                norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)
                mappers.append(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
            self.gene_node_colormaps["avg"] = dict()
            self.gene_node_colormaps["avg"]["vals"] = dict(zip(nodes_labels, vals))
            self.gene_node_colormaps["avg"]["mapper"] = mappers
        else:
            global_min, global_max = np.nanmin(vals), np.nanmax(vals)
            norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)
            mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            self.gene_node_colormaps["avg"] = dict()
            self.gene_node_colormaps["avg"]["vals"] = dict(zip(nodes_labels, vals))
            self.gene_node_colormaps["avg"]["mapper"] = mapper

        logger.debug(
            f"Created `self.gene_node_colormaps` with keys {list(self.gene_node_colormaps.keys())}"
        )

    def plot_tree(
        self,
        super_only=False,
        counts=False,
        root_fillcolor=None,
        events=False,
        color_subclusters=False,
        reset_names=True,
        ordered=False,
        genemode="avg",
        show_labels=True,
        color_by_weight=False,
        gene=None,
        fontcolor="black",
        pivot_probabilities=None,
        node_color_dict=None,
        show_root=False,
        label_fontsize=24,
        size_fontsize=12,
    ):

        if node_color_dict is None:
            if gene is not None:
                if len(self.gene_node_colormaps.keys()) == 0:
                    self.initialize_gene_node_colormaps()

                vals = self.gene_node_colormaps[genemode]["vals"]
                mapper = self.gene_node_colormaps[genemode]["mapper"]
                if isinstance(mapper, list):
                    mapper = mapper[gene]
                node_color_dict = dict()
                for name in vals:
                    color = (
                        matplotlib.colors.to_hex(mapper.to_rgba(vals[name][gene]))
                        if not np.isnan(vals[name][gene])
                        else "gray"
                    )
                    node_color_dict[name] = color

        if super_only:
            g = self.tssb_dict2graphviz(
                counts=counts,
                root_fillcolor=root_fillcolor,
                events=events,
                show_labels=show_labels,
                gene=gene,
                genemode=genemode,
                fontcolor=fontcolor,
                label_fontsize=label_fontsize,
                size_fontsize=size_fontsize,
            )
        else:
            g = self.root["node"].plot_tree(
                counts=counts,
                reset_names=True,
                root_fillcolor=root_fillcolor,
                events=events,
                show_labels=show_labels,
                gene=gene,
                genemode=genemode,
                fontcolor=fontcolor,
                node_color_dict=node_color_dict,
            )

            def descend(root, g):
                for i, child in enumerate(root["children"]):
                    g = child["node"].plot_tree(
                        g,
                        reset_names=True,
                        counts=counts,
                        root_fillcolor=root_fillcolor,
                        show_labels=show_labels,
                        gene=gene,
                        genemode=genemode,
                        fontcolor=fontcolor,
                        events=events,
                        node_color_dict=node_color_dict,
                        label_fontsize=label_fontsize,
                        size_fontsize=size_fontsize,
                    )
                    if counts:
                        lab = str(child["pivot_node"].num_local_data())
                        if show_labels:
                            lab = (
                                "<"
                                + f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{child["pivot_node"].label.replace("-", "")}</B></FONT>'
                                + "<br/><br/>"
                                + f'<FONT FACE="Arial">{lab} cells</FONT>'
                                + ">"
                            )
                        g.node(child["pivot_node"].label, lab)
                    elif events:
                        lab = child["pivot_node"].event_str
                        if show_labels:
                            lab = (
                                f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{child["pivot_node"].label.replace("-", "")}</B></FONT>'
                                + "<br/><br/>"
                                + lab
                            )
                        g.node(child["pivot_node"].label, "<" + lab + ">")
                    if pivot_probabilities is not None:
                        if child["node"].root["label"] in pivot_probabilities:
                            for pivot in pivot_probabilities[
                                child["node"].root["label"]
                            ]:
                                prob = pivot_probabilities[child["node"].root["label"]][
                                    pivot
                                ]
                                weight = prob * 4.0
                                weight = np.max([0.1, weight])
                                prob_str = " " + f"{prob:0.2g}".lstrip("0")
                                arrowsize = np.min([1, weight])
                                g.edge(
                                    pivot,
                                    child["node"].root["label"],
                                    penwidth=str(weight),
                                    arrowsize=str(arrowsize),
                                    label=prob_str,
                                    color=child["node"].color,
                                )
                        else:
                            g.edge(
                                child["pivot_node"].label, child["node"].root["label"]
                            )
                    else:
                        g.edge(child["pivot_node"].label, child["node"].root["label"])
                    g = descend(child, g)
                return g

            g = descend(self.root, g)

            if show_root:
                if self.root["label"] != "root":
                    style = None
                    fillcolor = self.input_tree_dict["root"]["color"]
                    g.node(
                        "root",
                        f'<<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>root</B></FONT>>',
                        fillcolor=fillcolor,
                        style="filled",
                    )
                    g.edge("root", self.root["label"])

        return g

    def label_assignments(self, root_name="X", reset_names=True):
        if reset_names:
            self.set_node_names(root_name=root_name)
        else:
            self.set_subcluster_node_names()

        assignments = np.array([str(a)[-12:-1] for a in self.assignments])

        # Change root
        for n in range(len(assignments)):
            if self.assignments[n].parent() is None:
                assignments[n] = root_name

        def descend(root, name):
            for i, child in enumerate(root["children"]):
                child_name = child["label"]

                # Get occurrences of node in assignment list
                idx = np.where(assignments == (str(child["node"])[-12:-1]))[0]

                # and replace with name
                if len(idx) > 0:
                    assignments[idx] = child_name

                descend(child, child_name)

        descend(self.root, root_name)
        return assignments

    def set_node_event_strings(self, **kwargs):
        def descend(node):
            node.set_event_string(**kwargs)
            for child in list(node.children()):
                descend(child)

        descend(self.root["node"].root["node"])

    # ========= Methods to compute cell-cell distances in the tree. =========

    def create_augmented_tree_dict(self):
        self.node_dict = dict()

        def descend(node):
            self.node_dict[node.label] = dict()
            self.node_dict[node.label]["node"] = node
            if node.parent() is not None:
                self.node_dict[node.label]["parent"] = node.parent().label
            else:
                self.node_dict[node.label]["parent"] = "NULL"
            for child in list(node.children()):
                descend(child)

        descend(self.root["node"].root["node"])

    def get_distance(self, node1, node2):
        path = self.path_between_nodes(node1, node2)
        path_labels = [n.label for n in path]
        node_dict = dict(zip(path_labels, path))
        dist = 0
        prev_node = path_labels[0]
        for node in path_labels:
            if node != prev_node:
                dist += np.sqrt(
                    np.sum(
                        (node_dict[node].get_mean() - node_dict[prev_node].get_mean())** 2
                    )
                )
                prev_node = node

        return dist     

    def get_pairwise_obs_distances(self):
        n_cells = len(self.assignments)
        mat = np.zeros((n_cells, n_cells))
        nodes = self.get_nodes()

        for node1 in nodes:
            idx1 = np.where(self.assignments == node1)[0]
            if len(idx1) == 0:
                continue
            for node2 in nodes:
                idx2 = np.where(self.assignments == node2)[0]
                if len(idx2) == 0:
                    continue
                mat[np.meshgrid(idx1,idx2)] = self.get_distance(node1, node2)

        return mat

    def set_combined_params(self):
        def sub_descend(subroot, obs_param):
            for i, child in enumerate(subroot['children']):
                child['combined_param'] = child['node'].combine_params(child['param'], obs_param)
                sub_descend(child, obs_param)

        def descend(root):
            sub_descend(root['node'], root['obs_param'])
            for i, child in enumerate(root['children']):
                descend(child)

        descend(self.root)

    def set_ntssb_colors(self, **cmap_kwargs):
        # Traverse tree to update dict
        def descend(root):
            tree_colors.make_tree_colormap(root['node'].root, root['color'], **cmap_kwargs)
            for i, child in enumerate(root['children']):
                descend(child)
        descend(self.root)        

    def show_tree(self, **kwargs):
        self.set_learned_parameters()
        self.set_node_names()
        self.set_expected_weights()
        self.assign_samples()
        self.set_ntssb_colors()
        tree = self.get_param_dict()
        plt.figure(figsize=(4,4))
        ax = plt.gca()
        plot_full_tree(tree, ax=ax, node_size=101, **kwargs)