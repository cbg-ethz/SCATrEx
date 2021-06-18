"""
This module contains the NestedTSSB class.
"""

from functools import partial
from copy import deepcopy

from graphviz import Digraph
import matplotlib
import matplotlib.cm

import numpy as np
from numpy import *

import jax
from jax.api import jit, grad, vmap
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp
import jax.nn as jnn

from ..util import *
from .tssb import TSSB


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

    min_dp_alpha    = 0.001
    max_dp_alpha    = 10.0
    min_dp_gamma    = 0.001
    max_dp_gamma    = 10.0
    min_alpha_decay = 0.001
    max_alpha_decay = 0.80

    def __init__(self, input_tree, node_constructor, dp_alpha=1.0, dp_gamma=1.0, alpha_decay=1.0,
                        min_depth=0, max_depth=15, fixed_weights_pivot_sampling=True, use_weights=True,
                        node_hyperparams=dict()):
        if input_tree is None:
            raise Exception("Input tree must be specified.")

        self.min_depth   = min_depth
        self.max_depth   = max_depth
        self.dp_alpha    = dp_alpha # smaller dp_alpha => larger nu => less nodes
        self.dp_gamma    = dp_gamma # smaller dp_gamma => larger psi => less nodes
        self.alpha_decay = alpha_decay

        self.input_tree  = input_tree
        self.input_tree_dict = self.input_tree.tree_dict
        self.node_constructor = node_constructor

        self.fixed_weights_pivot_sampling = fixed_weights_pivot_sampling

        self.assignments = []

        self.elbo = -np.inf
        self.data = None
        self.num_data = None

        self.opt_init = None
        self.get_params = None
        self.opt_update = None
        self.opt_state = None
        self.max_nodes = len(self.input_tree_dict.keys()) * 1 # upper bound on number of nodes
        self.n_nodes = len(self.input_tree_dict.keys())

        self.cnv_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cmap", ["#2040C8", "white", "#EE241D"])
        self.exp_cmap = matplotlib.cm.viridis

        self.reset_tree(use_weights=use_weights, node_hyperparams=node_hyperparams)

    # ========= Functions to initialize tree. =========
    def reset_tree(self, use_weights=False, node_hyperparams=dict()):
        if use_weights and "weight" not in self.input_tree_dict['A'].keys():
            raise KeyError("No weights were specified in the input tree.")

        # Clear tree
        self.assignments = []

        input_tree_dict = self.input_tree_dict

        obj = self.node_constructor(True, input_tree_dict['A']['params'], parent=None, label='A', **node_hyperparams)
        input_tree_dict['A']['subtree'] = TSSB(obj, 'A', ntssb=self, dp_alpha=input_tree_dict['A']['dp_alpha_subtree'],
                                                             alpha_decay=input_tree_dict['A']['alpha_decay_subtree'],
                                                             dp_gamma=input_tree_dict['A']['dp_gamma_subtree'],
                                                             color=input_tree_dict['A']['color'])

        main = boundbeta(1.0, self.dp_alpha) if self.min_depth == 0 else 0.0 # if min_depth > 0, no data can be added to the root (main stick is nu)
        if use_weights:
            main = self.input_tree_dict['A']['weight']
            input_tree_dict['A']['subtree'].weight = self.input_tree_dict['A']['weight']

        self.root = {    'node'     : input_tree_dict['A']['subtree'],
                         'main'     : main,
                         'sticks'   : empty((0,1)), # psi sticks
                         'children' : [],
                         'label'    : 'A',
                         'super_parent'   :  None,
                         'parent'   : None    }

        # Recursively construct tree of subtrees
        def descend(super_tree, label, depth=0):
            for i, child in enumerate(input_tree_dict[label]['children']):

                stick = boundbeta(1, self.dp_gamma)
                if use_weights:
                    stick = self.input_tree.get_sum_weights_subtree(child)
                    if i < len(input_tree_dict[label]['children']) - 1:
                        sum = 0
                        for j, c in enumerate(input_tree_dict[label]['children'][i:]):
                            sum = sum + self.input_tree.get_sum_weights_subtree(c)
                        stick = stick / sum
                    else:
                        stick = 1.

                super_tree['sticks'] = vstack([ super_tree['sticks'], stick if i < len(input_tree_dict[label]['children']) - 1 else 1.])

                main = boundbeta(1.0, (self.alpha_decay**(depth+1))*self.dp_alpha)
                if use_weights:
                    main = self.input_tree_dict[child]['weight']
                    subtree_weights_sum = self.input_tree.get_sum_weights_subtree(child)
                    main = main / subtree_weights_sum

                if len(input_tree_dict[child]['children']) < 1:
                    main = 1. # stop at leaf node

                pivot_tssb = input_tree_dict[label]['subtree']
                # pivot_tssb.dp_alpha = input_tree_dict[child]['dp_alpha_parent_edge']
                # pivot_tssb.alpha_decay = input_tree_dict[child]['alpha_decay_parent_edge']
                # pivot_tssb.truncate()

                # pivot_tssb.eta = input_tree_dict[child]['eta']
                pivot_node = super_tree['node'].root['node']

                obj = self.node_constructor(True, input_tree_dict[child]['params'], parent=pivot_node, label=child)

                input_tree_dict[child]['subtree'] = TSSB(obj, child, ntssb=self, dp_alpha=input_tree_dict[child]['dp_alpha_subtree'],
                                                                     alpha_decay=input_tree_dict[child]['alpha_decay_subtree'],
                                                                     dp_gamma=input_tree_dict[child]['dp_gamma_subtree'],
                                                                     color=input_tree_dict[child]['color']
                                                                     )
                input_tree_dict[child]['subtree'].eta = input_tree_dict[child]['eta']

                if use_weights:
                    input_tree_dict[child]['subtree'].weight = self.input_tree_dict[child]['weight']

                super_tree['children'].append({  'node'         : input_tree_dict[child]['subtree'],
                                                 'main'         : main if self.min_depth <= (depth+1) else 0.0,
                                                 'sticks'       : empty((0,1)), # psi sticks
                                                 'children'     : [],
                                                 'label'        : child,
                                                 'super_parent' : super_tree['node'],
                                                 'pivot_node'   : pivot_node,
                                                 'pivot_tssb'   : pivot_tssb  })

                descend(super_tree['children'][-1], child, depth+1)
        descend(self.root, 'A')

    def reset_variational_parameters(self, **kwargs):
        # Reset node parameters
        def descend(super_tree):
            super_tree['node'].reset_node_variational_parameters(**kwargs)
            for child in super_tree['children']:
                descend(child)
        descend(self.root)

    def reset_node_parameters(self, root_params=True, down_params=True, node_hyperparams=None):
        # Reset node parameters
        def descend(super_tree):
            super_tree['node'].reset_node_variational_parameters()
            super_tree['node'].reset_node_parameters(root_params=root_params, down_params=down_params, node_hyperparams=node_hyperparams)
            for child in super_tree['children']:
                descend(child)
        descend(self.root)

    def sync_subtrees(self):
        subtrees = self.get_subtrees()
        for subtree in subtrees:
            subtree.ntssb = self

    def put_data_in_nodes(self, num_data, eta=0):
        self.assignments = []
        subtrees = self.get_subtrees()
        for tssb in subtrees:
            tssb.assignments = []

        self.num_data = num_data

        # Get mixture weights
        nodes, weights = self.get_node_weights(eta=eta)

        for node in nodes:
            node.remove_data()

        for n in range(self.num_data):
            node = np.random.choice(nodes)
            node.tssb.assignments.append(node)
            node.add_datum(n)
            self.assignments.append(node)

    def normalize_data(self):
        if self.data is None:
            raise Exception("Need to `call add_data(self, data, to_root=False)` first.")

        self.normalized_data = np.log(10000 * self.data / np.sum(self.data, axis=1).reshape(self.num_data, 1) + 1)

    def add_data(self, data, to_root=False):
        self.data        = data
        self.num_data    = 0 if data is None else data.shape[0]
        print(f'Adding data of shape {data.shape} to NTSSB')

        self.assignments = []

        for n in range(self.num_data):
            if to_root:
                subtree = self.root['node']
                node = self.root['node'].root['node']
            else:
                u = rand()
                subtree, _, u = self.find_node(u)

                # Now choose the node
                node, _, _ = subtree.find_node(u)

            subtree.assignments.append(node)
            node.add_datum(n)
            self.assignments.append(node)

        try:
            # Reset root node parameters to set data-dependent variables if applicable
            self.root['node'].root['node'].reset_data_parameters()
        except AttributeError:
            pass

    def clear_data(self):
        def descend(root):
            for index, child in enumerate(root['children']):
                descend(child)
            root['node'].clear_data()
        self.root['node'].clear_data()
        descend(self.root)
        self.assignments = []

    def truncate_subtrees(self):
        def descend(root):
            for index, child in enumerate(root['children']):
                descend(child)
            root['node'].truncate()
        descend(self.root)

    def set_pivot_dp_alphas(self, tree_dict):
        def descend(super_tree, label, depth=0):
            for i, child in enumerate(tree_dict[label]['children']):
                super_tree['children'][i]['pivot_tssb'].dp_alpha = tree_dict[child]['dp_alpha_parent_edge']
                descend(super_tree['children'][i], child, depth+1)
        descend(self.root, 'A')

    def create_new_tree(self, n_extra_per_observed=1, num_data=None):
        # Clear current tree (including subtrees)
        self.reset_tree(True, node_hyperparams=self.root['node'].root['node'].node_hyperparams)
        self.reset_node_parameters(node_hyperparams=self.root['node'].root['node'].node_hyperparams)
        self.plot_tree(super_only=False) # update names

        # Add nodes to subtrees
        subtrees = self.get_subtrees()
        for subtree in subtrees:
            n_nodes = 0
            while n_nodes < n_extra_per_observed:
                _, nodes = subtree.get_mixture()
                # Uniformly choose a node from the subtree
                snode = np.random.choice(nodes)
                self.add_node_to(snode.label, optimal_init=False)
                self.plot_tree(super_only=False) # update names
                n_nodes = n_nodes + 1

        # Choose pivots
        def descend(super_tree):
            for child in super_tree['children']:
                weights, nodes = super_tree['node'].get_fixed_weights(eta=child['node'].eta)
                pivot_node = np.random.choice(nodes, p=weights)
                child['pivot_node'] = pivot_node
                child['node'].root['node'].set_parent(pivot_node)

                descend(child)
        descend(self.root)

        if num_data is not None:
            self.put_data_in_nodes(num_data, eta=0)

    def sample_new_tree(self, num_data, use_weights=False):
        self.num_data = num_data

        # Clear current tree (including subtrees)
        self.reset_tree(use_weights, node_hyperparams=self.root['node'].root['node'].node_hyperparams)

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
            for child in super_tree['children']:
                child['node'].cull_tree()
                descend(child)
        self.root['node'].cull_tree()
        descend(self.root)

        # And now recursively sample the pivots
        def descend(super_tree):
            for child in super_tree['children']:
                # We have new subtrees, so update the pivot trees before
                # choosing pivot nodes
                weights, nodes = super_tree['node'].get_fixed_weights(eta=child['node'].eta)
                pivot_node = np.random.choice(nodes, p=weights)
                child['pivot_node'] = pivot_node
                child['node'].root['node'].set_parent(pivot_node)

                descend(child)
        descend(self.root)

    # ========= Functions to sample tree parameters. =========
    def sample_pivot_main_sticks(self):
        def descend(super_tree):
            for child in super_tree['children']:
                child['pivot_tssb'].sample_main_sticks(truncate=True)
                descend(child)
        descend(self.root)

    def sample_pivot_sticks(self, balanced=True):
        def descend(super_tree):
            for child in super_tree['children']:
                child['pivot_tssb'].sample_sticks(truncate=True, balanced=balanced)
                descend(child)
        descend(self.root)

    def sample_pivots(self):
        def descend(super_tree):
            for child in super_tree['children']:
                nodes, weights = super_tree['node'].get_fixed_weights()
                pivot_node = np.random.choice(nodes, p=weights)
                child['pivot_node'] = pivot_node
                descend(child)
        descend(self.root)

    # ========= Functions to access tree features. =========
    def get_pivot_tree(self, subtree_label):
        def descend(root):
            if root['label'] == subtree_label:
                return root['pivot_node'], root['pivot_tssb']
            for child in root['children']:
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
            raise Exception(f'Did not find pivot of node {subtree_label}')

        def descend(root, depth=1):
            if root['node'] == pivot_node:
                return depth
            for child in root['children']:
                n = descend(child, depth + 1)
                if n is not None:
                    return n

        depth = descend(pivot_tssb.root)

        # Depth of parent subtree
        total_depth = pivot_tssb.get_n_levels()

        return depth/total_depth

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

    def get_mixture(self, reset_names=False):
        if reset_names:
            self.set_node_names()

        def descend(root, mass):
            weight  = [ mass * root['main'] ]
            subtree = [ root['node'] ]
            edges   = sticks_to_edges(root['sticks'])
            weights = diff(hstack([0.0, edges]))

            for i, child in enumerate(root['children']):
                (child_weights, child_subtrees) = descend(child, mass*(1.0-root['main'])*weights[i])
                weight.extend(child_weights)
                subtree.extend(child_subtrees)
            return (weight, subtree)
        return descend(self.root, 1.0)

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
            subtree = [ root['node'] ]
            roots   = [root]

            for i, child in enumerate(root['children']):
                if get_roots:
                    (child_subtrees, child_roots) = descend(child)
                    roots.extend(child_roots)
                else:
                    child_subtrees = descend(child)
                subtree.extend(child_subtrees)
            if get_roots:
                return (subtree, roots)
            return subtree
        out =  descend(self.root)
        if get_roots:
            return list(zip(out[0], out[1]))
        else:
            return out

    def get_subtree_leaves(self):
        def descend(root, l):
            for i, child in enumerate(root['children']):
                descend(child, l)
            if len(root['children']) == 0:
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

    def get_nodes(self, root_node=None, parent_vector=False):
        def descend(root, idx=0, prev_idx=-1):
            idx = idx + 1
            node = [root]
            parent_idx = [prev_idx]
            parent_idx = [prev_idx]
            prev_idx = idx
            for i, child in enumerate(list(root.children())):
                nodes, idx, parents_idx = descend(child, idx, prev_idx-1)
                node.extend(nodes)
                parent_idx.extend(parents_idx)
            return node, idx, parent_idx
        if root_node is None:
            root_node = self.root['node'].root['node']
        node_list, _, parent_list = descend(root_node)
        if parent_vector:
            return node_list, parent_list
        else:
            return node_list

    def get_width_distribution(self):
        def descend(root, depth, width_vec):
            width_vec[depth-1] = width_vec[depth-1] + 1
            if len(root['children']) == 0:
                return width_vec
            for i, child in enumerate(root['children']):
                descend(child, depth+1, width_vec)
            return width_vec
        width_vec = zeros(self.max_depth)
        return descend(self.root, 1, width_vec)

    def get_weight_distribtuion(self):
        def descend(root, mass, depth, mass_vec):
            edges   = sticks_to_edges(root['sticks'])
            weights = diff(hstack([0.0, edges]))
            for i, child in enumerate(root['children']):
                mass_vec[depth] = mass_vec[depth] + mass * weights[i] * child['main']
                mass_vec = descend(child, mass*(1.0-child['main'])*weights[i],depth+1,mass_vec)
            return mass_vec
        mass_vec = zeros(self.max_depth)
        edges   = sticks_to_edges(self.root['sticks'])
        weights = diff(hstack([0.0, edges]))
        if len(weights) > 0 :
            mass_vec[0] = weights[0] * self.root['main']
            return descend(self.root, 1.0-mass_vec[0], 1, mass_vec)
        else:
            return mass_vec

    def deepest_node_depth(self):
        def descend(root, depth):
            if len(root['children']) == 0:
                return depth
            deepest = depth
            for i, child in enumerate(root['children']):
                hdepth = descend(child, depth+1)
                if deepest < hdepth:
                    deepest = hdepth
            return deepest
        return descend(self.root, 1)

    # Represent TSSB tree with node clones ordered by label, assuming they have one
    def get_clone_sorted_root_dict(self):
        ordered_dict = self.root.copy()

        sorted_idxs = []
        labels = [int(node['label']) for node in self.root['children']]
        for i in range(len(self.root['children'])):
            min_idx = sorted_idxs[i-1] if i > 0 else 0
            for j in range(len(self.root['children'])):
                if labels[j] < labels[min_idx] and j not in sorted_idxs:
                    min_idx = j
            sorted_idxs.append(min_idx)
            labels[min_idx] = np.inf

        new_children = []
        for k in sorted_idxs:
            child = ordered_dict['children'][k]
            new_children.append(child)
        ordered_dict['children'] = new_children

        return ordered_dict

    def find_node(self, u):
            """This function breaks sticks in a tree where each node is a subtree.
            """
            def descend(root, u, depth=0):
                if depth >= self.max_depth:
                    #print >>sys.stderr, "WARNING: Reached maximum depth."
                    return (root['node'], [], u)
                elif u < root['main']:
                    return (root['node'], [], u / root['main'])
                else:
                    # Rescale the uniform variate to the remaining interval.
                    u = (u - root['main']) / (1.0 - root['main'])

                    # Don't need to break sticks

                    edges = 1.0 - cumprod(1.0 - root['sticks'])
                    index = sum(u > edges)
                    edges = hstack([0.0, edges])
                    u     = (u - edges[index]) / (edges[index+1] - edges[index])

                    (node, path, u_out) = descend(root['children'][index], u, depth+1)

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

    def get_node_mean(self, log_baseline, unobserved_factors, noise, cnvs):
        node_mean = jnp.exp(log_baseline + unobserved_factors + noise + jnp.log(cnvs/2))
        sum = jnp.sum(node_mean, axis=1).reshape(self.num_data, 1)
        node_mean = node_mean / sum
        return node_mean

    def get_tssb_indices(self, nodes, tssbs):
        max_len = self.max_nodes
        tssb_indices = []
        for node in nodes:
            tssb_indices.append(jnp.array([i for i, tssb in enumerate(tssbs) if tssb == node.tssb.label]))
            # if len(tssb_indices[-1].shape[0]) > max_len:
            #     max_len = len(tssb_indices[-1].shape[0])

        for i, c in enumerate(tssb_indices):
            l = c.shape[0]
            if l < max_len:
                c = jnp.concatenate([c, jnp.array([-1]*(max_len-l))])
                tssb_indices[i] = c
        tssb_indices = jnp.array(tssb_indices).astype(int)
        return tssb_indices

    def get_below_root(self, root_idx, children_vector, tssbs=None):
        def descend(idx):
            below_root = [idx]
            for child_idx in children_vector[idx]:
                if child_idx > 0:
                    if tssbs is not None:
                        if tssbs[child_idx] == tssbs[root_idx]:
                            aux = descend(child_idx)
                            below_root.extend(aux)
                    else:
                        aux = descend(child_idx)
                        below_root.extend(aux)
            return below_root
        return np.array(descend(root_idx))

    def get_children_vector(self, nodes, parent_vector):
        children_vector = []
        max_len = self.max_nodes
        for i, node in enumerate(nodes):
            children_vector.append(jnp.where(parent_vector == i)[0])
            # if children_vector[-1].shape[0] > max_len:
            #     max_len = children_vector[-1].shape[0]

        for i, c in enumerate(children_vector):
            l = c.shape[0]
            if l < max_len:
                c = jnp.concatenate([c, jnp.array([-1]*(max_len-l))])
                children_vector[i] = c
        children_vector = jnp.array(children_vector).astype(int)
        return children_vector

    def get_ancestor_indices(self, nodes, parent_vector, inclusive=False):
        ancestor_indices = []
        max_len = self.max_nodes
        for i in range(len(nodes)):
            # get ancestor nodes in the same subtree
            p = i
            indices = []
            while p != -1 and nodes[p].tssb == nodes[i].tssb:
                if not(not inclusive and p == i):
                    indices.append(p)
                p = parent_vector[p]

            indices = jnp.array(indices)
            ancestor_indices.append(indices)
            # if len(indices) > max_len:
            #     max_len = len(indices)

        for i, c in enumerate(ancestor_indices):
            l = c.shape[0]
            if l < max_len:
                c = jnp.concatenate([c, jnp.array([-1]*(max_len-l))])
                ancestor_indices[i] = c
        ancestor_indices = jnp.array(ancestor_indices).astype(int)
        return ancestor_indices

    def get_previous_branches_indices(self, nodes):
        previous_branches_indices = []
        max_len = self.max_nodes
        for node in nodes:
            indices = []
            if not node.is_observed:
                for j, prev_child in enumerate(list(node.parent().children())):
                    if prev_child.is_observed:
                        continue
                    if prev_child == node:
                        break
                    # Locate prev_child in nodes list
                    for idx, n_ in enumerate(nodes):
                        if n_ == prev_child:
                            indices.append(idx)
                            break
            previous_branches_indices.append(jnp.array(indices))
            # if len(indices) > max_len:
            #     max_len = len(indices)

        for i, c in enumerate(previous_branches_indices):
            l = c.shape[0]
            if l < max_len:
                c = jnp.concatenate([c, jnp.array([-1]*(max_len-l))])
                previous_branches_indices[i] = c
        previous_branches_indices = jnp.array(previous_branches_indices).astype(int)
        return previous_branches_indices

    def Eq_log_p_nu(self, dp_alpha, nu_sticks_alpha, nu_sticks_beta):
        l = 0
        aux = (digamma(nu_sticks_beta) - digamma(nu_sticks_alpha+nu_sticks_beta))
        l = l + (dp_alpha-1)*aux - betaln(1, dp_alpha)
        return l

    def Eq_log_q_nu(self, nu_sticks_alpha, nu_sticks_beta):
        l = 0
        aux = digamma(nu_sticks_alpha+nu_sticks_beta)
        aux1 = digamma(nu_sticks_alpha) - aux
        aux2 = digamma(nu_sticks_beta) - aux
        l = l + (nu_sticks_alpha-1)*aux1 + (nu_sticks_beta-1)*aux2 - betaln(nu_sticks_alpha, nu_sticks_beta)
        return l

    def Eq_log_p_psi(self, dp_gamma, psi_sticks_alpha, psi_sticks_beta):
        l = 0
        aux = (digamma(psi_sticks_beta) - digamma(psi_sticks_alpha+psi_sticks_beta))
        l = l + (dp_gamma-1)*aux - betaln(1, dp_gamma)
        return l

    def Eq_log_q_psi(self, psi_sticks_alpha, psi_sticks_beta):
        l = 0
        aux = digamma(psi_sticks_alpha+psi_sticks_beta)
        aux1 = digamma(psi_sticks_alpha) - aux
        aux2 = digamma(psi_sticks_beta) - aux
        l = l + (psi_sticks_alpha-1)*aux1 + (psi_sticks_beta-1)*aux2 - betaln(psi_sticks_alpha, psi_sticks_beta)
        return l

    def Eq_log_p_tau(self, tau_alpha, tau_beta):
        l = 0
        aux = (digamma(tau_beta) - digamma(tau_alpha+tau_beta))
        l = l + (2-1)*aux - betaln(1, 2)
        return l

    def Eq_log_q_tau(self, tau_alpha, tau_beta):
        l = 0
        aux = digamma(tau_alpha+tau_beta)
        aux1 = digamma(tau_alpha) - aux
        aux2 = digamma(tau_beta) - aux
        l = l + (tau_alpha-1)*aux1 + (tau_beta-1)*aux2 - betaln(tau_alpha, tau_beta)
        return l

    def tssb_log_priors(self):
        nodes, parent_vector = self.get_nodes(root_node=None, parent_vector=True)
        tssb_weights = jnp.array([node.tssb.weight for node in nodes])
        init_nu_log_alphas = jnp.array([node.nu_log_alpha for node in nodes])
        init_nu_log_betas = jnp.array([node.nu_log_beta for node in nodes])
        init_psi_log_alphas = jnp.array([node.psi_log_alpha for node in nodes])
        init_psi_log_betas = jnp.array([node.psi_log_beta for node in nodes])
        ancestor_nodes_indices = self.get_ancestor_indices(nodes, parent_vector)
        previous_branches_indices = self.get_previous_branches_indices(nodes)

        rem = self.max_nodes - len(nodes)
        init_psi_log_betas = jnp.concatenate([init_psi_log_betas, -1*jnp.ones((rem,))])
        init_psi_log_alphas = jnp.concatenate([init_psi_log_alphas, -1*jnp.ones((rem,))])
        init_nu_log_betas = jnp.concatenate([init_nu_log_betas, -1*jnp.ones((rem,))])
        init_nu_log_alphas = jnp.concatenate([init_nu_log_alphas, -1*jnp.ones((rem,))])
        tssb_weights = jnp.concatenate([tssb_weights, 10*jnp.ones((rem,))])
        previous_branches_indices = jnp.concatenate([previous_branches_indices, -1*jnp.ones((rem, previous_branches_indices.shape[1]))], axis=0).astype(int)
        ancestor_nodes_indices = jnp.concatenate([ancestor_nodes_indices, -1*jnp.ones((rem, ancestor_nodes_indices.shape[1]))], axis=0).astype(int)

        nu_sticks = jnp.exp(init_nu_log_alphas) / (jnp.exp(init_nu_log_alphas) + jnp.exp(init_nu_log_betas))
        psi_sticks = jnp.exp(init_psi_log_alphas) / (jnp.exp(init_psi_log_alphas) + jnp.exp(init_psi_log_betas))

        print(nu_sticks, psi_sticks)

        logpis = []
        for i, node in enumerate(nodes):
            logpis.append(self.tssb_log_prior(i, nu_sticks, psi_sticks, previous_branches_indices, ancestor_nodes_indices, tssb_weights))
        ws = list(jnp.exp(np.array(logpis)))
        return list(np.array(logpis)), ws

    def tssb_log_prior(self, i, nu_sticks, psi_sticks, previous_branches_indices, ancestor_nodes_indices, tssb_weights):
        # TSSB prior
        nu_stick = nu_sticks[i]
        psi_stick = psi_sticks[i]
        def prev_branches_psi(idx):
            return (idx != -1) * jnp.log(1. - psi_sticks[idx])
        def ancestors_nu(idx):
            _log_phi = jnp.log(psi_sticks[idx]) + jnp.sum(vmap(prev_branches_psi)(previous_branches_indices[idx]))
            _log_1_nu = jnp.log(1. - nu_sticks[idx])
            total = _log_phi + _log_1_nu
            return (idx != -1) * total
        log_phi = jnp.log(psi_stick) + jnp.sum(vmap(prev_branches_psi)(previous_branches_indices[i]))
        log_node_weight = jnp.log(nu_stick) + log_phi + jnp.sum(vmap(ancestors_nu)(ancestor_nodes_indices[i]))
        log_node_weight = log_node_weight + jnp.log(tssb_weights[i])

        return log_node_weight

    def batch_elbo(self, rng, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, params, num_samples):
        # Average over a batch of random samples from the var approx.
        rngs = random.split(rng, num_samples)
        init = [0]
        init.extend([None] * (15 + len(params)))
        vectorized_elbo = vmap(self.root['node'].root['node'].compute_elbo, in_axes=init)
        return jnp.mean(vectorized_elbo(rngs, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, *params))

    @partial(jit, static_argnums=(0,16))
    def objective(self, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, num_samples, params, t):
        print("Recompiling objective!")
        rng = random.PRNGKey(t)
        return -self.batch_elbo(rng, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, params, num_samples)

    @partial(jit, static_argnums=(0,14))
    def batch_objective(self, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, do_global, global_only, sticks_only, num_samples, params, t):
        print("Recompiling batch objective!")
        rng = random.PRNGKey(t)
        return -self.batch_elbo(rng, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, jnp.ones((self.num_data,)), jnp.arange(self.num_data), do_global, global_only, sticks_only, params, num_samples)

    @partial(jit, static_argnums=(0,16))
    def do_grad(self, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, num_samples, params, i):
        return jax.value_and_grad(self.objective, argnums=16)(obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, num_samples, params, i)

    def update(self, obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, num_samples, i, opt_state):
        # print("Recompiling update!")
        params = self.get_params(opt_state)
        value, gradient = self.do_grad(obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only, num_samples, params, i)
        opt_state = self.opt_update(i, gradient, opt_state)
        return opt_state, gradient, params, value

    def optimize_elbo(self, root_node=None, local_node=None, global_only=False, sticks_only=False, unique_node=None, num_samples=10, n_iters=100, thin=10, step_size=0.05, debug=False, tol=1e-5, run=True, max_nodes=5, init=False, opt=None, mb_size=100, callback=None, **callback_kwargs):
        self.max_nodes = len(self.input_tree_dict.keys()) * max_nodes # upper bound on number of nodes
        self.data = jnp.array(self.data, dtype='float32')

        # Var params of nodes below root
        nodes, parent_vector = self.get_nodes(root_node=None, parent_vector=True)

        # Root node label
        data_indices = list(range(self.num_data))
        do_global = False
        if root_node is not None:
            root_label = root_node.label
            data_indices = set()
            def descend(root):
                data_indices.update(root.data)
                for child in root.children():
                    descend(child)
            descend(root_node)
            data_indices = list(data_indices)
            if len(data_indices) == 0 and root_node.parent() is not None:
                data_indices = list(root_node.parent().data)
            # data_indices = list(root_node.data)
        else:
            do_global = True
            root_label = self.root['node'].label
            root_node = self.root['node'].root['node']

        if local_node is not None:
            do_global = False
            root_node = local_node.parent()
            root_label = root_node.label
            init_ass_logits = np.array(np.array([node.data_ass_logits for node in nodes]).T)
            init_ass_probs = jnn.softmax(init_ass_logits, axis=1)
            data_indices = list(np.where(init_ass_probs[:,np.where(root_node == np.array(nodes))[0][0]] > 1./np.sqrt(len(nodes)))[0])
            if len(data_indices) == 0:
                data_indices = np.arange(self.num_data)

        data_mask = np.zeros((self.num_data,))
        data_mask[data_indices] = 1.
        data_mask = data_mask.astype(int)

        parent_vector = jnp.array(np.array(parent_vector))
        tssbs = [node.tssb.label for node in nodes]
        tssb_indices = self.get_tssb_indices(nodes, tssbs)
        children_vector = self.get_children_vector(nodes, parent_vector)
        ancestor_nodes_indices = self.get_ancestor_indices(nodes, parent_vector)
        previous_branches_indices = self.get_previous_branches_indices(nodes)
        node_idx = np.where(np.array(nodes)==root_node)[0][0]
        node_mask_idx = node_idx
        if local_node is None:
            node_mask_idx = self.get_below_root(node_idx, children_vector, tssbs=None)
        else:
            local_node_idx = np.where(np.array(nodes)==local_node)[0][0]
            node_mask_idx = np.array([node_idx, local_node_idx])

        if unique_node is not None:
            node_idx = np.where(np.array(nodes)==unique_node)[0][0]
            node_mask_idx = node_idx
            do_global = False
            data_indices = list(unique_node.data)
            data_mask = np.zeros((self.num_data,))
            data_mask[data_indices] = 1.
            data_mask = data_mask.astype(int)
            sticks_only = True

        node_mask = np.zeros((len(nodes),))
        node_mask[node_mask_idx] = 1

        dp_alphas = jnp.array([node.tssb.dp_alpha for node in nodes])
        dp_gammas = jnp.array([node.tssb.dp_gamma for node in nodes])
        tssb_weights = jnp.array([node.tssb.weight for node in nodes])

        global_params = list(nodes[0].variational_parameters['globals'].values())
        global_params = list(map(lambda arr: jnp.array(arr), global_params))
        global_names = list(nodes[0].variational_parameters['globals'].keys())

        local_params = [list(node.variational_parameters['locals'].values()) for node in nodes]
        local_names = list(nodes[0].variational_parameters['locals'].keys())

        obs_params = jnp.array(np.array([node.observed_parameters for node in nodes]))
        # local_params_names = [list(local_param.keys()) for local_param in local_params]
        # local_params = [list(local_param.values()) for local_param in local_params]
        # params = global_params + local_params

        # # Var params of probs of pivots
        # init_pivot_ass_logits = jnp.array([node.pivot_ass_logits for node in nodes]).T # tssb
        # init_pivot_ass_logits = init_pivot_ass_logits - jnp.mean(init_pivot_ass_logits, axis=1).reshape(-1,1)

        # Pad all of the arrays up to a maximum number of nodes in order to avoid recompiling the ELBO with every structure update
        n_nodes = len(nodes)
        rem = self.max_nodes - n_nodes
        # global_params =

        tssb_weights = jnp.concatenate([tssb_weights, 10*jnp.ones((rem,))])
        dp_gammas = jnp.concatenate([dp_gammas, 1*jnp.ones((rem,))])
        dp_alphas = jnp.concatenate([dp_alphas, 1*jnp.ones((rem,))])
        previous_branches_indices = jnp.concatenate([previous_branches_indices, -1*jnp.ones((rem, previous_branches_indices.shape[1]))], axis=0).astype(int)
        ancestor_nodes_indices = jnp.concatenate([ancestor_nodes_indices, -1*jnp.ones((rem, ancestor_nodes_indices.shape[1]))], axis=0).astype(int)
        children_vector = jnp.concatenate([children_vector, -1*jnp.ones((rem, children_vector.shape[1]))], axis=0).astype(int)
        tssb_indices = jnp.concatenate([tssb_indices, -1*jnp.ones((rem, tssb_indices.shape[1]))], axis=0).astype(int)
        parent_vector = jnp.concatenate([parent_vector, -2*jnp.ones((rem,))]).astype(int)
        obs_params = jnp.concatenate([obs_params, jnp.zeros((rem, nodes[0].observed_parameters.size))], axis=0)
        node_mask = jnp.concatenate([jnp.array(node_mask), -2*jnp.ones((rem,))]).astype(int)
        all_nodes_mask = np.ones(len(node_mask)) * -2
        all_nodes_mask[np.where(node_mask >= 0)[0]] = 1
        all_nodes_mask = jnp.array(all_nodes_mask)
        local_params_list = []
        for param_idx in range(len(local_params[0])):
            l = []
            for node_idx in range(len(nodes)):
                l.append(local_params[node_idx][param_idx])
            l = np.hstack(l).reshape(len(nodes), -1)

            # Add dummies
            param_shape = l[0].shape[0]
            l = jnp.array(np.concatenate([l, np.zeros((rem, param_shape))], axis=0))

            local_params_list.append(l)
        # print([node.label for node in nodes])
        # print(parent_vector)
        # print(children_vector)
        # print([cnv[0] for cnv in cnvs])

        if not do_global:
            print("Won't take derivatives wrt global parameters")
        elif global_only:
            print("Won't take derivatives wrt local parameters")

        do_global = do_global * jnp.array(1.)
        global_only = global_only * jnp.array(1.)
        sticks_only = sticks_only * jnp.array(1.)

        opt_state = self.opt_state
        init_params = local_params_list + global_params
        if self.opt_init is None or init:
            if opt is None:
                opt = optimizers.adam
            opt_init, opt_update, get_params = opt(step_size=step_size)
            self.get_params = jit(get_params)
            self.opt_update = jit(opt_update)
            self.opt_init = jit(opt_init)
        opt_state = self.opt_init(init_params)

        # print(f"Time to prepare optimizer: {end-start} s")
        # n_nodes = jnp.array(n_nodes)
        self.n_nodes = n_nodes
        # print(n_nodes)
        if callback is None:
            def callback(params, t, tree, elbos, tol):
                pass
                # print("Iteration {} lower bound {}".format(t, self.batch_objective(cnvs, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, params, t)))

        print(data_mask)
        print(node_mask)

        # print(all_nodes_mask)
        full_data_indices = jnp.array(np.arange(self.num_data))
        data_mask_subset = jnp.array(data_mask)
        current_elbo = self.elbo
        if run:
            # Main loop.
            current_elbo = self.elbo
            # if self.elbo == -np.inf:
                # current_elbo = -self.batch_objective(obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, all_nodes_mask, do_global, global_only, sticks_only, num_samples, init_params, 0)
            # print(f"Current ELBO: {current_elbo:.5f}")
            # print(f"Optimizing variational parameters from node {root_label}...")
            elbos = []
            means = []
            minibatch_probs = np.ones((self.num_data,))
            minibatch_probs[np.where(data_mask)[0]] = 1e6
            minibatch_probs = minibatch_probs/np.sum(minibatch_probs)
            for t in range(n_iters):
                minibatch_idx = np.random.choice(self.num_data, p=minibatch_probs, size=mb_size, replace=False)
                minibatch_idx = jnp.array(np.sort(minibatch_idx))
                data_mask_subset = jnp.array(data_mask[minibatch_idx])
                # minibatch_idx = np.arange(self.num_data)
                # data_mask_subset = data_mask
                opt_state, g, params, elbo = self.update(obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, minibatch_idx, do_global, global_only, sticks_only, num_samples, t, opt_state)
                elbos.append(-elbo)
                # if t/thin >= 0 and np.mod(t, thin) == 0:
                #     idx = int(t/thin)
                #     params = self.get_params(opt_state)
                #     elbos.append(-self.batch_objective(obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, all_nodes_mask, do_global, global_only, sticks_only, num_samples, params, idx))
                #     try:
                #         callback(params, idx, self, elbos, tol)
                #     except StopIteration:
                #         print(f'Convergence achieved at iteration {t}.')
                #         break

            self.elbo = np.array(-self.batch_objective(obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, all_nodes_mask,
                                    do_global, global_only, sticks_only, num_samples, self.get_params(opt_state), 10))

            # Weigh by tree prior
            subtrees = self.get_mixture()[1][1:] # without the root
            for subtree in subtrees:
                pivot_node = subtree.root['node'].parent()
                parent_subtree = pivot_node.tssb
                prior_weights, subnodes = parent_subtree.get_fixed_weights()
                # Weight ELBO by chosen pivot's prior probability
                node_idx = np.where(pivot_node == np.array(subnodes))[0][0]
                self.elbo = self.elbo + np.log(prior_weights[node_idx])

            # Combinatorial penalization to avoid duplicates -- also avoids real clusters!
            # self.elbo = self.elbo + np.log(1/(2**len(data_indices)))

            # self.opt_state = opt_state
            new_elbo = self.elbo
            # print(f"Done. Speed: {avg_speed} s/it, Total: {total} s")
            # print(f"New ELBO: {new_elbo:.5f}")
            # print(f"New ELBO improvement: {(new_elbo - current_elbo)/np.abs(current_elbo) * 100:.3f}%\n")

            self.set_node_means(self.get_params(opt_state), nodes, local_names, global_names)
            self.update_ass_logits(variational=True)
            self.assign_to_best()
            return elbos
        else:
            self.elbo = np.array(-self.batch_objective(obs_params, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, all_nodes_mask,
                                    do_global, global_only, sticks_only, num_samples, init_params, 10))
            self.update_ass_logits(variational=True)
            self.assign_to_best()
            return None

    def set_node_means(self, params, nodes, local_names, global_names):
        globals_start = len(local_names)
        params_idx = 0
        for i, global_param in enumerate(global_names):
                params_idx = globals_start + i
                self.root['node'].root['node'].variational_parameters['globals'][global_param] = params[params_idx]

        for node_idx, node in enumerate(nodes):
            for i, local_param in enumerate(local_names):
                node.variational_parameters['locals'][local_param] = params[i][node_idx]
            node.set_mean(variational=True)

    def update_ass_logits(self, indices=None, variational=False, prior=True):
        if indices is None:
            indices = list(range(self.num_data))

        nodes, weights = self.get_node_mixture()

        for i, node in enumerate(nodes):
            node_lls = node.loglh(np.array(indices), variational=variational, axis=1)
            node_lls = node_lls + np.log(weights[i]) if prior else node_lls
            node.data_ass_logits[indices] = node_lls

    def assign_to_best(self):
        nodes = self.get_nodes()
        assignment_logits = np.array([node.data_ass_logits for node in nodes]).T
        assignment_probs = np.array(jnn.softmax(assignment_logits, axis=1))

        # Clear all
        for node in nodes:
            node.remove_data()

        for n in range(self.num_data):
            new_node = nodes[np.argmax(assignment_probs[n])]

            # if new_node != self.assignments[n]['node']:
            #     self.assignments[n]['node'].remove_datum(n)
            new_node.add_datum(n)
            self.assignments[n] = new_node

    # ========= Functions to update tree structure. =========

    def add_node_to(self, node, optimal_init=True):
        nodes = self._get_nodes(get_roots=True)
        nodes_list = np.array([node[0] for node in nodes])
        roots_list = np.array([node[1] for node in nodes])
        if isinstance(node, str):
            self.plot_tree(super_only=False);
            nodes_list = np.array([node[0].label for node in nodes])
        node_idx = np.where(nodes_list == node)[0][0]

        root = roots_list[node_idx]

        # Create child
        stick_length = boundbeta(1, self.dp_gamma)
        root['sticks'] = np.vstack([ root['sticks'], stick_length ])
        root['children'].append({ 'node'     : root['node'].spawn(False, root['node'].observed_parameters),
                                  'main'     : boundbeta(1.0, (self.alpha_decay**(root['node'].depth+1))*self.dp_alpha) if self.min_depth <= (root['node'].depth+1) else 0.0,
                                  'sticks'   : np.empty((0,1)),
                                  'children' : []   })
        root['children'][-1]['node'].reset_variational_parameters()

        if optimal_init:
            # Remove some mass from the parent
            root['node'].variational_parameters['locals']['nu_log_mean'] = np.array(0.)
            root['node'].variational_parameters['locals']['nu_log_std'] = np.array(0.)
            root['children'][-1]['node'].data_ass_logits = -np.inf * np.ones((self.num_data))

            # Initialize the mean
            # if len(root['children']) == 1: # Worst explained by parent
            data_indices = list(root['node'].data.copy())
            if len(data_indices) > 0:
                # worst_index = np.argmin(root['node'].data_ass_logits[data_indices])
                worst_index = np.random.choice(np.array(data_indices)[np.array([np.argsort(root['node'].data_ass_logits[data_indices])[:5]])].ravel())
                print(f'Setting new node to explain datum {worst_index}')
                worst_datum = self.data[worst_index]
                baseline = np.append(1, np.exp(self.root['node'].root['node'].log_baseline_caller()))
                noise = self.root['node'].root['node'].variational_parameters['globals']['cell_noise_mean'][worst_index].dot(self.root['node'].root['node'].variational_parameters['globals']['noise_factors_mean'])
                total_rna = np.sum(baseline * root['node'].cnvs/2 * np.exp(root['node'].variational_parameters['locals']['unobserved_factors_mean'] + noise))
                root['children'][-1]['node'].variational_parameters['locals']['unobserved_factors_mean'] = np.log((worst_datum+1) * total_rna/(self.root['node'].root['node'].lib_sizes[worst_index]*baseline * root['node'].cnvs/2 * np.exp(noise)))
                root['children'][-1]['node'].set_mean(root['children'][-1]['node'].get_mean(unobserved_factors=root['children'][-1]['node'].variational_parameters['locals']['unobserved_factors_mean'], baseline=baseline))

        return root['children'][-1]['node']

    def perturb_node(self, node, target):
        # Perturb parameters of node to become closer to data explained by target
        if isinstance(node, str) and isinstance(target, str):
            self.plot_tree(super_only=False);
            nodes_list = np.array(self.get_nodes())
            node_labels = np.array([node[0].label for node in nodes_list])
            node = nodes_list[np.where(node_labels == node)[0][0]]
            target = nodes_list[np.where(node_labels == target)[0][0]]

        data_indices = list(target.data.copy())

        if len(data_indices) > 0:
            index = np.random.choice(np.array(data_indices))
            # worst_index = np.argmin(root['node'].data_ass_logits[data_indices])
            # worst_index = np.random.choice(np.array(data_indices)[np.array([np.argsort(target.data_ass_logits[data_indices])[:5]])].ravel())
            print(f'Setting node to explain datum {index}')
            worst_datum = self.data[index]
            baseline = np.append(1, np.exp(self.root['node'].root['node'].log_baseline_caller()))
            noise = self.root['node'].root['node'].variational_parameters['globals']['cell_noise_mean'][index].dot(self.root['node'].root['node'].variational_parameters['globals']['noise_factors_mean'])
            total_rna = np.sum(baseline * node.cnvs/2 * np.exp(node.variational_parameters['locals']['unobserved_factors_mean'] + noise))
            node.variational_parameters['locals']['unobserved_factors_mean'] = np.log((worst_datum+1) * total_rna/(self.root['node'].root['node'].lib_sizes[index]*baseline * node.cnvs/2 * np.exp(noise)))
            node.set_mean(node.get_mean(unobserved_factors=node.variational_parameters['locals']['unobserved_factors_mean'], baseline=baseline))

    def remove_last_leaf_node(self, parent_label):
        nodes = self._get_nodes(get_roots=True)
        node_labels = np.array([node[0].label for node in nodes])
        node_idx = np.where(node_labels == parent_label)[0][0]
        root = nodes[node_idx][1]

        # Remove last child
        root['sticks'] = root['sticks'][:-2]
        root['children'][-1]['node'].kill()
        del root['children'][-1]['node']
        root['children'] = root['children'][:-2]

    def pivot_reattach_to(self, subtree, pivot):
        nodes = self._get_nodes(get_roots=False)
        nodes = np.array(nodes)
        if isinstance(subtree, str) or isinstance(pivot, str):
            self.plot_tree(super_only=False);
            node_labels = np.array([node.label for node in nodes])
            subtree = nodes[np.where(node_labels == subtree)[0][0]]
            subtree = subtree.tssb
            pivot = nodes[np.where(node_labels == pivot)[0][0]]

        subtree_label = subtree.label

        root_node_idx = np.where(nodes == subtree.root['node'])[0][0]
        root_node = nodes[root_node_idx]
        pivot_node_idx = np.where(nodes == pivot)[0][0]
        pivot_node = nodes[pivot_node_idx]

        subtrees = self.get_subtrees(get_roots=True)
        subtree_objs = np.array([s[0] for s in subtrees])
        subtree_idx = np.where(subtree_objs == subtree)[0][0]
        subtrees[subtree_idx][1]['pivot_node'] = pivot_node

        # prev_unobserved_factors = root_node[0].unobserved_factors_mean
        root_node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = np.log(root_node.unobserved_factors_kernel_concentration_caller())*np.ones((root_node.n_genes,))
        root_node.set_parent(pivot_node, reset=False)
        root_node.set_mean(variational=True)
        # Reset the kernel posterior
        # root_node[0].unobserved_factors_kernel_log_mean = -1.*jnp.ones((root_node[0].n_genes,))
        # root_node[0].unobserved_factors_kernel_log_std = -1.*jnp.ones((root_node[0].n_genes,))
        # root_node[0].unobserved_factors_mean = prev_unobserved_factors
        # root_node[0].set_mean(variational=True)
        # self.update_ass_logits(variational=True)
        # self.assign_to_best()

    def push_subtree(self, node):
        """
        push_subtree(B):
        A-0 -> B -> B-0 to A-0 -> A-0-0 -> B -> B-0
        """
        if isinstance(node, str):
            self.plot_tree(super_only=False);
            nodes = self.get_nodes(None)
            node_labels = np.array([node.label for node in nodes])
            node = nodes[np.where(node_labels == node)[0][0]]

        if node.parent() is None:
            raise ValueError("Can't pull from root tree")
        if not node.is_observed:
            raise ValueError("Can't pull unobserved node")

        parent_node = node.parent()
        children = list(node.children())
        if len(children) > 0:
            children = [n for n in children if not n.is_observed]
        child_node = None
        if len(children) > 0:
            child_node = children[0]

        # Add node below parent
        new_node = self.add_node_to(parent_node)
        self.pivot_reattach_to(node.tssb, new_node)
        paramsB = node.variational_parameters['locals']['unobserved_factors_mean']
        paramsB_k = node.variational_parameters['locals']['unobserved_factors_kernel_log_mean']
        dataB = node.data.copy()
        logitsB = np.array(node.data_ass_logits)
        if child_node:
            node.variational_parameters['locals']['unobserved_factors_mean'] = child_node.variational_parameters['locals']['unobserved_factors_mean']
            node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = child_node.variational_parameters['locals']['unobserved_factors_kernel_log_mean']
            node.data = child_node.data.copy()
            node.data_ass_logits = np.array(child_node.data_ass_logits)
            # Merge B with child that it has become equal to
            self.merge_nodes(child_node, node)
        node.set_mean(variational=True)

        # Set new node's parameters equal to the previous parameters of node
        new_node.variational_parameters['locals']['unobserved_factors_mean'] = paramsB
        new_node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = paramsB_k
        new_node.data = dataB.copy()
        new_node.data_ass_logits = np.array(logitsB)
        new_node.set_mean(variational=True)

    def swap_nodes(self, nodeA, nodeB):
        if isinstance(nodeA, str) and isinstance(nodeB, str):
            self.plot_tree(super_only=False);
            nodes = self.get_nodes(None)
            node_labels = np.array([node.label for node in nodes])
            nodeA = nodes[np.where(node_labels == nodeA)[0][0]]
            nodeB = nodes[np.where(node_labels == nodeB)[0][0]]

        # If we are swapping the root node, need to change the baseline too
        root_node = None
        non_root_node = None
        child_unobserved_factors = None
        initial_log_baseline = None
        if nodeA.parent() is None:
            root_node = nodeA
            non_root_node = nodeB
        elif nodeB.parent() is None:
            root_node = nodeB
            non_root_node = nodeA

        def swap_params(nA, nB):
            paramsA = nA.variational_parameters['locals']['unobserved_factors_mean']
            paramsA_k = nA.variational_parameters['locals']['unobserved_factors_kernel_log_mean']
            dataA = nA.data.copy()
            logitsA = np.array(nA.data_ass_logits)
            nA.variational_parameters['locals']['unobserved_factors_mean'] = nB.variational_parameters['locals']['unobserved_factors_mean']
            nA.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = nB.variational_parameters['locals']['unobserved_factors_kernel_log_mean']
            nA.data = nB.data.copy()
            nA.data_ass_logits = np.array(nB.data_ass_logits)
            nA.set_mean(variational=True)
            nB.variational_parameters['locals']['unobserved_factors_mean'] = paramsA
            nB.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = paramsA_k
            nB.data = dataA.copy()
            nB.data_ass_logits = np.array(logitsA)
            nB.set_mean(variational=True)

        if not root_node:
            if nodeA.tssb == nodeB.tssb:
                swap_params(nodeA, nodeB)
            else:
                # e.g. A-0 with C
                if nodeA.parent() == nodeB or nodeB.parent() == nodeA:
                    unobserved_node_idx = np.where([not nodeA.is_observed, not nodeB.is_observed])[0]
                    if len(unobserved_node_idx) > 0:
                        # change A -> A-0 -> B to A-> B -> B-0:
                        unobserved_node_idx = unobserved_node_idx[0]
                        unobserved_node = [nodeA, nodeB][unobserved_node_idx]
                        observed_node = [nodeA, nodeB][1 - unobserved_node_idx]
                        parent_unobserved = unobserved_node.parent()
                        # if unobserved node is parent of more than one subtree, don't proceed with full swap
                        unobserved_node_children = unobserved_node.children()
                        if not (np.sum(np.array([child.is_observed for child in unobserved_node_children])) > 1):
                            # Update params
                            init_obs_params = observed_node.variational_parameters['locals']['unobserved_factors_mean']
                            observed_node.variational_parameters['locals']['unobserved_factors_mean'] = parent_unobserved.variational_parameters['locals']['unobserved_factors_mean']
                            unobserved_node.variational_parameters['locals']['unobserved_factors_mean'] = init_obs_params
                            observed_node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = np.log(observed_node.unobserved_factors_kernel_concentration_caller())*np.ones((observed_node.n_genes,))
                            unobserved_node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = np.log(unobserved_node.unobserved_factors_kernel_concentration_caller())*np.ones((unobserved_node.n_genes,))

                            # Put same-subtree children of unobserved node in its parent in order to not move a whole subtree
                            nodes = self._get_nodes(get_roots=True)
                            unobserved_node_tssb_root = [node[1] for node in nodes if node[0] == unobserved_node][0]
                            parent_unobserved_node_tssb_root = [node[1] for node in nodes if node[0] == parent_unobserved][0]
                            for i, unobs_child in enumerate(unobserved_node_tssb_root['children']):
                                unobs_child['node'].set_parent(unobserved_node.parent())
                                # Add children from unobserved to the parent dict
                                parent_unobserved_node_tssb_root['children'].append(unobs_child)
                                parent_unobserved_node_tssb_root['sticks'] = np.vstack([ parent_unobserved_node_tssb_root['sticks'], unobserved_node_tssb_root['sticks'][i] ])
                            if len(unobserved_node_tssb_root['children']) > 0:
                                # Remove children from unobserved
                                unobserved_node_tssb_root['sticks']   = np.array([]).reshape(0,1)
                                unobserved_node_tssb_root['children'] = []

                            # Now move the unobserved node to below the observed one
                            observed_node.set_parent(parent_unobserved)
                            unobserved_node.set_parent(observed_node)
                            unobserved_node.tssb = observed_node.tssb
                            unobserved_node.cnvs = observed_node.cnvs
                            n_siblings = len(list(observed_node.children()))
                            unobserved_node.label = observed_node.label + '-' + str(n_siblings-1)

                            nodes = self._get_nodes(get_roots=True)
                            unobserved_node_tssb_root = [node[1] for node in nodes if node[0] == unobserved_node][0]
                            parent_unobserved_node_tssb_root = [node[1] for node in nodes if node[0] == parent_unobserved][0]

                            # Update dicts
                            # Remove unobserved_node from its parent dict
                            childnodes = np.array([n['node'] for n in parent_unobserved_node_tssb_root['children']])
                            tokeep = np.where(childnodes != unobserved_node_tssb_root['node'])[0].astype(int).ravel()
                            parent_unobserved_node_tssb_root['sticks']   = parent_unobserved_node_tssb_root['sticks'][tokeep]
                            parent_unobserved_node_tssb_root['children'] = list(np.array(parent_unobserved_node_tssb_root['children'])[tokeep])
                            # Update observed_node's pivot_node to unobserved_node's parent
                            observed_node_ntssb_root = observed_node.tssb.get_ntssb_root()
                            observed_node_ntssb_root['pivot_node'] = parent_unobserved
                            # Add unobserved_node to observed_node's dict
                            observed_node_tssb_root = observed_node_ntssb_root['node'].root
                            observed_node_tssb_root['children'].append(unobserved_node_tssb_root)
                            observed_node_tssb_root['sticks'] = np.vstack([observed_node_tssb_root['sticks'], 1.])
        else: # e.g. A with A-0
            init_baseline = np.mean(self.data / np.sum(self.data, axis=1).reshape(-1,1) * self.data.shape[1], axis=0)
            init_baseline = init_baseline / init_baseline[0]
            init_log_baseline = np.log(init_baseline[1:])
            root_node.variational_parameters['globals']['log_baseline_mean'] = init_log_baseline
            root_node.variational_parameters['locals']['unobserved_factors_mean'] = np.zeros((self.data.shape[1],))
            root_node.set_mean(variational=True)

            non_root_node.variational_parameters['locals']['unobserved_factors_mean'] = normal_sample(0., gamma_sample(root_node.unobserved_factors_kernel_concentration_caller(),
                                                                                                root_node.unobserved_factors_kernel_concentration_caller(), size=self.data.shape[1]))
            data_indices = list(root_node.data)
            if len(data_indices) > 0:
                idx = np.random.choice(np.array(data_indices))
                print(f'Setting new node to explain datum {idx}')
                datum = self.data[idx]
                baseline = np.append(1, np.exp(non_root_node.log_baseline_caller()))
                total_rna = np.sum(baseline * non_root_node.cnvs/2 * np.exp(root_node.variational_parameters['locals']['unobserved_factors_mean']))
                non_root_node.variational_parameters['locals']['unobserved_factors_mean'] = np.log((datum+1) * total_rna/(root_node.lib_sizes[idx]*baseline * root_node.cnvs/2))
            non_root_node.set_mean(variational=True)
            dataRoot = root_node.data.copy()
            logitsRoot = np.array(root_node.data_ass_logits)
            root_node.data = non_root_node.data.copy()
            root_node.data_ass_logits = np.array(non_root_node.data_ass_logits)
            non_root_node.data = dataRoot.copy()
            non_root_node.data_ass_logits = np.array(logitsRoot)

        if nodeA.tssb == nodeB.tssb:
            root = nodeB.tssb.get_ntssb_root()
            # For each subtree, if pivot was swapped, update it
            for child in root['children']:
                if child['pivot_node'] == nodeA:
                    child['pivot_node'] = nodeB
                    child['node'].root['node'].set_parent(nodeB, reset=False)
                    child['node'].root['node'].set_mean(variational=True)
                elif child['pivot_node'] == nodeB:
                    child['pivot_node'] = nodeA
                    child['node'].root['node'].set_parent(nodeA, reset=False)
                    child['node'].root['node'].set_mean(variational=True)

    # def swap_nodes(self, nodeA, nodeB):
    #     if isinstance(nodeA, str) and isinstance(nodeB, str):
    #         self.plot_tree(super_only=False)
    #         nodes = self.get_nodes(None)
    #         node_labels = np.array([node.label for node in nodes])
    #         nodeA = nodes[np.where(node_labels == nodeA)[0][0]]
    #         nodeB = nodes[np.where(node_labels == nodeB)[0][0]]
    #
    #     # If we are swapping the root node, need to change the baseline too
    #     root_node = None
    #     non_root_node = None
    #     child_unobserved_factors = None
    #     initial_log_baseline = None
    #     if nodeA.parent() is None:
    #         root_node = nodeA
    #         non_root_node = nodeB
    #     elif nodeB.parent() is None:
    #         root_node = nodeB
    #         non_root_node = nodeA
    #     if root_node and non_root_node:
    #         child_unobserved_factors = non_root_node.unobserved_factors
    #         child_unobserved_factors_k = non_root_node.unobserved_factors
    #         initial_log_baseline = root_node.log_baseline_mean
    #
    #     if not root_node:
    #         paramsA = nodeA.variational_parameters['locals']['unobserved_factors_mean']
    #         paramsA_k = nodeA.variational_parameters['locals']['unobserved_factors_kernel_log_mean']
    #         # sticks_alphaA = nodeA.nu_log_alpha
    #         # sticks_betaA = nodeA.nu_log_beta
    #         dataA = nodeA.data
    #         logitsA = nodeA.data_ass_logits
    #
    #         nodeA.variational_parameters['locals']['unobserved_factors_mean'] = nodeB.variational_parameters['locals']['unobserved_factors_mean']
    #         nodeA.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = nodeB.variational_parameters['locals']['unobserved_factors_kernel_log_mean']
    #         nodeA.data = nodeB.data
    #         nodeA.data_ass_logits = nodeB.data_ass_logits
    #         # nodeA.nu_log_alpha = nodeB.nu_log_alpha
    #         # nodeA.nu_log_beta = nodeB.nu_log_beta
    #         nodeA.set_mean(variational=True)
    #
    #         nodeB.variational_parameters['locals']['unobserved_factors_mean'] = paramsA
    #         nodeB.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = paramsA_k
    #         nodeB.data = dataA
    #         nodeB.data_ass_logits = logitsA
    #         # nodeB.nu_log_alpha = sticks_alphaA
    #         # nodeB.nu_log_beta = sticks_betaA
    #         nodeB.set_mean(variational=True)
    #
    #     if root_node and non_root_node:
    #         root_node.variational_parameters['locals']['unobserved_factors_mean'] = child_unobserved_factors
    #         root_node.variational_parameters['globals']['log_baseline_mean'] = non_root_node.variational_parameters['locals']['unobserved_factors_mean'][1:]
    #         root_node.set_mean(variational=True)
    #         non_root_node.variational_parameters['locals']['unobserved_factors_mean'] = np.append(0., initial_log_baseline)
    #         non_root_node.set_mean(variational=True)
    #
    #     if nodeA.tssb == nodeB.tssb:
    #         # Go to subtrees
    #         # For each subtree, if pivot was swapped, update it
    #         root = nodeB.tssb.get_ntssb_root()
    #         for child in root['children']:
    #             if child['pivot_node'] == nodeA:
    #                 child['pivot_node'] = nodeB
    #                 child['node'].root['node'].set_parent(nodeB, reset=False)
    #                 child['node'].root['node'].set_mean(variational=True)
    #             elif child['pivot_node'] == nodeB:
    #                 child['pivot_node'] = nodeA
    #                 child['node'].root['node'].set_parent(nodeA, reset=False)
    #                 child['node'].root['node'].set_mean(variational=True)
    #     else:
    #         if nodeA.parent() == nodeB or nodeB.parent() == nodeA:
    #             unobserved_node_idx = np.where([not nodeA.is_observed, not nodeB.is_observed])[0]
    #             if len(unobserved_node_idx) > 0:
    #                 # change A -> A-0 -> B to A-> B -> B-0:
    #                 unobserved_node_idx = unobserved_node_idx[0]
    #                 unobserved_node = [nodeA, nodeB][unobserved_node_idx]
    #                 # if unobserved node is parent of more than one subtree, don't proceed with full swap
    #                 unobserved_node_children = unobserved_node.children()
    #                 if not (np.sum(np.array([child.is_observed for child in unobserved_node_children])) > 1):
    #                     observed_node = [nodeA, nodeB][1 - unobserved_node_idx]
    #                     parent_unobserved = unobserved_node.parent()
    #                     observed_node.set_parent(parent_unobserved)
    #                     unobserved_node.set_parent(observed_node)
    #
    #                     nodes = self._get_nodes(get_roots=True)
    #                     unobserved_node_tssb_root = [node[1] for node in nodes if node[0] == unobserved_node][0]
    #                     parent_unobserved_node_tssb_root = [node[1] for node in nodes if node[0] == parent_unobserved][0]
    #
    #                     # Update dicts
    #                     # Remove unobserved_node from its parent dict
    #                     childnodes = np.array([n['node'] for n in parent_unobserved_node_tssb_root['children']])
    #                     tokeep = np.where(childnodes != unobserved_node_tssb_root['node'])[0].astype(int).ravel()
    #                     parent_unobserved_node_tssb_root['sticks']   = parent_unobserved_node_tssb_root['sticks'][tokeep]
    #                     parent_unobserved_node_tssb_root['children'] = list(np.array(parent_unobserved_node_tssb_root['children'])[tokeep])
    #
    #                     # Update observed_node's pivot_node to unobserved_node's parent
    #                     observed_node_ntssb_root = observed_node.tssb.get_ntssb_root()
    #                     observed_node_ntssb_root['pivot_node'] = parent_unobserved
    #
    #                     # Add unobserved_node to observed_node's dict
    #                     observed_node_tssb_root = observed_node_ntssb_root['node'].root
    #                     observed_node_tssb_root['children'].append(unobserved_node_tssb_root)
    #                     observed_node_tssb_root['sticks'] = np.vstack([observed_node_tssb_root['sticks'], 1.])

    def merge_nodes(self, nodeA, nodeB):
        if isinstance(nodeA, str) and isinstance(nodeB, str):
            self.plot_tree(super_only=False)
            nodes = self.get_nodes(None)
            node_labels = np.array([node.label for node in nodes])
            nodeA = nodes[np.where(node_labels == nodeA)[0][0]]
            nodeB = nodes[np.where(node_labels == nodeB)[0][0]]

        nodes = self._get_nodes(get_roots=True)
        nodes_list = np.array([node[0] for node in nodes])
        nodeA_idx = np.where(nodes_list == nodeA)[0][0]
        nodeB_idx = np.where(nodes_list == nodeB)[0][0]
        nodeA_root = nodes[nodeA_idx][1]
        nodeB_root = nodes[nodeB_idx][1]
        nodeA_parent_root = nodes[np.where(np.array(nodes) == nodeA.parent())[0][0]][1]

        # Move child nodes of nodeA to nodeB and remove nodeA
        n_childrenA = len(nodeA_root['children'])
        n_childrenB = len(nodeB_root['children'])
        for i, nodeA_child in enumerate(nodeA_root['children']):
            nodeA_child['node'].set_parent(nodeB_root['node'], reset=False)
            nodeA_child['node'].set_mean(variational=True)
            nodeB_root['children'].append(nodeA_child)
            nodeB_root['sticks'] = np.vstack([nodeB_root['sticks'], 1.])
        nodeA_root['children'] = []

        # If nodeA was the pivot of a downstream tree, update the pivot to nodeB
        nodeA_children = nodeA_root['node'].children().copy()
        for nodeA_child in nodeA_children:
            if nodeA_child.tssb != nodeA_root['node'].tssb:
                nodeA_child.set_parent(nodeB_root['node'], reset=False)
                nodeA_child.set_mean(variational=True)
                ntssb_root = nodeA_child.tssb.get_ntssb_root()
                ntssb_root['pivot_node'] = nodeB_root['node']
        nodeA_root['node'].children().clear()

        nodeB.data.update(nodeA.data)

        # Remove nodeA from tssb root dict
        nodes = np.array([n['node'] for n in nodeA_parent_root['children']])
        tokeep = np.where(nodes != nodeA)[0].astype(int).ravel()
        nodeA_root['node'].kill()
        del nodeA_root['node']

        nodeA_parent_root['sticks']   = nodeA_parent_root['sticks'][tokeep]
        nodeA_parent_root['children'] = list(np.array(nodeA_parent_root['children'])[tokeep])

    def subtree_reattach_to(self, node, target_clone):
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

        def descend(root):
            ns = [root['node']]
            for child in root['children']:
                child_node = descend(child)
                ns.extend(child_node)
            return ns
        nodes_below_nodeA = descend(roots[nodeA_idx])

        # Get the target clone subtree
        subtrees = self.get_subtrees(get_roots=True)
        subtrees = [subtree[1] for subtree in subtrees] # the roots
        subtree_labels = np.array([subtree['node'].label for subtree in subtrees])
        if isinstance(target_clone, str):
            subtree_idx = np.where(np.array(subtree_labels) == target_clone)[0][0]
        else:
            subtree_idx = np.where(np.array(subtrees) == target_clone)[0][0]
        target_subtree = subtrees[subtree_idx]
        subtreeA = subtrees[np.where(np.array([s['node'].label for s in subtrees]) == nodes[nodeA_idx].tssb.label)[0][0]]

        # Check if there is a pivot here
        for subtree_child in subtreeA['children']:
            for n in nodes_below_nodeA:
                if subtree_child['pivot_node'] == n:
                    subtree_child['node'].root['node'].set_parent(subtreeA['node'].root['node'])
                    subtree_child['node'].root['node'].set_mean(variational=True)
                    subtree_child['pivot_node'] = subtreeA['node'].root['node']
                    break

        # Move subtree
        roots[nodeA_idx]['node'].set_parent(target_subtree['node'].root['node'])
        roots[nodeA_idx]['node'].set_mean(variational=True)
        roots[nodeA_idx]['node'].tssb = target_subtree['node']
        target_subtree['node'].root['children'].append(roots[nodeA_idx])
        target_subtree['node'].root['sticks'] = np.vstack([target_subtree['node'].root['sticks'], 1.])

        childnodes = np.array([n['node'] for n in roots[nodeA_parent_idx]['children']])
        tokeep = np.where(childnodes != roots[nodeA_idx]['node'])[0].astype(int).ravel()
        roots[nodeA_parent_idx]['sticks']   = roots[nodeA_parent_idx]['sticks'][tokeep]
        roots[nodeA_parent_idx]['children'] = list(np.array(roots[nodeA_parent_idx]['children'])[tokeep])

        # Set new unobserved factors to explain the same data as before (i.e. keep mean equal)
        # baseline = jnp.append(1, jnp.exp(self.root['node'].root['node'].log_baseline_caller()))
        # total_rna = jnp.sum(baseline * roots[nodeA_idx]['node'].cnvs/2 * jnp.exp(roots[nodeA_idx]['node'].unobserved_factors_mean))
        # roots[nodeA_idx]['node'].unobserved_factors_mean = np.log(init_mean * total_rna / (baseline * roots[nodeA_idx]['node'].cnvs/2))

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
            for child in super_tree['children']:
                nodes, weights = super_tree['node'].get_fixed_weights()
                # Re-weight the node pivot probabilities by the likelihood
                # of the resulting subtree's root parameter
                reweights = [child['node'].root['node'].unobserved_factors_ll(node.unobserved_factors) + np.log(weights[i]) for i, node in enumerate(nodes)]
                reweights = np.array(reweights)
                probs = np.exp(reweights - logsumexp(reweights))
                pivot_node = np.random.choice(nodes, p=probs)
                if verbose:
                    print(f"Pivot of {child['node'].root['node'].label}: {child['pivot_node'].label}->{pivot_node.label}")
                child['pivot_node'] = pivot_node
                if child['node'].root['node'].parent() != pivot_node:
                    child['node'].root['node'].set_parent(pivot_node, reset=True) # update parameters

                descend(child)
        descend(self.root)

    def cull_subtrees(self, verbose=False, resample_sticks=True):
        culled = []
        def descend(super_tree):
            for child in super_tree['children']:
                descend(child)
            culled.append(super_tree['node'].cull_tree(verbose=verbose, resample_sticks=resample_sticks))
        descend(self.root)
        return culled

    def cull_tree(self):
        """
        If a leaf node has no data assigned to it, remove it
        """
        def descend(root):
            counts = list(map(lambda child: descend(child), root['children']))
            keep   = len(trim_zeros(counts, 'b'))

            for child in root['children'][keep:]:
                child['node'].kill()
                del child['node']

            root['sticks']   = root['sticks'][:keep]
            root['children'] = root['children'][:keep]

            return sum(counts) + root['node'].num_local_data() + (not root['node'].is_subcluster) # if it is a subcluster, returns 0.
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
                    llhs.append(node.num_local_data()*log(node_weights[j])*log(subtree_weights[i]) + node.data_log_likelihood())
        llh = sum(array(llhs))

        return llh

    def complete_data_log_likelihood_nomix(self):
        weights, nodes = self.get_mixture();
        llhs = []
        lln = []
        for i, node in enumerate(nodes):
            if node.num_local_data():
                llhs.append(node.data_log_likelihood())
                lln.append(node.num_local_data()*log(weights[i]))
                #lln.append(weights[i])
        return (sum(array(lln)),sum(array(llhs)))

    def unnormalized_posterior_with_hypers(self):
        weights, nodes = self.get_mixture();
        llhs = []
        for i, node in enumerate(nodes):
            if node.num_local_data():
                llhs.append(node.num_local_data()*log(weights[i]) +
                            node.data_log_likelihood() +
                            node.parameter_log_prior() )
        llh = sum(array(llhs))

        def alpha_descend(root, depth=0):
            llh = betapdfln(root['main'], 1.0,
                            (self.alpha_decay**depth)*self.dp_alpha) if self.min_depth <= depth else 0.0
            for child in root['children']:
                llh += alpha_descend(child, depth+1)
            return llh
        weights_log_prob = alpha_descend(self.root)

        def gamma_descend(root):
            llh = 0
            for i, child in enumerate(root['children']):
                llh += betapdfln(root['sticks'][i], 1.0, self.dp_gamma)
                llh += gamma_descend(child)
            return llh
        sticks_log_prob =  gamma_descend(self.root)

        return llh + weights_log_prob + sticks_log_prob

    def unnormalized_posterior(self, verbose=False, compound=False):
        # Go to each subtree
        subtree_weights, subtrees = self.get_mixture()
        llhs = []
        for i, subtree in enumerate(subtrees):
            node_weights, nodes, roots, depths = subtree.get_mixture(get_roots=True, get_depths=True, truncate=True)
            for j, node in enumerate(nodes):
                if node.num_local_data():
                    llhs.append(node.num_local_data()*log(node_weights[j]*subtree_weights[i]) +
                                node.data_log_likelihood() +
                                subtree.sticks_node_logprior(roots[j], depths[j]) +
                                node.logprior())
                    if verbose:
                        print(f'{node.label}: {node.num_local_data()*log(node_weights[j]*subtree_weights[i])}\t{node.data_log_likelihood()}\t{subtree.sticks_node_logprior(roots[j], depths[j])}\t{node.logprior()}')
        llh = sum(array(llhs))

        return llh

    def remove_empty_nodes(self):
        def descend(root):
            for index, child in enumerate(root['children']):
                descend(child)
            root['node'].remove_empty_nodes()
        descend(self.root)

    # ========= Functions to update tree metadata. =========

    def label_nodes(self, counts=False, names=False):
        if not counts or names is True:
            self.label_nodes_names()
        elif not names or counts is True:
            self.label_nodes_counts()

    def set_node_names(self, root_name='X'):
        self.root['label'] = str(root_name)
        self.root['node'].label = str(root_name)

        def descend(root, name):
            for i, child in enumerate(root['children']):
                child_name = "%s-%d" % (name, i)

                root['children'][i]['label'] = child_name
                root['children'][i]['node'].label = child_name

                descend(child, child_name)

        descend(self.root, root_name)

    def tssb_dict2graphviz(self, g=None, counts=False, root_fillcolor=None, events=False, color_subclusters=False, show_labels=True, gene=None, genemode='raw', fontcolor='black'):
        if g is None:
            g = Digraph()
            g.attr(fontcolor=fontcolor)

        if gene is not None:
            name_exp_dict = dict([(n.label,nodeavg[gene]) for n, nodeavg in zip(*self.get_avg_subtree_exp())])
            cmap = self.exp_cmap
            arr = np.array(list(name_exp_dict.values()))
            norm = matplotlib.colors.Normalize(vmin=min(arr), vmax=max(arr))
            if genemode == 'observed':
                name_exp_dict = dict([(n.label,nodeavg[gene]) for n, nodeavg in zip(*self.get_subtree_obs())])
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
            root_label = self.root['label']
        if counts:
            root_label = self.root['node'].num_data()
        if show_labels and counts:
            root_label = self.root['label'] + '<br/><br/>' + str(self.root['node'].num_local_data()) + ' cells'
        if events:
            root_label = self.root['node'].event_str
        if show_labels and events:
            root_label = self.root['label'] + '<br/><br/>' + self.root['node'].event_str

        style = None
        if root_fillcolor is not None:
            style = 'filled'
        if root_fillcolor is None:
            root_fillcolor = self.root['node'].color
        if gene is not None:
            style = 'filled'
            fillcolor=name_color_dict[str(self.root['label'])]
        g.node(str(self.root['label']), '<' + str(root_label) + '>', fillcolor=root_fillcolor, style=style)

        def descend(root, g):
            name = root['label']
            for i, child in enumerate(root['children']):
                child_name = child['label']
                child_label = ""
                if show_labels:
                    child_label = child_name

                if counts:
                    child_label = root['children'][i]['node'].num_data()

                if show_labels and counts:
                    child_label = child_name + '<br/><br/>' + str(root['children'][i]['node'].num_local_data()) + ' cells'

                if events:
                    child_label = root['children'][i]['node'].event_str

                if show_labels and events:
                    child_label = child_name + '<br/><br/>' + self.root['node'].event_str

                fillcolor = child['node'].color
                if gene is not None:
                    fillcolor = name_color_dict[str(child_name)]
                    style = 'filled'
                g.node(str(child_name), '<' + str(child_label) + '>', fillcolor=fillcolor, style=style)

                edge_color = 'black'

                g.edge(str(name), str(child_name), color=edge_color)

                g = descend(child, g)

            return g

        g = descend(self.root, g)
        return g

    def get_node_unobs(self):
        nodes = self.get_nodes(None)
        unobs = []
        for node in nodes:
            unobs.append(node.unobserved_factors)
        return nodes, unobs

    def get_node_obs(self):
        nodes = self.get_nodes(None)
        obs = []
        for node in nodes:
            obs.append(node.observed_parameters)
        return nodes, obs

    def get_avg_node_exp(self, norm=True):
        nodes = self.get_nodes(None)
        data = self.normalized_data if norm else self.data
        avgs = []
        for node in nodes:
            idx = np.array(list(node.data))
            avgs.append(np.mean(data[idx], axis=0))
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
            obs.append(subtree.root['node'].cnvs)
        return subtrees, obs

    def plot_tree(self, super_only=False, counts=False, root_fillcolor=None, events=False, color_subclusters=False, reset_names=True, ordered=False,
                    genemode='raw', show_labels=True, color_by_weight=False, gene=None, fontcolor='black'):
        if super_only:
            g = self.tssb_dict2graphviz(counts=counts, root_fillcolor=root_fillcolor, events=events, show_labels=show_labels, gene=gene, genemode=genemode, fontcolor=fontcolor)
        else:
            g = self.root['node'].plot_tree(counts=counts, reset_names=True, root_fillcolor=root_fillcolor, events=events, show_labels=show_labels, gene=gene, genemode=genemode, fontcolor=fontcolor)
            def descend(root, g):
                for i, child in enumerate(root['children']):
                    g = child['node'].plot_tree(g, reset_names=True, counts=counts, root_fillcolor=root_fillcolor, show_labels=show_labels, gene=gene, genemode=genemode, fontcolor=fontcolor, events=events)
                    if counts:
                        lab = str(child['pivot_node'].num_local_data())
                        if show_labels:
                            lab = child['pivot_node'].label + '<br/><br/>' + lab + ' cells'
                        g.node(child['pivot_node'].label, '<' + lab + '>')
                    elif events:
                        lab = child['pivot_node'].event_str
                        if show_labels:
                            lab = child['pivot_node'].label + '<br/><br/>' + lab
                        g.node(child['pivot_node'].label, '<' + lab + '>')
                    g.edge(child['pivot_node'].label, child['node'].root['label'])
                    g = descend(child, g)
                return g
            g = descend(self.root, g)

        return g

    def label_assignments(self, root_name='X', reset_names=True):
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
            for i, child in enumerate(root['children']):
                child_name = child['label']

                # Get occurrences of node in assignment list
                idx = np.where(assignments == (str(child['node'])[-12:-1]))[0]

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
        descend(self.root['node'].root['node'])
