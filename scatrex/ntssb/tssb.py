"""
Tree-structured stick breaking process.
"""

import sys
import scipy.stats
import copy
import string

from graphviz import Digraph

from time import *
from numpy import *
from numpy.random import *
import numpy as np

import matplotlib
from ..utils.math_utils import *

import logging

logger = logging.getLogger(__name__)


class TSSB(object):
    min_dp_alpha = 0.001
    max_dp_alpha = 10.0
    min_dp_gamma = 0.001
    max_dp_gamma = 10.0
    min_alpha_decay = 0.001
    max_alpha_decay = 0.80

    def __init__(
        self,
        root_node,
        label,
        ntssb=None,
        parent=None,
        dp_alpha=1.0,
        dp_gamma=1.0,
        min_depth=0,
        max_depth=15,
        alpha_decay=1.,
        eta=1.0,
        children_root_nodes=[], # Root nodes of any children TSSBs
        color="black",
        seed=42,
    ):
        if root_node is None:
            raise Exception("Root node must be specified.")

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.dp_alpha = dp_alpha  # smaller dp_alpha => larger nu => less nodes
        self.dp_gamma = dp_gamma  # smaller dp_gamma => larger psi => less nodes
        self.alpha_decay = alpha_decay
        self.weight = 1.0
        self.children_root_nodes = children_root_nodes
        self.color = color
        self.seed = seed
        self._parent = parent

        self.eta = eta  # wether to put more weight on top or bottom of tree in case we want fixed weights

        self.label = label

        self.assignments = []

        self.root = {
            "node": root_node,
            "main": boundbeta(1.0, dp_alpha, np.random.default_rng(self.seed))
            if self.min_depth == 0
            else 0.0,  # if min_depth > 0, no data can be added to the root (main stick is nu)
            "sticks": empty((0, 1)),  # psi sticks
            "children": [],
            "label": label,
        }
        root_node.tssb = self
        self.root_node = root_node

        self.ntssb = ntssb

        self.data_weights = 0.
        self.unnormalized_data_weights = 0.
        self.elbo = -1e6
        self.ell = -1e6
        self.ew = -1e6
        self.kl = -1e6
        self._data = set()

        self.n_nodes = 1

        self.variational_parameters = {'delta_1': 1., 'delta_2': 1., # nu stick
                                'sigma_1': 1., 'sigma_2': 1., # psi stick
                                'q_c': [], # prob of assigning each cell to this TSSB
                                'LSE_z': [], # normalizing constant for prob of assigning each cell to nodes
                                'LSE_rho': [], # normalizing constant for each child TSSB's probs of assigning each node to nodes in this TSSB (this is a list)
                                'll': [], # auxiliary quantity 
                                'sum_E_log_1_nu': 0., # auxiliary quantity
                                'E_log_phi': 0., # auxiliary quantity
                                }

    def parent(self):
        return self._parent

    def get_param_dict(self):
        """
        Go from a dictionary where each node is a TSSB to a dictionary where each node is a dictionary,
        with `params` and `weight` keys 
        """
        param_dict = {
                "param": self.root['node'].get_params(),
                "mean": self.root['node'].get_mean(),
                "weight": self.root['weight'],
                "children": [],
                "label": self.root['label'],
                "color": self.root['color'],
                "size": len(self.root['node'].data),
                "pivot_probs": self.root['node'].variational_parameters['q_rho'],
        }
        def descend(root, root_new):
            for child in root["children"]:
                child_new = {
                        "param": child['node'].get_params(),
                        "mean": child['node'].get_mean(),
                        "weight": child['weight'],
                        "children": [],
                        "label": child['label'],
                        "color": child['color'],
                        "size": len(child['node'].data),
                        "pivot_probs": child['node'].variational_parameters['q_rho'],
                    }
                root_new['children'].append(child_new)
                descend(child, root_new['children'][-1])
        
        descend(self.root, param_dict)
        return param_dict

    def add_datum(self, id):
        self._data.add(id)

    def remove_data(self):
        self._data.clear()

    def num_data(self):
        def descend(root, n=0):
            for child in root["children"]:
                n = descend(child, n)
            n = n + root["node"].num_local_data()
            return n

        return descend(self.root)

    def clear_data(self):
        def descend(root):
            for index, child in enumerate(root["children"]):
                descend(child)
            root["node"].remove_data()

        self.root["node"].remove_data()
        descend(self.root)
        self.assignments = []
        self.remove_data()

    def reset_tree(self):
        # Clear tree
        self.assignments = []
        self.root_node.remove_data()
        self.root = {
            "node": self.root_node,
            "main": boundbeta(1.0, self.dp_alpha)
            if self.min_depth == 0
            else 0.0,  # if min_depth > 0, no data can be added to the root (main stick is nu)
            "sticks": empty((0, 1)),  # psi sticks
            "children": [],
            "label": self.label,
        }

    def reset_node_parameters(
        self, min_dist=0.7, **node_hyperparams
    ):
        def get_distance(nodeA, nodeB):
            return np.sqrt(np.sum((nodeA.get_mean() - nodeB.get_mean())**2))

        # Reset node parameters
        def descend(root):
            for i, child in enumerate(root["children"]):
                accepted = False
                while not accepted:
                    child["node"].reset_parameters(
                        **node_hyperparams
                    )
                    dist_to_parent = get_distance(root["node"], child["node"])
                    # Reject sample if too close to any other child
                    dists = []
                    for j, child2 in enumerate(root["children"]):
                        if j < i:
                            dists.append(get_distance(child["node"], child2["node"]))
                    if np.all(np.array(dists) >= min_dist*dist_to_parent):
                        accepted = True
                    else:
                        child["node"].seed += 1
                descend(child)

        self.root["node"].reset_parameters(**node_hyperparams)
        descend(self.root)

    def set_node_hyperparams(self, **kwargs):
        def descend(root):
            root['node'].set_node_hyperparams(**kwargs)
            for child in root['children']:
                descend(child)
        descend(self.root)

    def set_weights(self, node_weights_dict):
        def descend(root):
            root['weight'] = node_weights_dict[root['label']]
            for child in root['children']:
                descend(child)
        descend(self.root)

    def set_sticks_from_weights(self):
        def descend(root):
            stick = self.input_tree.get_sum_weights_subtree(child)
            if i < len(input_tree_dict[label]["children"]) - 1:
                sum = 0
                for j, c in enumerate(input_tree_dict[label]["children"][i:]):
                    sum = sum + self.input_tree.get_sum_weights_subtree(c)
                stick = stick / sum
            else:
                stick = 1.0

            sticks = vstack(
                [
                    root['sticks'],
                    stick
                ]
            )
            main = root["weight"]
            subtree_weights_sum = self.input_tree.get_sum_weights_subtree(child)
            main = main / subtree_weights_sum
            root['main'] = main
            root['sticks'] = sticks    
            for child in root['children']:
                descend(child)

    def set_pivot_priors(self):
        def descend(root, depth=0):
            root["node"].pivot_prior_prob = self.eta ** depth
            prior_norm = root["node"].pivot_prior_prob
            for child in root['children']:
                child_prior_norm = descend(child, depth=depth+1)
                prior_norm += child_prior_norm
            return prior_norm
        
        prior_norm = descend(self.root)
        
        def descend(root, depth=0):
            root["node"].pivot_prior_prob = root["node"].pivot_prior_prob/prior_norm
            for child in root['children']:
                descend(child, depth=depth+1)
        
        # Normalize
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

    def get_pivot_probabilities(self, i):
        def descend(root):
            probs = [root['node'].variational_parameters['q_rho'][i]]
            nodes = [root['node']]
            for child in root['children']:
                child_probs, child_nodes = descend(child)
                nodes.extend(child_nodes)
                probs.extend(child_probs)
            return probs, nodes
        return descend(self.root)

    def reset_sufficient_statistics(self, num_batches=1):
        def descend(root):
            root["node"].reset_sufficient_statistics(num_batches=num_batches)
            for child in root["children"]:
                descend(child)
        descend(self.root)
        
        self.suff_stats = {
            'mass': {'total': 0, 'batch': [0.] * num_batches}, # \sum_n q(c_n = this tree)
            'ent': {'total': 0, 'batch': [0.] * num_batches}, # \sum_n q(c_n = this tree) log q(c_n = this tree)
        }

    def reset_variational_parameters(self, alpha_nu=1., beta_nu=1., alpha_psi=1., beta_psi=1., **kwargs):
        # Reset NTSSB parameters relative to this TSSB
        self.variational_parameters = {'delta_1': alpha_nu, 'delta_2': beta_nu, # nu stick
                'sigma_1': alpha_psi, 'sigma_2': beta_psi, # psi stick
                'q_c': jnp.ones(self.ntssb.num_data,) * alpha_nu/(alpha_nu+beta_nu), # prob of assigning each cell to this TSSB
                'LSE_z': jnp.ones(self.ntssb.num_data,), # normalizing constant for prob of assigning each cell to nodes
                'LSE_rho': [1.] * len(self.children_root_nodes), # normalizing constant for each child TSSB's probs of assigning each node to nodes in this TSSB (this is a list)
                'll': jnp.ones(self.ntssb.num_data,), # auxiliary quantity 
                'sum_E_log_1_nu': 0., # auxiliary quantity
                'E_log_phi': 0., # auxiliary quantity
                }
                
        # Reset node parameters
        def descend(root):
            root["node"].reset_variational_parameters(**kwargs)
            z_norm = jnp.array(root["node"].variational_parameters['q_z'])
            rho_norm = jnp.array(root["node"].variational_parameters['q_rho'])
            for child in root["children"]:
                child_z_norm, child_rho_norm = descend(child)
                z_norm += child_z_norm
                rho_norm += child_rho_norm
            return z_norm, rho_norm

        self.root["node"]._parent = None

        # Get normalizing constants
        z_norm, rho_norm = descend(self.root)

        # Apply normalization
        def descend(root):
            root["node"].reset_variational_parameters(**kwargs)
            root["node"].variational_parameters['q_z'] = root["node"].variational_parameters['q_z'] / z_norm
            root["node"].variational_parameters['q_rho'] = list(root["node"].variational_parameters['q_rho'] / rho_norm)
            for child in root["children"]:
                descend(child)
        descend(self.root)


    def sample_new_tree(self, num_data, cull=True):
        # Clear current tree
        self.reset_tree()

        # Break sticks to assign data
        for n in range(num_data):
            u = rand()
            (new_node, new_path, _) = self.find_node(u)
            new_node.add_datum(n)
            self.assignments.append(new_node)

        if cull:
            # Remove empty leaf nodes
            self.cull_tree()

    def get_ntssb_root(self):
        ntssb = self.ntssb

        def descend(root):
            if root["node"] == self:
                return root
            for child in root["children"]:
                out = descend(child)
                if out:
                    return out

        return descend(ntssb.root)

    def get_n_nodes(self):
        def descend(root, n_nodes=0):
            for child in root["children"]:
                n_nodes = descend(child, n_nodes)
            n_nodes = n_nodes + 1
            return n_nodes

        return descend(self.root)

    def get_n_branches(self):
        def descend(root, n_branches=0):
            for child in root["children"]:
                n_branches = descend(child, n_branches)
            n_branches = n_branches + len(root["children"])
            return n_branches

        return descend(self.root)

    def get_n_levels(self):
        def descend(root, n_levels_list=[], n_levels=1):
            for child in root["children"]:
                descend(child, n_levels_list, n_levels + 1)
            n_levels_list.append(n_levels)
            return n_levels_list

        return max(descend(self.root))

    def get_node_depths(self):
        def descend(root, n_levels_list=[], n_levels=1):
            for child in root["children"]:
                descend(child, n_levels_list, n_levels + 1)
            n_levels_list.append((root["node"], n_levels))
            return n_levels_list

        return descend(self.root)

    # def get_mean_n_branches_per_level(self):

    def truncate(self):
        # Sets main sticks of leaf nodes last psi sticks to 1
        def descend(root, depth=0):
            for i, child in enumerate(root["children"]):
                root["sticks"][i] = (
                    1.0 if i == len(root["children"]) - 1 else root["sticks"][i]
                )
                descend(child, depth + 1)
            if len(root["children"]) < 1:
                root["main"] = 1.0

        descend(self.root)

    def add_data(self, data, initialize_assignments=False):
        self.data = data
        self.num_data = 0 if data is None else data.shape[0]

        self.assignments = []
        for n in range(self.num_data):
            # randomly find a node and assign
            u = rand()
            node, path, _ = self.find_node(u)
            node.add_datum(n)
            self.assignments.append(node)

    def add_node(self, target_root, seed=None):
        if seed is None:
            seed = self.seed + target_root["node"].seed + len(target_root["children"])

        node = target_root["node"].spawn(target_root["node"].observed_parameters, seed=seed)

        rng = np.random.default_rng(seed)
        stick_length = boundbeta(1, self.dp_gamma, rng)
        target_root["sticks"] = np.vstack([target_root["sticks"], stick_length])
        target_root["children"].append(
            {
                "node": node,
                "main": boundbeta(
                    1.0, (self.alpha_decay ** (target_root["node"].depth + 1)) * self.dp_alpha, np.random.default_rng(seed+1)
                )
                if self.min_depth <= (target_root["node"].depth + 1)
                else 0.0,
                "sticks": np.empty((0, 1)),
                "children": [],
                "label": node.label,
            }
        )

        # Update pivot prior
        self.set_pivot_priors()

        self.n_nodes += 1

        return node

    def merge_nodes(self, parent_root, source_root, target_root):
        # Add mass of source to mass of target
        target_root['node'].variational_parameters['q_z'] += source_root['node'].variational_parameters['q_z']
        # Only need to update totals, because after the merges we go back to iterate
        target_root['node'].merge_suff_stats(source_root['node'].suff_stats)

        # Update pivot probs
        for i in range(len(self.children_root_nodes)):
            target_root['node'].variational_parameters['q_rho'][i] += source_root['node'].variational_parameters['q_rho'][i]

        # Set children of source as children of target
        for i, child in enumerate(source_root['children']):
            target_root['children'].append(child)
            target_root["sticks"] = np.vstack([target_root["sticks"], source_root['sticks'][i]])
            child['node'].set_parent(target_root['node'])

        # Remove source from its parent's dict
        nodes = np.array([n["node"] for n in parent_root["children"]])
        tokeep = np.where(nodes != source_root['node'])[0].astype(int).ravel()
        parent_root["sticks"] = parent_root["sticks"][tokeep]
        parent_root["children"] = list(np.array(parent_root["children"])[tokeep])
        
        # Delete source node object and dict
        source_root["node"].kill()
        del source_root["node"]

        # Update names
        self.set_node_names(root=parent_root, root_name=parent_root['node'].label)

        # Update pivot prior
        self.set_pivot_priors()

        self.n_nodes -= 1

    def compute_elbo(self, idx):
        """
        Compute the ELBO of the model in a tree traversal, abstracting away the likelihood and kernel specific functions
        for the model. The seed is used for MC sampling from the variational distributions for which Eq[logp] is not analytically
        available (which is the likelihood and the kernel distribution).
        """
        def descend(root, depth=0, ll_contrib=0, ass_contrib=0, global_contrib=0):
            self.n_nodes += 1
            # Assignments
            ## E[log p(z|nu,psi))] 
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            eq_logp_z = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            if root['node'].parent() is not None:
                eq_logp_z += root['node'].parent().variational_parameters['sum_E_log_1_nu']
                eq_logp_z += root['node'].variational_parameters['E_log_phi']
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu + root['node'].parent().variational_parameters['sum_E_log_1_nu']
            else:
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu
            ## E[log q(z)]
            eq_logq_z = jax.lax.select(root['node'].variational_parameters['q_z'][idx] != 0, 
                                       root['node'].variational_parameters['q_z'][idx] * jnp.log(root['node'].variational_parameters['q_z'][idx]), 
                                       root['node'].variational_parameters['q_z'][idx])
            ass_contrib += eq_logp_z*root['node'].variational_parameters['q_z'][idx] - eq_logq_z

            # Sticks
            E_log_nu = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl = (self.dp_alpha * self.alpha_decay**depth - root['node'].variational_parameters['delta_2']) * E_log_1_nu
            nu_kl -= (root['node'].variational_parameters['delta_1'] - 1) * E_log_nu
            nu_kl += logbeta_func(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl -= logbeta_func(1, self.dp_alpha * self.alpha_decay**depth)
            psi_kl = 0.
            if depth != 0:
                E_log_psi = E_log_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                E_log_1_psi = E_log_1_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl = (self.dp_gamma - root['node'].variational_parameters['sigma_2']) * E_log_1_psi
                psi_kl -= (root['node'].variational_parameters['sigma_1'] - 1) * E_log_psi
                psi_kl += logbeta_func(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl -= logbeta_func(1, self.dp_gamma)      
            global_contrib += nu_kl + psi_kl
            
            # Kernel
            if root['node'].parent() is None and self.parent() is None: # is the root of the root TSSB
                ## E[log p(kernel)]
                eq_logp_kernel = root['node'].compute_root_prior()
                ## -E[log q(kernel)]
                negeq_logq_kernel = root['node'].compute_root_entropy()
            else:
                if root['node'].parent() is not None: # Not a TSSB root. The root param probs are computed in the parent TSSBs, weighted by the pivots
                    ## E[log p(kernel)]
                    eq_logp_kernel = root['node'].compute_kernel_prior()
                else:
                    eq_logp_kernel = 0.
                ## -E[log q(kernel)]
                negeq_logq_kernel = root['node'].compute_kernel_entropy()
            global_contrib += eq_logp_kernel + negeq_logq_kernel

            # Pivots
            eq_logp_rootkernel = 0.
            eq_logp_rho = 0.
            eq_logq_rho = 0.
            for i, next_tssb_root_node in enumerate(self.children_root_nodes):
                ## E[log p(root kernel | rho kernel)]
                eq_logp_rootkernel += root['node'].variational_parameters['q_rho'][i] * next_tssb_root_node.compute_root_kernel_prior(root['node'].samples)
                ## E[log p(rho))]
                eq_logp_rho += root['node'].pivot_prior_prob
                ## E[log q(rho))]
                eq_logq_rho = jax.lax.select(root['node'].variational_parameters['q_rho'][i] != 0, 
                        root['node'].variational_parameters['q_rho'][i] * jnp.log(root['node'].variational_parameters['q_rho'][i]), 
                        root['node'].variational_parameters['q_rho'][i])
            global_contrib += eq_logp_rootkernel + eq_logp_rho-eq_logq_rho

            # Likelihood
            # Use node's kernel sample to evaluate likelihood
            ll_contrib += root['node'].compute_loglikelihood(idx) * root['node'].variational_parameters['q_z'][idx]

            sum_E_log_1_psi = 0.
            for child in root['children']:
                # Auxiliary quantities
                E_log_psi = E_log_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                child['node'].variational_parameters['E_log_phi'] = E_log_psi + sum_E_log_1_psi
                E_log_1_psi = E_log_1_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                sum_E_log_1_psi += E_log_1_psi

                # Go down
                ll_contrib, ass_contrib, global_contrib = descend(child, depth=depth+1, 
                                                                  ll_contrib=ll_contrib, 
                                                                  ass_contrib=ass_contrib, 
                                                                  global_contrib=global_contrib)

            return ll_contrib, ass_contrib, global_contrib

        self.n_nodes = 0
        return descend(self.root)

    def compute_elbo_suff(self):
        """
        Compute the ELBO of the model in a tree traversal, abstracting away the likelihood and kernel specific functions
        for the model. The seed is used for MC sampling from the variational distributions for which Eq[logp] is not analytically
        available (which is the likelihood and the kernel distribution).
        """
        def descend(root, depth=0, ll_contrib=0, ass_contrib=0, global_contrib=0):
            self.n_nodes += 1
            # Assignments
            ## E[log p(z|nu,psi))] 
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            eq_logp_z = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            if root['node'].parent() is not None:
                eq_logp_z += root['node'].parent().variational_parameters['sum_E_log_1_nu']
                eq_logp_z += root['node'].variational_parameters['E_log_phi']
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu + root['node'].parent().variational_parameters['sum_E_log_1_nu']
            else:
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu
            ## E[log q(z)]
            ass_contrib += eq_logp_z*root['node'].suff_stats['mass']['total'] + root['node'].suff_stats['ent']['total']

            # Sticks
            E_log_nu = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl = (self.dp_alpha * self.alpha_decay**depth - root['node'].variational_parameters['delta_2']) * E_log_1_nu
            nu_kl -= (root['node'].variational_parameters['delta_1'] - 1) * E_log_nu
            nu_kl += logbeta_func(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            nu_kl -= logbeta_func(1, self.dp_alpha * self.alpha_decay**depth)
            psi_kl = 0.
            if depth != 0:
                E_log_psi = E_log_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                E_log_1_psi = E_log_1_beta(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl = (self.dp_gamma - root['node'].variational_parameters['sigma_2']) * E_log_1_psi
                psi_kl -= (root['node'].variational_parameters['sigma_1'] - 1) * E_log_psi
                psi_kl += logbeta_func(root['node'].variational_parameters['sigma_1'], root['node'].variational_parameters['sigma_2'])
                psi_kl -= logbeta_func(1, self.dp_gamma)      
            global_contrib += nu_kl + psi_kl
            
            # Kernel
            if root['node'].parent() is None and self.parent() is None: # is the root of the root TSSB
                ## E[log p(kernel)]
                eq_logp_kernel = root['node'].compute_root_prior()
                ## -E[log q(kernel)]
                negeq_logq_kernel = root['node'].compute_root_entropy()
            else:
                if root['node'].parent() is not None: # Not a TSSB root. The root param probs are computed in the parent TSSBs, weighted by the pivots
                    ## E[log p(kernel)]
                    eq_logp_kernel = root['node'].compute_kernel_prior()
                else:
                    eq_logp_kernel = 0.
                ## -E[log q(kernel)]
                negeq_logq_kernel = root['node'].compute_kernel_entropy()
            global_contrib += eq_logp_kernel + negeq_logq_kernel

            # Pivots
            eq_logp_rootkernel = 0.
            eq_logp_rho = 0.
            eq_logq_rho = 0.
            for i, next_tssb_root_node in enumerate(self.children_root_nodes):
                ## E[log p(root kernel | rho kernel)]
                eq_logp_rootkernel += root['node'].variational_parameters['q_rho'][i] * next_tssb_root_node.compute_root_kernel_prior(root['node'].samples)
                ## E[log p(rho))]
                eq_logp_rho += root['node'].pivot_prior_prob
                ## E[log q(rho))]
                eq_logq_rho = jax.lax.select(root['node'].variational_parameters['q_rho'][i] != 0, 
                        root['node'].variational_parameters['q_rho'][i] * jnp.log(root['node'].variational_parameters['q_rho'][i]), 
                        root['node'].variational_parameters['q_rho'][i])
            global_contrib += eq_logp_rootkernel + eq_logp_rho-eq_logq_rho

            # Likelihood
            # Use node's kernel sample to evaluate likelihood
            ll_contrib += root['node'].compute_loglikelihood_suff()

            sum_E_log_1_psi = 0.
            for child in root['children']:
                # Auxiliary quantities
                E_log_psi = E_log_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                child['node'].variational_parameters['E_log_phi'] = E_log_psi + sum_E_log_1_psi
                E_log_1_psi = E_log_1_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                sum_E_log_1_psi += E_log_1_psi

                # Go down
                ll_contrib, ass_contrib, global_contrib = descend(child, depth=depth+1, 
                                                                  ll_contrib=ll_contrib, 
                                                                  ass_contrib=ass_contrib, 
                                                                  global_contrib=global_contrib)

            return ll_contrib, ass_contrib, global_contrib

        self.n_nodes = 0
        return descend(self.root)


    def update_sufficient_statistics(self, batch_idx=None):
        def descend(root):
            root['node'].update_sufficient_statistics(batch_idx=batch_idx)
            for child in root['children']:
                descend(child)
        
        descend(self.root)
        
        if batch_idx is not None:
            idx = self.ntssb.batch_indices[batch_idx]
        else:
            idx = jnp.arange(self.ntssb.num_data)
        
        ent = assignment_entropies(self.variational_parameters['q_c'][idx])
        E_ass = self.variational_parameters['q_c'][idx]
        
        new_ent = jnp.sum(ent)
        new_mass = jnp.sum(E_ass)

        if batch_idx is not None:
            self.suff_stats['ent']['total'] -= self.suff_stats['ent']['batch'][batch_idx]
            self.suff_stats['ent']['batch'][batch_idx] = new_ent
            self.suff_stats['ent']['total'] += self.suff_stats['ent']['batch'][batch_idx]

            self.suff_stats['mass']['total'] -= self.suff_stats['mass']['batch'][batch_idx]
            self.suff_stats['mass']['batch'][batch_idx] = new_mass
            self.suff_stats['mass']['total'] += self.suff_stats['mass']['batch'][batch_idx]
        else:
            self.suff_stats['ent']['total'] = new_ent
            self.suff_stats['mass']['total'] = new_mass

    def update_local_params(self, idx, update_ass=True, take_gradients=False, **kwargs):
        """
        This performs a tree traversal to update the cell to node attachment probabilities. 
        Returns \sum_node Eq[logp(y_n|psi_node)] * q(z_n=node) with the updated q(z_n=node)
        """
        def descend(root, local_grads=None):
            E_log_1_nu = E_log_1_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            logprior = E_log_beta(root['node'].variational_parameters['delta_1'], root['node'].variational_parameters['delta_2'])
            if root['node'].parent() is not None:
                logprior += root['node'].parent().variational_parameters['sum_E_log_1_nu']
                logprior += root['node'].variational_parameters['E_log_phi']
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu + root['node'].parent().variational_parameters['sum_E_log_1_nu']
            else:
                root['node'].variational_parameters['sum_E_log_1_nu'] = E_log_1_nu

            # Take gradient of locals
            weights = root['node'].variational_parameters['q_z'][idx] * self.variational_parameters['q_c'][idx]
            if take_gradients:
                local_grads_down = root["node"].compute_ll_locals_grad(self.ntssb.data[idx], idx, weights) # returns a tuple of grads for each cell-specific param
                if local_grads is None:
                    local_grads = list(local_grads_down)
                else:
                    for i, grads in enumerate(list(local_grads_down)):
                        local_grads[i] += grads
            ll = root['node'].compute_loglikelihood(idx)
            root['node'].variational_parameters['ll'] = ll
            root['node'].variational_parameters['logprior'] = logprior
            new_log_prob = ll + logprior
            if update_ass:
                root['node'].variational_parameters['q_z'] = root['node'].variational_parameters['q_z'].at[idx].set(new_log_prob)

            logqs = [new_log_prob]
            sum_E_log_1_psi = 0.
            for child in root['children']:
                E_log_psi = E_log_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                child['node'].variational_parameters['E_log_phi'] = E_log_psi + sum_E_log_1_psi
                E_log_1_psi = E_log_1_beta(child['node'].variational_parameters['sigma_1'], child['node'].variational_parameters['sigma_2'])
                sum_E_log_1_psi += E_log_1_psi

                # Go down
                child_log_probs, child_local_param_grads = descend(child, local_grads=local_grads)
                logqs.extend(child_log_probs)
                if child_local_param_grads is not None:
                    for i, grads in enumerate(list(child_local_param_grads)):
                        local_grads[i] += grads
                # local_grads += child_local_param_grads
            
            return logqs, local_grads
                
        logqs, local_grads = descend(self.root)
        
        # Compute LSE
        logqs = jnp.array(logqs).T # make it obs by nodes
        self.variational_parameters['LSE_z'] = jax.scipy.special.logsumexp(logqs, axis=1)
        # Set probs and return sum of weighted likelihoods
        def descend(root):
            if update_ass:
                newvals = jnp.exp(root['node'].variational_parameters['q_z'][idx] - self.variational_parameters['LSE_z'])
                root['node'].variational_parameters['q_z'] = root['node'].variational_parameters['q_z'].at[idx].set(newvals)
            ell = (root['node'].variational_parameters['ll'] + root['node'].variational_parameters['logprior']) * root['node'].variational_parameters['q_z'][idx]
            ent = -jax.lax.select(root['node'].variational_parameters['q_z'][idx] != 0, 
                    root['node'].variational_parameters['q_z'][idx] * jnp.log(root['node'].variational_parameters['q_z'][idx]), 
                    root['node'].variational_parameters['q_z'][idx])

            for child in root['children']:
                ell_, ent_ = descend(child)
                ell += ell_
                ent += ent_
            return ell, ent
        
        ell, ent = descend(self.root)
        return ell, ent, local_grads
    
    def get_global_grads(self, idx):
        """
        This performs a tree traversal to update the global parameters
        """
        def descend(root, globals_grads=None):
            weights = root['node'].variational_parameters['q_z'][idx] * self.variational_parameters['q_c'][idx]
            globals_grads_down = root["node"].compute_ll_globals_grad(self.ntssb.data[idx], idx, weights)
            if globals_grads is None:
                globals_grads = list(globals_grads_down)
            else:
                for i, grads in enumerate(list(globals_grads_down)):
                    globals_grads[i] += grads
            for child in root['children']:
                child_globals_grads = descend(child, globals_grads=globals_grads)
                for i, grads in enumerate(list(child_globals_grads)):
                    globals_grads[i] += grads
            return globals_grads
        
        # Get gradient of loss of data likelihood weighted by assignment probability to each node wrt current sample of global params
        return descend(self.root)

    def update_stick_params(self, root=None, memoized=True):
        def descend(root, depth=0):
            mass_down = 0
            for child in root['children'][::-1]:
                child_mass = descend(child, depth=depth+1)

                child['node'].variational_parameters['sigma_1'] = 1.0 + child_mass
                child['node'].variational_parameters['sigma_2'] = self.dp_gamma + mass_down
                mass_down += child_mass

            if memoized:
                mass_here = root['node'].suff_stats['mass']['total']
            else:
                mass_here = jnp.sum(root['node'].variational_parameters['q_z'] * self.variational_parameters['q_c'])
            root['node'].variational_parameters['delta_1'] = 1.0 + mass_here
            root['node'].variational_parameters['delta_2'] = (self.alpha_decay**depth) * self.dp_alpha + mass_down

            return mass_here + mass_down

        # Update sticks
        if root is None:
            root = self.root
        descend(root)

    def update_node_params(self, key, root=None, memoized=True, step_size=0.0001, mc_samples=10, i=0, adaptive=True, **kwargs):
        """
        Update variational parameters for kernels, sticks and pivots

        Each node must have two parameters for the kernel: a direction and a state.
        We assume the tree kernel, regardless of the model, is always defined as 
        P(direction|parent_direction) and P(state|direction,parent_state).
        For each node, we first update the direction and then the state, taking one gradient step for each
        parameter and then moving on to the next nodes in the tree traversal

        PSEUDOCODE:
        def descend(root):
            alpha, alpha_grad = sample_grad_alpha
            psi, psi_grad = sample_grad_psi

            alpha_grad += Gradient of logp(alpha|parent_alpha) wrt this alpha 
            alpha_grad += Gradient of logp(psi|parent_psi,alpha) wrt this alpha

            alpha_grad += Gradient of logq(alpha) wrt this alpha
            psi_grad += Gradient of logq(psi) wrt this psi

            psi_grad += Gradient of logp(x|psi) wrt this psi

            for each child:
                child_alpha, child_psi = descend(child)
                alpha_grad += Gradient of logp(child_alpha|alpha) wrt this alpha
                psi_grad += Gradient of logp(child_psi|psi,child_alpha) wrt this psi

            for each child_root:
                alpha_grad += Gradient of logp(child_root_alpha|alpha) wrt this alpha
                psi_grad += Gradient of logp(child_root_psi|psi,child_root_alpha) wrt this psi

            new_alpha_params = alpha_params + alpha_grad * step_size
            new_alpha = sample_alpha

            psi_grad += Gradient of logp(psi|parent_psi,new_alpha) wrt this psi
            new_psi_params = psi_params + psi_grad * step_size
            new_psi = sample_psi

            return new_alpha, new_psi
        
        """
        def descend(root, key, depth=0):
            direction_sample_grad = 0.
            state_sample_grad = 0.

            if depth != 0:
                key, sample_grad = root['node'].direction_sample_and_grad(key, n_samples=mc_samples)
                direction_curr_sample, direction_params_grad = sample_grad
                key, sample_grad = root['node'].state_sample_and_grad(key, n_samples=mc_samples)
                state_curr_sample, state_params_grad = sample_grad
            else:
                root['node'].sample_kernel(n_samples=mc_samples)
                direction_curr_sample = root['node'].get_direction_sample()
                state_curr_sample = root['node'].get_state_sample()
            
            if depth != 0:
                direction_parent_sample = root["node"].parent().get_direction_sample()
                state_parent_sample = root["node"].parent().get_state_sample()

                direction_sample_grad += root["node"].compute_direction_prior_grad(direction_curr_sample, direction_parent_sample, state_parent_sample)
                direction_sample_grad += root["node"].compute_state_prior_grad_wrt_direction(state_curr_sample, state_parent_sample, direction_curr_sample)
                
                direction_params_entropy_grad = root["node"].compute_direction_entropy_grad()
                state_params_entropy_grad = root["node"].compute_state_entropy_grad()

                if memoized:
                    state_sample_grad += root["node"].compute_ll_state_grad_suff(state_curr_sample)
                else:
                    weights = root['node'].variational_parameters['q_z'] * self.variational_parameters['q_c']
                    state_sample_grad += root["node"].compute_ll_state_grad(self.ntssb.data, weights, state_curr_sample)

            mass_down = 0
            for child in root['children'][::-1]:
                child_mass, direction_child_sample, state_child_sample = descend(child, key, depth=depth+1)

                child['node'].variational_parameters['sigma_1'] = 1.0 + child_mass
                child['node'].variational_parameters['sigma_2'] = self.dp_gamma + mass_down
                mass_down += child_mass

                if depth != 0:
                    direction_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_direction(direction_child_sample, direction_curr_sample, state_curr_sample)
                    state_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_state(direction_child_sample, direction_curr_sample, state_curr_sample)
                    state_sample_grad += root["node"].compute_state_prior_child_grad(state_child_sample, state_curr_sample, direction_child_sample)
            
            if depth != 0:
                for ii, child_root in enumerate(self.children_root_nodes):
                    direction_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_direction(child_root.get_direction_sample(), direction_curr_sample, state_curr_sample) * root['node'].variational_parameters['q_rho'][ii]
                    state_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_state(child_root.get_direction_sample(), direction_curr_sample, state_curr_sample) * root['node'].variational_parameters['q_rho'][ii]
                    state_sample_grad += root["node"].compute_root_state_prior_child_grad(child_root.get_state_sample(), state_curr_sample, child_root.get_direction_sample()) * root['node'].variational_parameters['q_rho'][ii]

            if depth != 0:
                if adaptive and i == 0:
                    root['node'].reset_opt()
            
                # Combine gradients of functions wrt sample with gradient of sample wrt var params
                if adaptive:
                    root['node'].update_direction_adaptive(direction_params_grad, direction_sample_grad, direction_params_entropy_grad, 
                                                    step_size=step_size, i=i)
                else:
                    root['node'].update_direction_params(direction_params_grad, direction_sample_grad, direction_params_entropy_grad, 
                                                        step_size=step_size)
                key, sample_grad = root['node'].direction_sample_and_grad(key, n_samples=mc_samples)
                direction_curr_sample, _ = sample_grad

                state_sample_grad += root["node"].compute_state_prior_grad(state_curr_sample, state_parent_sample, direction_curr_sample)
                
                if adaptive:
                    root['node'].update_state_adaptive(state_params_grad, state_sample_grad, state_params_entropy_grad, 
                                                        step_size=step_size, i=i)    
                else:
                    root['node'].update_state_params(state_params_grad, state_sample_grad, state_params_entropy_grad, 
                                                        step_size=step_size)
                key, sample_grad = root['node'].state_sample_and_grad(key, n_samples=mc_samples)
                state_curr_sample, _ = sample_grad
                root['node'].samples[0] = state_curr_sample
                root['node'].samples[1] = direction_curr_sample

            if memoized:
                mass_here = root['node'].suff_stats['mass']['total']
            else:
                mass_here = jnp.sum(root['node'].variational_parameters['q_z'] * self.variational_parameters['q_c'])
            root['node'].variational_parameters['delta_1'] = 1.0 + mass_here
            root['node'].variational_parameters['delta_2'] = (self.alpha_decay**depth) * self.dp_alpha + mass_down

            return mass_here + mass_down, direction_curr_sample, state_curr_sample

        # Update kernels and sticks
        if root is None:
            root = self.root
        descend(root, key)


    def sample_grad_root_node(self, key, memoized=True, mc_samples=10, **kwargs):
        """
        B->B0 --> C
        Compute gradient of p(stateB0|dirB0,stateB) wrt stateB, p(dirB0|dirB,stateB) wrt dirB, stateB
        and gradient p(stateC|dirC,stateB) wrt stateB, p(dirC|dirB,stateB) wrt dirB, stateB
        """
        # Sample root params and compute initial gradients wrt children
        root = self.root
        direction_sample_grad = 0.
        state_sample_grad = 0.

        key, sample_grad = root['node'].direction_sample_and_grad(key, n_samples=mc_samples)
        direction_curr_sample, direction_params_grad = sample_grad
        key, sample_grad = root['node'].state_sample_and_grad(key, n_samples=mc_samples)
        state_curr_sample, state_params_grad = sample_grad
        
        # Gradient of entropy
        direction_params_entropy_grad = root["node"].compute_direction_entropy_grad()
        state_params_entropy_grad = root["node"].compute_state_entropy_grad()

        # Gradient of likelihood
        if memoized:
            ll_grad = root["node"].compute_ll_state_grad_suff(state_curr_sample)
        else:
            weights = root['node'].variational_parameters['q_z'] * self.variational_parameters['q_c']
            ll_grad = root["node"].compute_ll_state_grad(self.ntssb.data, weights, state_curr_sample)

        # Gradient of children in TSSB
        for child in root['children'][::-1]:
            direction_child_sample = child['node'].get_direction_sample()
            state_child_sample = child['node'].get_state_sample()
            direction_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_direction(direction_child_sample, direction_curr_sample, state_curr_sample)
            state_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_state(direction_child_sample, direction_curr_sample, state_curr_sample)
            state_sample_grad += root["node"].compute_state_prior_child_grad(state_child_sample, state_curr_sample, direction_child_sample)

        # Gradient of roots of children TSSB
        for i, child_root in enumerate(self.children_root_nodes):
            direction_child_sample = child_root.get_direction_sample()
            state_child_sample = child_root.get_state_sample()
            # Gradient of the root nodes of children TSSBs wrt to their parameters using this TSSB root as parent
            direction_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_direction(direction_child_sample, direction_curr_sample, state_curr_sample) * root['node'].variational_parameters['q_rho'][i]
            state_sample_grad += root["node"].compute_direction_prior_child_grad_wrt_state(direction_child_sample, direction_curr_sample, state_curr_sample) * root['node'].variational_parameters['q_rho'][i]
            state_sample_grad += root["node"].compute_state_prior_child_grad(state_child_sample, state_curr_sample, direction_child_sample) * root['node'].variational_parameters['q_rho'][i]

        direction_locals_grads = [direction_params_grad, direction_params_entropy_grad]
        state_locals_grads = [state_params_grad, state_params_entropy_grad]

        return ll_grad, [direction_locals_grads, state_locals_grads], [direction_sample_grad, state_sample_grad]

    def compute_children_root_node_grads(self, **kwargs):
        """
        A -> B
        Compute gradient of p(dirB|dirA,stateA) wrt dirB and p(stateB|dirB,stateA) wrt stateB, dirB
        """
        def descend(root, children_grads=None):
            direction_curr_sample = root['node'].samples[1]
            state_curr_sample = root['node'].samples[0]
        
            # Compute gradient of children roots wrt their params
            if children_grads is None:
                children_grads = [[0., 0.]] * len(self.children_root_nodes)
            for i, child_root in enumerate(self.children_root_nodes):
                # Gradient of the root nodes of children TSSBs wrt to their parameters using this 
                direction_child_sample = child_root.get_direction_sample()
                state_child_sample = child_root.get_state_sample()
                direction_sample_grad = root["node"].compute_direction_prior_grad(direction_child_sample, direction_curr_sample, state_curr_sample) * root['node'].variational_parameters['q_rho'][i]
                direction_sample_grad += root["node"].compute_state_prior_grad_wrt_direction(state_child_sample, state_curr_sample, direction_child_sample) * root['node'].variational_parameters['q_rho'][i]
                state_sample_grad = root["node"].compute_state_prior_grad(state_child_sample, state_curr_sample, direction_child_sample) * root['node'].variational_parameters['q_rho'][i]
                children_grads[i][0] += direction_sample_grad
                children_grads[i][1] += state_sample_grad

            for child in root['children']:
                descend(child, children_grads=children_grads)

            return children_grads
        
        # Return list of children, for each child a tuple with direction grad sum and sample grad sum
        return descend(self.root)
    

    def update_root_node_params(self, key, ll_grad, local_grads, children_grads, parent_grads, i=0, adaptive=True, step_size=0.0001, mc_samples=10, **kwargs):
        """
        ll_grads: ll_grad_state_D
        local_grads: params_grad_D, params_entropy_grad_D
        children_grads: grad_D p(D0|D) + grad_D p(D|D)
        parent_grads: grad_D p(D|B) + grad_D p(D|B0)
        """
        if adaptive and i == 0:
            self.root['node'].reset_opt()

        direction_params_grad, direction_params_entropy_grad = local_grads[0]
        direction_sample_grad = children_grads[0] + parent_grads[0]

        # Combine gradients of functions wrt sample with gradient of sample wrt var params
        if adaptive:
            self.root['node'].update_direction_adaptive(direction_params_grad, direction_sample_grad, direction_params_entropy_grad, 
                                                step_size=step_size, i=i)
        else:
            self.root['node'].update_direction_params(direction_params_grad, direction_sample_grad, direction_params_entropy_grad, 
                                                step_size=step_size)
        key, sample_grad = self.root['node'].direction_sample_and_grad(key, n_samples=mc_samples)
        direction_curr_sample, _ = sample_grad
        self.root['node'].samples[1] = direction_curr_sample
        
        state_params_grad, state_params_entropy_grad = local_grads[1]
        state_sample_grad = children_grads[1] + parent_grads[1]
        state_sample_grad += ll_grad
        
        if adaptive:
            self.root['node'].update_state_adaptive(state_params_grad, state_sample_grad, state_params_entropy_grad, 
                                                step_size=step_size, i=i)
        else:
            self.root['node'].update_state_params(state_params_grad, state_sample_grad, state_params_entropy_grad, 
                                                step_size=step_size)
        key, sample_grad = self.root['node'].state_sample_and_grad(key, n_samples=mc_samples)
        state_curr_sample, _ = sample_grad
        self.root['node'].samples[0] = state_curr_sample
        

    def update_pivot_probs(self, **kwargs):
        def descend(root):
            # Update pivot assignment probabilities
            new_log_probs = []
            root['node'].variational_parameters['q_rho'] = [0.] * len(self.children_root_nodes)
            for i, child_root in enumerate(self.children_root_nodes):
                pivot_direction_score = child_root.compute_root_direction_prior(root['node'].get_direction_sample())
                pivot_state_score = child_root.compute_root_state_prior(root['node'].get_state_sample())
                new_log_prob = pivot_direction_score + pivot_state_score + jnp.log(root['node'].pivot_prior_prob)
                root['node'].variational_parameters['q_rho'][i] = new_log_prob
                new_log_probs.append([new_log_prob])
            
            for child in root['children']:
                children_log_probs = descend(child)
                for i, child_root in enumerate(self.children_root_nodes):
                    new_log_probs[i].extend(children_log_probs[i])
            
            return new_log_probs

        logqs = descend(self.root)
        # Compute LSE for pivot probs
        for i in range(len(self.children_root_nodes)):
            this_logqs = jnp.array(logqs[i])
            self.variational_parameters['LSE_rho'][i] = jax.scipy.special.logsumexp(this_logqs)

        # Set probs and return sum of weighted likelihoods
        def descend(root):
            for i, child_root in enumerate(self.children_root_nodes):
                root['node'].variational_parameters['q_rho'][i] = jnp.exp(root['node'].variational_parameters['q_rho'][i] - self.variational_parameters['LSE_rho'][i])
            for child in root['children']:
                descend(child)

        # Normalize pivot probs
        descend(self.root)
    
    def cull_tree(self, verbose=False, resample_sticks=True):
        """
        If a leaf node has no data assigned to it, remove it
        """
        culled = []

        def descend(root):
            counts = array(list(map(lambda child: descend(child), root["children"])))

            tokeep = np.where(counts != 0)[0].astype(int).ravel()
            tokill = np.where(counts == 0)[0].astype(int).ravel()

            if len(tokill) > 0:
                for child in list(array(root["children"])[tokill]):
                    logger.debug(f"Removing {child['node'].label}")
                    culled.append(child["node"])
                    child["node"].kill()
                    del child["node"]

            root["sticks"] = root["sticks"][tokeep]
            root["children"] = list(array(root["children"])[tokeep])

            s = 0
            if (
                len(list(root["node"].children())) > 0
            ):  # make sure we only delete leaves
                s = 1
            return sum(counts) + root["node"].num_local_data() + s

        descend(self.root)
        if len(culled) > 0 and resample_sticks:
            self.resample_sticks()
        return culled

    def get_subtree_depth(self, root):
        def descend(root, n_levels_list=[], n_levels=1):
            for child in root["children"]:
                descend(child, n_levels_list, n_levels + 1)
            n_levels_list.append(n_levels)
            return n_levels_list

        return max(descend(root))

    def sample_main_sticks(self, truncate):
        def descend(root, depth=0):
            main = (
                boundbeta(1.0, (self.alpha_decay ** (depth + 1)) * self.dp_alpha)
                if self.min_depth <= (depth + 1)
                else 0.0
            )
            root["main"] = 1.0 if len(root["children"]) < 1 and truncate else main
            for i, child in enumerate(root["children"]):
                descend(child, depth + 1)

        descend(self.root)

    def sample_sticks(self, truncate, balanced):
        def descend(root, depth=0):
            for i, child in enumerate(root["children"]):
                stick = boundbeta(1, self.dp_gamma)
                if balanced:
                    stick = 1.0 / (len(root["children"]) - i)
                root["sticks"][i] = (
                    1.0 if i == len(root["children"]) - 1 and truncate else stick
                )
                descend(child, depth + 1)

        descend(self.root)

    def balance_sticks(self, truncate):
        def descend(root, depth=0):
            for i, child in enumerate(root["children"]):
                stick = 1.0 / (len(root["children"]) - i)
                root["sticks"][i] = (
                    1.0 if i == len(root["children"]) - 1 and truncate else stick
                )
                descend(child, depth + 1)

        descend(self.root)

    def resample_sticks(self, top_node=None, prior=False, truncate=False):
        def descend(root, depth=0):
            data_down = 0
            indices = list(range(len(root["children"])))
            # indices.reverse()
            indices = indices[::-1]
            for i in indices:
                child = root["children"][i]
                child_data = descend(child, depth + 1)
                post_alpha = 1.0 + child_data
                post_beta = self.dp_gamma + data_down
                if prior:
                    root["sticks"][i] = boundbeta(
                        1.0, self.dp_gamma
                    )  # Updates psi-sticks
                else:
                    root["sticks"][i] = boundbeta(
                        post_alpha, post_beta
                    )  # Updates psi-sticks
                if i == len(root["children"]) - 1 and truncate:
                    root["sticks"][i] = 1.0
                data_down += child_data

            data_here = root["node"].num_local_data()

            # Resample the main break.
            post_alpha = 1.0 + data_here
            post_beta = (self.alpha_decay**depth) * self.dp_alpha + data_down
            if prior:
                root["main"] = (
                    boundbeta(1.0, (self.alpha_decay**depth) * self.dp_alpha)
                    if self.min_depth <= depth
                    else 0.0
                )  # Updates nu-sticks
            else:
                root["main"] = (
                    boundbeta(post_alpha, post_beta) if self.min_depth <= depth else 0.0
                )  # Updates nu-sticks
            if len(root["children"]) < 1 and truncate:
                root["main"] = 1.0

            return data_here + data_down

        if top_node is None:
            top_node = self.root
        descend(top_node)

    def set_stick_params(self, top_node=None, prior=False, truncate=False):
        def descend(root, depth=0):
            data_down = 0
            indices = list(range(len(root["children"])))
            # indices.reverse()
            indices = indices[::-1]
            for i in indices:
                child = root["children"][i]
                child_data = descend(child, depth + 1)
                post_alpha = 1.0 + child_data
                post_beta = self.dp_gamma + data_down
                child["node"].variational_parameters["locals"]["psi_log_mean"] = np.log(
                    post_alpha
                )
                child["node"].variational_parameters["locals"]["psi_log_std"] = np.log(
                    post_beta
                )
                data_down += child_data

            data_here = root["node"].num_local_data()

            # Resample the main break.
            post_alpha = 1.0 + data_here
            post_beta = (self.alpha_decay**depth) * self.dp_alpha + data_down
            root["node"].variational_parameters["locals"]["nu_log_mean"] = np.log(
                post_alpha
            )
            root["node"].variational_parameters["locals"]["nu_log_std"] = np.log(
                post_beta
            )

            return data_here + data_down

        if top_node is None:
            top_node = self.root
        descend(top_node)

    def _resample_stick_orders(self):
        """
        Resample the order of the psi-sticks.
        """

        def descend(root, depth=0):
            if not root["children"]:  # Only nodes with children contain  psi-sticks
                return

            new_order = []
            represented = set(
                filter(
                    lambda i: root["children"][i]["node"].has_data(),
                    range(len(root["children"])),
                )
            )
            all_weights = diff(hstack([0.0, sticks_to_edges(root["sticks"])]))
            while True:
                if not represented:
                    break

                u = rand()
                while True:
                    sub_indices = list(
                        filter(
                            lambda i: i not in new_order, range(root["sticks"].shape[0])
                        )
                    )
                    sub_weights = hstack(
                        [all_weights[sub_indices], 1.0 - sum(all_weights)]
                    )
                    sub_weights = sub_weights / sum(sub_weights)
                    index = sum(u > cumsum(sub_weights))

                    if index == len(sub_indices):
                        root["sticks"] = vstack(
                            [root["sticks"], boundbeta(1, self.dp_gamma)]
                        )
                        root["children"].append(
                            {
                                "node": root["node"].spawn(),
                                "main": boundbeta(
                                    1.0,
                                    (self.alpha_decay ** (depth + 1)) * self.dp_alpha,
                                )
                                if self.min_depth <= (depth + 1)
                                else 0.0,
                                "sticks": empty((0, 1)),
                                "children": [],
                            }
                        )
                        all_weights = diff(
                            hstack([0.0, sticks_to_edges(root["sticks"])])
                        )
                    else:
                        index = sub_indices[index]
                        break

                new_order.append(index)
                represented.discard(index)

            new_children = []
            for k in new_order:
                child = root["children"][k]
                new_children.append(child)
                descend(child, depth + 1)

            for k in list(
                filter(lambda k: k not in new_order, range(root["sticks"].shape[0]))
            ):
                root["children"][k]["node"].kill()
                del root["children"][k]["node"]

            root["children"] = new_children
            root["sticks"] = zeros((len(root["children"]), 1))

        descend(self.root)

        # Immediately resample sticks.
        self.resample_sticks()

    def resample_stick_orders(self):
        """
        Resample the order of the psi-sticks. Go to each node and change the ordering
        of its children.
        """

        def descend(root, depth=0):
            if (
                len(root["children"]) == 0
            ):  # Only nodes with children contain  psi-sticks
                return

            new_order = []
            represented = set(range(len(root["children"])))
            all_weights = diff(hstack([0.0, sticks_to_edges(root["sticks"])]))
            j = 0
            while True:
                if not represented:
                    break

                j = j + 1
                if j > 10:
                    logger.debug(j, new_order, root["sticks"].shape[0], all_weights)
                    logger.debug(new_order[10000])
                sub_indices = list(
                    filter(lambda i: i not in new_order, range(root["sticks"].shape[0]))
                )
                sub_weights = all_weights[sub_indices]
                sub_weights = sub_weights / sum(sub_weights)

                u = rand()
                index = sum(u > cumsum(sub_weights))
                index = sub_indices[index]

                new_order.append(index)
                represented.discard(index)

            new_children = []
            new_sticks = []
            for k in new_order:
                child = root["children"][k]
                stick = root["sticks"][k]
                new_children.append(child)
                new_sticks.append(stick)
                descend(child, depth + 1)

            root["children"] = new_children
            root["sticks"] = np.array(new_sticks).reshape(
                -1, 1
            )  # zeros((len(root['children']),1))

        descend(self.root)

        # Immediately resample sticks.
        self.resample_sticks()

    def resample_hypers(self, dp_alpha=True, alpha_decay=True, dp_gamma=True):
        """
        Note: when subclustering, mu-stick of the root is 0, its last psi-stick is 1, and the mu-stick
        of all subclusters is 1. Take this into account when resampling the stick hyperparams.
        Also remember we have different hyperparams for subclusters
        """

        def dp_alpha_llh(
            dp_alpha, alpha_decay
        ):  # Don't consider the root on these computations, nor the subclusters
            def descend(dp_alpha, root, depth=0):
                if root["node"].parent() is None or root["node"].is_subcluster:
                    return 0.0
                else:
                    llh = (
                        betapdfln(root["main"], 1.0, (alpha_decay**depth) * dp_alpha)
                        if self.min_depth <= depth
                        else 0.0
                    )
                    for child in root["children"]:
                        llh += descend(dp_alpha, child, depth + 1)
                    return llh

            return descend(dp_alpha, self.root)

        if dp_alpha:
            upper = self.max_dp_alpha
            lower = self.min_dp_alpha
            llh_s = log(rand()) + dp_alpha_llh(self.dp_alpha, self.alpha_decay)
            while True:
                new_dp_alpha = (upper - lower) * rand() + lower
                new_llh = dp_alpha_llh(new_dp_alpha, self.alpha_decay)
                if new_llh > llh_s:
                    break
                elif new_dp_alpha < self.dp_alpha:
                    lower = new_dp_alpha
                elif new_dp_alpha > self.dp_alpha:
                    upper = new_dp_alpha
                else:
                    raise Exception("Slice sampler shrank to zero!")
            self.dp_alpha = new_dp_alpha

        if alpha_decay:
            upper = self.max_alpha_decay
            lower = self.min_alpha_decay
            llh_s = log(rand()) + dp_alpha_llh(self.dp_alpha, self.alpha_decay)
            while True:
                new_alpha_decay = (upper - lower) * rand() + lower
                new_llh = dp_alpha_llh(self.dp_alpha, new_alpha_decay)
                if new_llh > llh_s:
                    break
                elif new_alpha_decay < self.alpha_decay:
                    lower = new_alpha_decay
                elif new_alpha_decay > self.alpha_decay:
                    upper = new_alpha_decay
                else:
                    raise Exception("Slice sampler shrank to zero!")
            self.alpha_decay = new_alpha_decay

        def dp_gamma_llh(dp_gamma):
            def descend(dp_gamma, root):
                llh = 0
                if root["node"].parent() is None:  # only update root's children
                    for i, child in enumerate(root["children"]):
                        # If this is the last child of the root, its stick length is fixed and
                        # so it shouldn't contribute to the llh
                        if root["node"].parent() is None and (
                            i < len(root["children"]) - 1
                        ):
                            llh += betapdfln(root["sticks"][i], 1.0, dp_gamma)
                        llh += descend(dp_gamma, child)
                return llh

            return descend(dp_gamma, self.root)

        if dp_gamma:
            upper = self.max_dp_gamma
            lower = self.min_dp_gamma
            llh_s = log(rand()) + dp_gamma_llh(self.dp_gamma)
            while True:
                new_dp_gamma = (upper - lower) * rand() + lower
                new_llh = dp_gamma_llh(new_dp_gamma)
                if new_llh > llh_s:
                    break
                elif new_dp_gamma < self.dp_gamma:
                    lower = new_dp_gamma
                elif new_dp_gamma > self.dp_gamma:
                    upper = new_dp_gamma
                else:
                    raise Exception("Slice sampler shrank to zero!")
            self.dp_gamma = new_dp_gamma

    def get_fixed_weights(self, eta=None):
        # Set stick lengths to achieve node weights according to their depths
        # w = d_i ** gamma / (sum_j d_j**gamma)
        if eta is None:
            eta = self.eta

        node_depths = self.get_node_depths()

        nodes = []
        weights = []
        for node, depth in node_depths:
            nodes.append(node)
            weights.append(depth**eta)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        # node_weight_tuples = [(node_depth[0], weights[i]) for i, node_depth in enumerate(node_depths)]

        return weights, nodes

    def get_node(self, subtree_label):
        def descend(root):
            if root["label"] == subtree_label:
                return root["pivot_node"], root["pivot_tssb"]
            for child in root["children"]:
                out = descend(child)
                if out is not None:
                    return out

        pivot_node, pivot_tssb = descend(self.root)

        return pivot_node, pivot_tssb

    def find_node(self, u, include_leaves=True):
        def descend(root, u, depth=0):
            if depth >= self.max_depth:
                return (root["node"], [], root)
            elif u < root["main"]:
                return (root["node"], [], root)
            else:
                # Rescale the uniform variate to the remaining interval.
                u = (u - root["main"]) / (1.0 - root["main"])

                # Don't break sticks

                edges = 1.0 - cumprod(1.0 - root["sticks"])
                index = sum(u > edges)
                if index >= len(root['sticks']):
                    return (root["node"], [], root)
                edges = hstack([0.0, edges])
                u = (u - edges[index]) / (edges[index + 1] - edges[index])

                # Perhaps stop before continuing to a leaf
                if not include_leaves and len(root["children"][index]["children"]) == 0:
                    return (root["node"], [], root)
                
                (node, path, root) = descend(root["children"][index], u, depth + 1)

                path.insert(0, index)

                return (node, path, root)

        return descend(self.root, u)

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

    def get_expected_mixture(self, reset_names=False):
        """
        Computes the expected weight for each node.
        """
        if reset_names:
            self.set_node_names()

        def descend(root, mass, depth=0):
            weight = [
                mass * (1.0 / (1.0 + self.dp_alpha * self.alpha_decay ** (depth + 1)))
                if len(root["children"]) != 0
                else mass
            ]
            node = [root["node"]]
            sticks = [1.0 / (1.0 + self.dp_gamma) for stick in root["sticks"]]

            if len(sticks) > 0:
                sticks[-1] = 1.0
            edges = sticks_to_edges(np.array(sticks))
            weights = diff(hstack([0.0, edges]))

            for i, child in enumerate(root["children"]):
                (child_weights, child_nodes) = descend(
                    child,
                    mass
                    * (
                        1.0
                        - 1.0 / (1.0 + self.dp_alpha * self.alpha_decay ** (depth + 1))
                    )
                    * weights[i],
                    depth + 1,
                )
                weight.extend(child_weights)
                node.extend(child_nodes)
            return (weight, node)

        return descend(self.root, 1.0)

    def get_node_roots(self):
        def descend(root):
            sr = [root]
            for child in root["children"]:
                cr = descend(child)
                sr.extend(cr)
            return sr
        return descend(self.root)

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
            for child in root['children']:
                descend(child)
        descend(self.root)

    def get_mixture(
        self, reset_names=False, get_roots=False, get_depths=False, truncate=False
    ):
        """
        Computes Eq. (2) from Adams et al, 2010.
        """
        if reset_names:
            self.set_node_names()

        def descend(root, mass, depth=0):
            main = 1.0 if len(root["children"]) < 1 and truncate else root["main"]
            weight = [mass * main]
            node = [root["node"]]
            roots = [root]
            depths = [depth]
            sticks = np.array(
                [
                    1.0
                    if i == len(root["children"]) - 1 and truncate
                    else root["sticks"][i]
                    for i in range(len(root["children"]))
                ]
            ).astype(float)
            edges = sticks_to_edges(sticks)
            weights = diff(hstack([0.0, edges]))

            for i, child in enumerate(root["children"]):
                if get_roots:
                    if get_depths:
                        (
                            child_weights,
                            child_nodes,
                            child_roots,
                            child_depths,
                        ) = descend(
                            child,
                            mass * (1.0 - root["main"]) * weights[i],
                            depth=depth + 1,
                        )
                    else:
                        (child_weights, child_nodes, child_roots) = descend(
                            child, mass * (1.0 - root["main"]) * weights[i]
                        )
                else:
                    (child_weights, child_nodes) = descend(
                        child, mass * (1.0 - root["main"]) * weights[i]
                    )
                weight.extend(child_weights)
                node.extend(child_nodes)
                if get_roots:
                    roots.extend(child_roots)
                    if get_depths:
                        depths.extend(child_depths)
            if get_roots:
                if get_depths:
                    return (weight, node, roots, depths)
                return (weight, node, roots)
            return (weight, node)

        return descend(self.root, 1.0)

    def get_variational_mixture(self, get_roots=False, get_depths=False):
        def descend(root, mass, depth=0):
            main = (
                1.0
                if len(root["children"]) < 1
                else jnp.exp(
                    root["node"].variational_parameters["locals"]["nu_log_mean"]
                )
                / (
                    jnp.exp(
                        root["node"].variational_parameters["locals"]["nu_log_mean"]
                    )
                    + jnp.exp(
                        root["node"].variational_parameters["locals"]["nu_log_std"]
                    )
                )
            )
            weight = [mass * main]
            node = [root["node"]]
            roots = [root]
            depths = [depth]
            sticks = np.array(
                [
                    1.0
                    if i == len(root["children"]) - 1
                    else jnp.exp(
                        root["children"][i]["node"].variational_parameters["locals"][
                            "psi_log_mean"
                        ]
                    )
                    / (
                        jnp.exp(
                            root["children"][i]["node"].variational_parameters[
                                "locals"
                            ]["psi_log_mean"]
                        )
                        + jnp.exp(
                            root["children"][i]["node"].variational_parameters[
                                "locals"
                            ]["psi_log_std"]
                        )
                    )
                    for i in range(len(root["children"]))
                ],
                dtype="object",
            ).astype(float)
            edges = sticks_to_edges(sticks)
            weights = diff(hstack([0.0, edges]))

            for i, child in enumerate(root["children"]):
                if get_roots:
                    if get_depths:
                        (
                            child_weights,
                            child_nodes,
                            child_roots,
                            child_depths,
                        ) = descend(
                            child,
                            mass
                            * (
                                1.0
                                - jnp.exp(
                                    root["node"].variational_parameters["locals"][
                                        "nu_log_mean"
                                    ]
                                )
                                / (
                                    jnp.exp(
                                        root["node"].variational_parameters["locals"][
                                            "nu_log_mean"
                                        ]
                                    )
                                    + jnp.exp(
                                        root["node"].variational_parameters["locals"][
                                            "nu_log_std"
                                        ]
                                    )
                                )
                            )
                            * weights[i],
                            depth=depth + 1,
                        )
                    else:
                        (child_weights, child_nodes, child_roots) = descend(
                            child,
                            mass
                            * (
                                1.0
                                - jnp.exp(
                                    root["node"].variational_parameters["locals"][
                                        "nu_log_mean"
                                    ]
                                )
                                / (
                                    jnp.exp(
                                        root["node"].variational_parameters["locals"][
                                            "nu_log_mean"
                                        ]
                                    )
                                    + jnp.exp(
                                        root["node"].variational_parameters["locals"][
                                            "nu_log_std"
                                        ]
                                    )
                                )
                            )
                            * weights[i],
                        )
                else:
                    (child_weights, child_nodes) = descend(
                        child,
                        mass
                        * (
                            1.0
                            - jnp.exp(
                                root["node"].variational_parameters["locals"][
                                    "nu_log_mean"
                                ]
                            )
                            / (
                                jnp.exp(
                                    root["node"].variational_parameters["locals"][
                                        "nu_log_mean"
                                    ]
                                )
                                + jnp.exp(
                                    root["node"].variational_parameters["locals"][
                                        "nu_log_std"
                                    ]
                                )
                            )
                        )
                        * weights[i],
                    )
                weight.extend(child_weights)
                node.extend(child_nodes)
                if get_roots:
                    roots.extend(child_roots)
                    if get_depths:
                        depths.extend(child_depths)
            if get_roots:
                if get_depths:
                    return (weight, node, roots, depths)
                return (weight, node, roots)
            return (weight, node)

        return descend(self.root, 1.0)

    def complete_data_log_likelihood(self):
        weights, nodes = self.get_mixture()
        llhs = []
        for i, node in enumerate(nodes):
            if node.num_local_data():
                llhs.append(
                    node.num_local_data() * log(weights[i]) + node.data_log_likelihood()
                )
        return sum(array(llhs))

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

    def sticks_node_logprior(self, root, depth, truncate=True):
        llh = 0.0

        if root["main"] != 1:
            llh = (
                betapdfln(
                    root["main"], 1.0, (self.alpha_decay**depth) * self.dp_alpha
                )
                if self.min_depth <= depth
                else 0.0
            )
        for i, child in enumerate(root["children"]):
            if i != len(root["children"]) - 1:
                llh = llh + betapdfln(root["sticks"][i], 1.0, self.dp_gamma)

        return np.array(llh).ravel()[0]

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

    def unnormalized_posterior(self):
        weights, nodes = self.get_mixture()
        llhs = []
        for i, node in enumerate(nodes):
            if node.num_local_data():
                llhs.append(
                    node.num_local_data() * log(weights[i])
                    + node.data_log_likelihood()
                    + node.unobserved_factors_ll()
                )
        llh = sum(array(llhs))

        return llh

    def remove_empty_nodes(self):
        def descend(root):

            if len(root["children"]) == 0:
                return

            while True:

                empty_nodes = list(
                    filter(
                        lambda i: len(root["children"][i]["node"].data) == 0,
                        range(len(root["children"])),
                    )
                )

                if len(empty_nodes) == 0:
                    break

                index = empty_nodes[0]

                cache_children = root["children"][index]["children"]

                # root['children'][index]['node'].kill()

                del root["children"][index]

                if len(cache_children) == 0:
                    continue
                else:

                    temp1 = root["children"][:index]

                    temp2 = root["children"][index:]

                    root["children"] = temp1 + cache_children + temp2
                    root["sticks"] = zeros((len(root["children"]), 1))

            for child in root["children"]:
                descend(child)

        descend(self.root)
        self.resample_sticks()

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

    def get_weight_distribution(self):
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

    def label_nodes(self, counts=False, names=False):
        if not counts or names is True:
            self.label_nodes_names()
        elif not names or counts is True:
            self.label_nodes_counts()

    def set_node_names(self, root=None, root_name="X"):
        if root is None:
            root = self.root

        root["label"] = str(root_name)
        root["node"].label = str(root_name)

        def descend(root, name):
            for i, child in enumerate(root["children"]):
                child_name = f"{name}-{i}"
                root["children"][i]["label"] = child_name
                root["children"][i]["node"].label = child_name
                descend(child, child_name)

        descend(root, root_name)

    def set_subcluster_node_names(self):
        # Assumes the other fixed nodes have already been named, and ignores the root

        def descend(root, name):
            for i, child in enumerate(root["children"]):
                child_name = "%s-%d" % (root["label"], i)

                if child["node"].is_subcluster:
                    child["label"] = child_name

                descend(child, child_name)

        descend(self.root, self.root["label"])

    def subsample_tree(self, p=0.5):
        root_dict = dict()
        root_dict[self.root["node"].label] = dict(
            parent="-1",
            params=self.root["node"].unobserved_factors,
            node=self.root["node"],
        )

        def descend(root, parent):
            for i, child in enumerate(root["children"]):
                keep = np.random.binomial(1, p)
                pa = root["node"].label
                if keep:
                    root_dict[child["node"].label] = dict(
                        parent=parent,
                        params=child["node"].unobserved_factors,
                        node=self.root["node"],
                    )
                    pa = child["node"].label
                descend(child, pa)

        parent = self.root["node"].label
        descend(self.root, parent)
        return root_dict

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

    def tssb_dict2graphviz(
        self,
        root_dict,
        g=None,
        counts=False,
        weights=False,
        root_fillcolor=None,
        color_by_weight=False,
        expected_weights=False,
        events=False,
        color_subclusters=False,
        show_labels=True,
        gene=None,
        genemode="raw",
        fontcolor="black",
        fontname=None,
        node_color_dict=None,
        label_fontsize=24,
        size_fontsize=12,
    ):
        if g is None:
            g = Digraph()
            g.attr(fontcolor=fontcolor, fontname=fontname)

        if node_color_dict is None:
            if color_by_weight or weights:
                self.set_node_names()
                if expected_weights:
                    name_weight_dict = dict(
                        [(n.label, w) for w, n in zip(*self.get_expected_mixture())]
                    )
                else:
                    name_weight_dict = dict(
                        [(n.label, w) for w, n in zip(*self.get_mixture())]
                    )
                w, nodes = self.get_fixed_weights()
                for i, node in enumerate(nodes):
                    name_weight_dict[node.label] = w[i]

                norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
                mapper = matplotlib.cm.ScalarMappable(
                    norm=norm, cmap=matplotlib.cm.Reds
                )
                node_color_dict = dict()
                for name in name_weight_dict:
                    color = matplotlib.colors.to_hex(
                        mapper.to_rgba(name_weight_dict[name])
                    )
                    node_color_dict[name] = color

            if gene is not None:
                if genemode == "unobserved":
                    nodes, vals = self.ntssb.get_node_unobs()
                    vals = np.array(vals)
                    global_min, global_max = np.min(vals), np.max(vals)
                    node_labs = [node.label for node in nodes]
                    gene_vals = [val[gene] for val in vals]
                    cmap = self.ntssb.exp_cmap
                    norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)
                elif genemode == "observed":
                    nodes, vals = self.ntssb.get_node_obs()
                    node_labs = [node.label for node in nodes]
                    gene_vals = [val[gene] for val in vals]
                    cmap = self.ntssb.obs_cmap
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=cmap.N - 1)
                else:
                    nodes, vals = self.ntssb.get_avg_node_exp()
                    vals = np.array(vals)
                    global_min, global_max = np.min(vals), np.max(vals)
                    node_labs = [node.label for node in nodes]
                    gene_vals = [val[gene] for val in vals]
                    name_exp_dict = dict(
                        [
                            (n.label, nodeavg[gene])
                            for n, nodeavg in zip(*self.ntssb.get_avg_node_exp())
                        ]
                    )
                    cmap = self.ntssb.exp_cmap
                    norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)
                name_exp_dict = dict(zip(node_labs, gene_vals))
                mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                node_color_dict = dict()
                for name in name_exp_dict:
                    color = matplotlib.colors.to_hex(
                        mapper.to_rgba(name_exp_dict[name])
                    )
                    node_color_dict[name] = color

        root_label = ""
        if show_labels:
            root_label = f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{self.root["label"].replace("-", "")}</B></FONT>'
        if counts:
            root_label = f'<FONT POINT-SIZE="{size_fontsize}" FACE="Arial">{self.root["node"].num_local_data()}</FONT>'
        if show_labels and counts:
            root_label = (
                f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{self.root["label"].replace("-", "")}</B></FONT>'
                + "<br/><br/>"
                + f'<FONT FACE="Arial">{str(self.root["node"].num_local_data())} cells</FONT>'
            )
        if events:
            root_label = self.root["node"].event_str
        if show_labels and events:
            root_label = self.root["label"] + "<br/><br/>" + self.root["node"].event_str
        if weights:
            root_label = name_weight_dict[self.root["label"]]
            root_label = str(root_label)[:5]

        style = None
        if root_fillcolor is not None:
            style = "filled"
            fillcolor = root_fillcolor
        elif node_color_dict is not None:
            style = "filled"
            fillcolor = node_color_dict[str(self.root["label"])]
        else:
            style = "filled"
            fillcolor = self.color
        g.node(
            str(self.root["label"]),
            "<" + str(root_label) + ">",
            fillcolor=fillcolor,
            style=style,
        )

        def descend(root, g):
            name = root["label"]
            for i, child in enumerate(root["children"]):
                child_name = child["label"]
                child_label = ""
                if show_labels:
                    child_label = f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{child_name.replace("-", "")}</B></FONT>'

                if counts:
                    child_label = f'<FONT POINT-SIZE="{size_fontsize}" FACE="Arial">{str(root["children"][i]["node"].num_local_data())} cells</FONT>'

                if show_labels and counts:
                    child_label = (
                        f'<FONT POINT-SIZE="{label_fontsize}" FACE="Arial"><B>{child_name.replace("-", "")}</B></FONT>'
                        + "<br/><br/>"
                        + f'<FONT FACE="Arial">{str(root["children"][i]["node"].num_local_data())} cells</FONT>'
                    )

                if events:
                    child_label = root["children"][i]["node"].event_str

                if show_labels and events:
                    child_label = (
                        child_name
                        + "<br/><br/>"
                        + root["children"][i]["node"].event_str
                    )

                if weights:
                    child_label = name_weight_dict[child["label"]]
                    child_label = str(child_label)[:5]

                fillcolor = None
                style = None
                if node_color_dict is not None:
                    fillcolor = node_color_dict[str(child_name)]
                    style = "filled"
                g.node(
                    str(child_name),
                    "<" + str(child_label) + ">",
                    fillcolor=fillcolor,
                    style=style,
                )

                edge_color = "black"

                g.edge(str(name), str(child_name), color=edge_color)

                g = descend(child, g)

            return g

        g = descend(self.root, g)
        return g

    def plot_tree(
        self,
        g=None,
        counts=False,
        root_fillcolor=None,
        color_by_weight=False,
        weights=False,
        expected_weights=False,
        events=False,
        color_subclusters=False,
        reset_names=True,
        show_labels=True,
        gene=None,
        genemode="raw",
        fontcolor="black",
        fontname=None,
        node_color_dict=None,
        label_fontsize=24,
        size_fontsize=12,
    ):
        if reset_names:
            self.set_node_names(root_name=self.label)
        g = self.tssb_dict2graphviz(
            self.root,
            g=g,
            counts=counts,
            root_fillcolor=root_fillcolor,
            color_by_weight=color_by_weight,
            weights=weights,
            expected_weights=expected_weights,
            events=events,
            color_subclusters=color_subclusters,
            show_labels=show_labels,
            gene=gene,
            genemode=genemode,
            fontcolor=fontcolor,
            fontname=fontname,
            node_color_dict=node_color_dict,
            label_fontsize=label_fontsize,
            size_fontsize=size_fontsize,
        )

        return g
