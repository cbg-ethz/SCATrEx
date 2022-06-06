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
from ..util import *

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
        dp_alpha=1.0,
        dp_gamma=1.0,
        min_depth=0,
        max_depth=15,
        alpha_decay=0.5,
        eta=0.0,
        color="black",
    ):
        if root_node is None:
            raise Exception("Root node must be specified.")

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.dp_alpha = dp_alpha  # smaller dp_alpha => larger nu => less nodes
        self.dp_gamma = dp_gamma  # smaller dp_gamma => larger psi => less nodes
        self.alpha_decay = alpha_decay
        self.weight = 1.0
        self.color = color

        self.eta = eta  # wether to put more weight on top or bottom of tree in case we want fixed weights

        self.label = label

        self.assignments = []

        self.root = {
            "node": root_node,
            "main": boundbeta(1.0, dp_alpha)
            if self.min_depth == 0
            else 0.0,  # if min_depth > 0, no data can be added to the root (main stick is nu)
            "sticks": empty((0, 1)),  # psi sticks
            "children": [],
            "label": label,
        }
        root_node.tssb = self
        self.root_node = root_node

        self.ntssb = ntssb

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
        self, root_params=True, down_params=True, node_hyperparams=None
    ):
        # Reset node parameters
        def descend(root):
            root["node"].reset_parameters(
                root_params=root_params, down_params=down_params, **node_hyperparams
            )
            for child in root["children"]:
                descend(child)

        descend(self.root)

    def reset_node_variational_parameters(self, **kwargs):
        # Reset node parameters
        def descend(root):
            root["node"].reset_variational_parameters(**kwargs)
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

    def resample_node_params(
        self,
        iters=1,
        independent_subtrees=False,
        top_node=None,
        data_indices=None,
        compound=True,
    ):
        """
        Go through all nodes in the tree, starting at the bottom, and resample their parameters iteratively.
        """
        if top_node is None:
            top_node = self.root

        for iter in range(iters):

            def descend(root):
                for index, child in enumerate(root["children"]):
                    descend(child)
                root["node"].resample_params(
                    independent_subtrees=independent_subtrees,
                    data_indices=data_indices,
                    compound=compound,
                )

            descend(top_node)

    def resample_assignment(self, current_node, n, obs):
        def path_lt(path1, path2):
            if len(path1) == 0 and len(path2) == 0:
                return 0
            elif len(path1) == 0:
                return 1
            elif len(path2) == 0:
                return -1
            s1 = "".join(map(lambda i: "%03d" % (i), path1))
            s2 = "".join(map(lambda i: "%03d" % (i), path2))

            return cmp(s2, s1)

        epsilon = finfo(float64).eps
        lengths = []
        reassign = 0
        better = 0

        # Get an initial uniform variate.
        ancestors = current_node.get_ancestors()
        current = self.root
        indices = []
        for anc in ancestors[1:]:
            index = list(map(lambda c: c["node"], current["children"])).index(anc)
            current = current["children"][index]
            indices.append(index)

        max_u = 1.0
        min_u = 0.0
        llh_s = log(rand()) + current_node.logprob(obs)
        # llh_s = self.assignments[n].logprob(self.data[n:n+1]) - 0.0000001
        while True:
            new_u = (max_u - min_u) * rand() + min_u
            (new_node, new_path, _) = self.find_node(new_u)
            new_llh = new_node.logprob(obs)
            if new_llh > llh_s:
                if new_node != current_node:
                    if new_llh > current_node.logprob(obs):
                        better += 1
                    current_node.remove_datum(n)
                    new_node.add_datum(n)
                    current_node = new_node
                    reassign += 1
                break
            elif abs(max_u - min_u) < epsilon:
                logger.debug("Slice sampler shrank down.  Keep current state.")
                break
            else:
                path_comp = path_lt(indices, new_path)
                if path_comp < 0:
                    min_u = new_u
                    # if we are at a leaf in a fixed tree, move on, because we can't create more nodes
                    if len(new_node.children()) == 0:
                        break
                elif path_comp > 0:
                    max_u = new_u
                else:
                    raise Exception("Slice sampler weirdness.")

        return current_node

    def resample_assignments(self):
        def path_lt(path1, path2):
            if len(path1) == 0 and len(path2) == 0:
                return 0
            elif len(path1) == 0:
                return 1
            elif len(path2) == 0:
                return -1
            s1 = "".join(map(lambda i: "%03d" % (i), path1))
            s2 = "".join(map(lambda i: "%03d" % (i), path2))

            return cmp(s2, s1)

        epsilon = finfo(float64).eps
        lengths = []
        reassign = 0
        better = 0
        for n in range(self.num_data):

            # Get an initial uniform variate.
            ancestors = self.assignments[n].get_ancestors()
            current = self.root
            indices = []
            for anc in ancestors[1:]:
                index = list(map(lambda c: c["node"], current["children"])).index(anc)
                current = current["children"][index]
                indices.append(index)

            max_u = 1.0
            min_u = 0.0
            llh_s = log(rand()) + self.assignments[n].logprob(self.data[n : n + 1])
            # llh_s = self.assignments[n].logprob(self.data[n:n+1]) - 0.0000001
            while True:
                new_u = (max_u - min_u) * rand() + min_u
                (new_node, new_path, _) = self.find_node(new_u)
                new_llh = new_node.logprob(self.data[n : n + 1])
                if new_llh > llh_s:
                    if new_node != self.assignments[n]:
                        if new_llh > self.assignments[n].logprob(self.data[n : n + 1]):
                            better += 1
                        self.assignments[n].remove_datum(n)
                        new_node.add_datum(n)
                        self.assignments[n] = new_node
                        reassign += 1
                    break
                elif abs(max_u - min_u) < epsilon:
                    logger.debug("Slice sampler shrank down.  Keep current state.")
                    break
                else:
                    path_comp = path_lt(indices, new_path)
                    if path_comp < 0:
                        min_u = new_u
                        # if we are at a leaf in a fixed tree, move on, because we can't create more nodes
                        if len(new_node.children()) == 0:
                            break
                    elif path_comp > 0:
                        max_u = new_u
                    else:
                        raise Exception("Slice sampler weirdness.")
            lengths.append(len(new_path))
        lengths = array(lengths)
        # logger.debug "reassign: "+str(reassign)+" better: "+str(better)

    # def resample_birth(self):
    #     # Break sticks to choose node
    #     u = rand()
    #     node, root = self.find_node(u)
    #
    #     # Create child
    #     stick_length = boundbeta(1, self.dp_gamma)
    #     root['sticks'] = vstack([ root['sticks'], stick_length ])
    #     root['children'].append({ 'node'     : root['node'].spawn(False, self.root_node.observed_parameters),
    #                               'main'     : boundbeta(1.0, (self.alpha_decay**(depth+1))*self.dp_alpha) if self.min_depth <= (depth+1) else 0.0,
    #                               'sticks'   : empty((0,1)),
    #                               'children' : []   })
    #
    #     # Update parameters of data in parent and child node until convergence:
    #     # assignments (starting from parent)
    #     # stick lengths
    #     # parameters
    #
    # def resample_merge(self):
    #     # Choose any node a at random
    #
    #     # Compute similarity of parameters of leaf sibilings and parent
    #
    #     # Choose closest node b
    #
    #
    #
    #     # If the merge is accepted, the child nodes of a are transferred to node b.

    def resample_tree_topology(self, children_trees, independent_subtrees=False):
        # x = self.complete_data_log_likelihood_nomix()
        post = self.ntssb.unnormalized_posterior()
        weights, nodes = self.get_mixture()

        empty_root = False
        if len(nodes) > 1:
            if len(nodes[0].data) == 0:
                logger.debug("Swapping root")
                empty_root = True
                nodeAnum = 0
            else:
                nodeAnum = randint(0, len(nodes))
                nodeBnum = randint(0, len(nodes))
                while nodeAnum == nodeBnum:
                    nodeBnum = randint(0, len(nodes))

            def swap_nodes(nodeAnum, nodeBnum, verbose=False):
                def findNodes(root, nodeNum, nodeA=False, nodeB=False):
                    node = root
                    if nodeNum == nodeAnum:
                        nodeA = node
                    if nodeNum == nodeBnum:
                        nodeB = node
                    for i, child in enumerate(root["children"]):
                        nodeNum = nodeNum + 1
                        (nodeA, nodeB, nodeNum) = findNodes(
                            child, nodeNum, nodeA, nodeB
                        )
                    return (nodeA, nodeB, nodeNum)

                (nodeA, nodeB, nodeNum) = findNodes(self.root, nodeNum=0)

                paramsA = nodeA["node"].unobserved_factors
                dataA = set(nodeA["node"].data)
                mainA = nodeA["main"]

                nodeA["node"].unobserved_factors = nodeB["node"].unobserved_factors
                nodeA["node"].node_mean = (
                    nodeA["node"].baseline_caller()
                    * nodeA["node"].cnvs
                    / 2
                    * np.exp(nodeA["node"].unobserved_factors)
                )

                for dataid in list(dataA):
                    nodeA["node"].remove_datum(dataid)
                for dataid in nodeB["node"].data:
                    nodeA["node"].add_datum(dataid)
                    self.ntssb.assignments[dataid]["node"] = nodeA["node"]
                nodeA["main"] = nodeB["main"]

                nodeB["node"].unobserved_factors = paramsA
                nodeB["node"].node_mean = (
                    nodeB["node"].baseline_caller()
                    * nodeB["node"].cnvs
                    / 2
                    * np.exp(nodeB["node"].unobserved_factors)
                )

                dataB = set(nodeB["node"].data)

                for dataid in list(dataB):
                    nodeB["node"].remove_datum(dataid)
                for dataid in dataA:
                    nodeB["node"].add_datum(dataid)
                    self.ntssb.assignments[dataid]["node"] = nodeB["node"]
                nodeB["main"] = mainA

                if not independent_subtrees:
                    # Go to subtrees
                    # For each subtree, if pivot was swapped, update it
                    for child in children_trees:
                        if child["pivot_node"] == nodeA["node"]:
                            child["pivot_node"] = nodeB["node"]
                            child["node"].root["node"].set_parent(
                                nodeB["node"], reset=False
                            )
                        elif child["pivot_node"] == nodeB["node"]:
                            child["pivot_node"] = nodeA["node"]
                            child["node"].root["node"].set_parent(
                                nodeA["node"], reset=False
                            )

                logger.debug(
                    f"Swapped {nodeA['node'].label} with {nodeB['node'].label}"
                )

            if empty_root:
                logger.debug("checking alternative root")
                nodenum = []
                for ii, nn in enumerate(nodes):
                    if len(nodes[ii].data) > 0:
                        nodenum.append(ii)
                post_temp = zeros(len(nodenum))
                for idx, nodeBnum in enumerate(nodenum):
                    logger.debug(f"nodeBnum: {nodeBnum}")
                    logger.debug(f"nodeAnum: {nodeAnum}")
                    swap_nodes(nodeAnum, nodeBnum)
                    for i in range(5):
                        self.resample_sticks()
                        self.ntssb.root["node"].root["node"].resample_cell_params()
                    post_new = self.ntssb.unnormalized_posterior()
                    post_temp[idx] = post_new

                    accept_prob = np.exp(np.min([0.0, post_new - post]))

                    if rand() > accept_prob:
                        swap_nodes(nodeAnum, nodeBnum)
                        for i in range(5):
                            self.resample_sticks()
                            self.ntssb.root["node"].root["node"].resample_cell_params()

                        if nodeBnum == len(nodes) - 1:
                            logger.debug("forced swapping")
                            nodeBnum = post_temp.argmax() + 1
                            swap_nodes(nodeAnum, nodeBnum)
                            for i in range(5):
                                self.resample_sticks()
                                self.ntssb.root["node"].root[
                                    "node"
                                ].resample_cell_params()

                            self.resample_node_params()
                            self.resample_stick_orders()
                    else:
                        logger.debug("Successful swap!")
                        self.resample_node_params()
                        self.resample_stick_orders()
                        break
            # else:
            #     swap_nodes(nodeAnum,nodeBnum, verbose=True)
            #     for i in range(5):
            #         self.resample_sticks()
            #         self.ntssb.root['node'].root['node'].resample_cell_params()
            #
            #     post_new = self.ntssb.unnormalized_posterior()
            #     accept_prob = np.exp(np.min([0., post_new - post]))
            #     if (rand() > accept_prob):
            #         logger.debug("Unsuccessful swap.")
            #         swap_nodes(nodeAnum,nodeBnum) # swap back
            #         for i in range(5):
            #             self.resample_sticks()
            #             self.ntssb.root['node'].root['node'].resample_cell_params()
            #
            #     else:
            #         logger.debug("Successful swap!")
            #         self.resample_node_params()
            #         self.resample_stick_orders()

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

    def find_node(self, u, truncated=False):
        def descend(root, u, depth=0):
            if depth >= self.max_depth:
                # logger.debug >>sys.stderr, "WARNING: Reached maximum depth."
                return (root["node"], [], root)
            elif u < root["main"]:
                return (root["node"], [], root)
            else:
                # Rescale the uniform variate to the remaining interval.
                u = (u - root["main"]) / (1.0 - root["main"])

                if not truncated:
                    # Perhaps break sticks out appropriately.
                    while (
                        not root["children"] or (1.0 - prod(1.0 - root["sticks"])) < u
                    ):
                        stick_length = boundbeta(1, self.dp_gamma)
                        root["sticks"] = vstack([root["sticks"], stick_length])
                        root["children"].append(
                            {
                                "node": root["node"].spawn(
                                    False, self.root_node.observed_parameters
                                ),
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
                else:
                    root["sticks"][-1] = 1.0

                edges = 1.0 - cumprod(1.0 - root["sticks"])
                index = sum(u > edges)
                edges = hstack([0.0, edges])
                u = (u - edges[index]) / (edges[index + 1] - edges[index])

                (node, path, root) = descend(root["children"][index], u, depth + 1)

                path.insert(0, index)

                return (node, path, root)

        return descend(self.root, u)

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
                else jnn.sigmoid(
                    root["node"].variational_parameters["locals"]["nu_log_mean"]
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
                    else jnn.sigmoid(
                        root["children"][i]["node"].variational_parameters["locals"][
                            "psi_log_mean"
                        ]
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
                                - jnn.sigmoid(
                                    root["node"].variational_parameters["locals"][
                                        "nu_log_mean"
                                    ]
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
                                - jnn.sigmoid(
                                    root["node"].variational_parameters["locals"][
                                        "nu_log_mean"
                                    ]
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
                            - jnn.sigmoid(
                                root["node"].variational_parameters["locals"][
                                    "nu_log_mean"
                                ]
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

    def set_node_names(self, root_name="X"):
        self.root["label"] = str(root_name)
        self.root["node"].label = str(root_name)

        def descend(root, name):
            for i, child in enumerate(root["children"]):
                child_name = "%s-%d" % (name, i)

                root["children"][i]["label"] = child_name
                root["children"][i]["node"].label = child_name

                descend(child, child_name)

        descend(self.root, root_name)

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
            root_label = self.root["label"]
        if counts:
            root_label = self.root["node"].num_local_data()
        if show_labels and counts:
            root_label = (
                self.root["label"]
                + "<br/><br/>"
                + str(self.root["node"].num_local_data())
                + " cells"
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
            "<" + str(root_label).replace("-", "") + ">",
            fillcolor=fillcolor,
            style=style,
        )

        def descend(root, g):
            name = root["label"]
            for i, child in enumerate(root["children"]):
                child_name = child["label"]
                child_label = ""
                if show_labels:
                    child_label = child_name

                if counts:
                    child_label = root["children"][i]["node"].num_local_data()

                if show_labels and counts:
                    child_label = (
                        child_name
                        + "<br/><br/>"
                        + str(root["children"][i]["node"].num_local_data())
                        + " cells"
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
                    "<" + str(child_label).replace("-", "") + ">",
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
        )

        return g
