import numpy as np
from copy import deepcopy
from tqdm.auto import trange
from time import time
import matplotlib.pyplot as plt

from ..utils.math_utils import *

import jax
import jax.numpy as jnp

import logging

logger = logging.getLogger(__name__)


class StructureSearch(object):
    def __init__(self, ntssb):
        # Keep a pointer to the current tree
        self.tree = deepcopy(ntssb)

        # And a pointer to the changed tree
        self.proposed_tree = deepcopy(self.tree)

        self.traces = dict()
        self.traces["tree"] = []
        self.traces["elbo"] = []
        self.traces["n_nodes"] = []
        self.traces["elbos"] = []
        self.best_tree = deepcopy(self.tree)

    def run_search(self, n_iters=10, n_epochs=10, mc_samples=10, step_size=0.01, moves_per_tssb=1, pr_freq=0., global_freq=0, memoized=True, update_roots=True, seed=42, swap_freq=0, update_outer_ass=True):
        """
        Start with learning the parameters of the model for the non-augmented tree,
        which are just the assignments of cells to TSSBs, the outer stick parameters, 
        and the global model parameters
        """
        # Generate PRNG key
        key = jax.random.PRNGKey(seed)

        self.proposed_tree = deepcopy(self.tree)

        # Run the structure search 
        t = trange(n_iters, desc='Finding NTSSB', leave=True)
        for i in t:
            key, subkey = jax.random.split(key)

            update_globals = False
            if global_freq != 0:
                if i % global_freq == 0: 
                    # Do birth-merge step in which other local params are updated
                    update_globals = True

            if pr_freq != 0 and i % pr_freq == 0:
                # Do prune-reattach move
                # Prune and reattach: traverse the tree and propose pruning nodes and reattaching somewhere else inside their TSSB
                self.prune_reattach(subkey, moves_per_tssb=moves_per_tssb)

            else:
                # Birth: traverse the tree and spawn a bunch of nodes (quick and helps escape local optima)
                self.birth_merge(subkey, n_epochs=n_epochs, memoized=memoized, mc_samples=mc_samples, step_size=step_size, moves_per_tssb=moves_per_tssb, update_roots=update_roots, update_globals=update_globals,
                                 update_outer_ass=update_outer_ass)

            # Swap roots: traverse the tree and propose swapping roots of TSSBs with their immediate children
            # self.swap_roots(subkey, moves_per_tssb=moves_per_tssb)

            # Keep best
            if self.tree.elbo > self.best_tree.elbo:
                self.best_tree = deepcopy(self.tree)

            # Compute ELBO
            self.traces['elbo'].append(self.tree.elbo)

            t.set_description(f"Finding NTSSB ({self.tree.n_total_nodes} nodes, elbo: {self.tree.elbo})" )
            t.refresh() # to show immediately the update

    def swap(self, key, memoized=True, n_epochs=10, **learn_kwargs):
        """
        Propose changing the root of a subtree. Preferably if 
        """
        if self.proposed_tree.n_total_nodes == self.proposed_tree.n_nodes:
            self.logger("Nothing to swap.")
            return

        n_children = 0
        while n_children == 0:
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey)
            # See in which subtree it lands
            subtree, _, u = self.proposed_tree.find_node(u)
            # Choose either couple of children to merge or a child to merge with parent
            parent = subtree.root
            n_children = len(parent['children'])

        # Swap parent-child
        source_idx = jax.random.choice(subkey, n_children)
        source = parent['children'][source_idx]
        target = parent

        self.proposed_tree.swap_root(source, target)
        self.proposed_tree.compute_elbo(memoized=memoized)

        if self.proposed_tree.elbo > self.tree.elbo:
            # print(f"Merged {source_label} to {target_label}")
            self.tree = deepcopy(self.proposed_tree)
        else:
            self.proposed_tree.learn_params(n_epochs, memoized=memoized, **learn_kwargs)
            self.proposed_tree.compute_elbo(memoized=memoized)
            if self.proposed_tree.elbo > self.tree.elbo:
                self.tree = deepcopy(self.proposed_tree)
            else:
                self.proposed_tree = deepcopy(self.tree)        

    def birth_merge(self, key, moves_per_tssb=1, n_epochs=100, update_roots=False, mc_samples=10, step_size=0.01, memoized=True, update_globals=False,
                    update_outer_ass=False):
        # Birth: traverse the tree and spawn a bunch of nodes (quick and helps escape local optima)
        self.birth(key, moves_per_tssb=moves_per_tssb)

        # Update parameters in n_epochs passes through the data, interleaving node updates with local batch updates
        self.tree.learn_params(int(n_epochs/2), update_roots=update_roots, mc_samples=mc_samples, 
                                step_size=step_size, memoized=memoized, update_outer_ass=update_outer_ass, ass_anneal=.1)
        self.tree.learn_params(int(n_epochs/2), update_roots=update_roots, mc_samples=mc_samples, 
                                step_size=step_size, memoized=memoized, update_outer_ass=update_outer_ass, ass_anneal=1.)        
        self.tree.compute_elbo(memoized=memoized)
        self.proposed_tree = deepcopy(self.tree)
        
        # Merge: traverse the tree and propose merges and accept/reject based on their summary statistics (reliable)
        self.merge(key, moves_per_tssb=moves_per_tssb, memoized=memoized, update_globals=update_globals,
                n_epochs=n_epochs, mc_samples=mc_samples, step_size=step_size)

    def birth(self, key, moves_per_tssb=1):
        """
        Spawn `moves_per_tssb=1` nodes. First select targets, and then add nodes. Gives a more local search
        """
        n_births = self.proposed_tree.n_nodes * moves_per_tssb
        targets = []
        for _ in range(n_births):
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey)
            target = self.proposed_tree.get_node(u, key=subkey, uniform=True, variational=True)
            targets.append(target)
        
        for target in targets:
            key, subkey = jax.random.split(key)
            new_node = target['node'].tssb.add_node(target, seed=int(subkey[1]))
            if jax.random.bernoulli(subkey):
                new_node.init_new_node_kernel()
        
        # Always accept
        self.tree = deepcopy(self.proposed_tree)

    def merge_root(self, key, memoized=True, n_epochs=10, **learn_kwargs):
        """
        Propose merging a root's child to the root and keeping the child's parameters. Optimize and accept if ELBO improves
        This is done by swapping the parameters of root and child and then merging child to root as usual
        """        
        if self.proposed_tree.n_total_nodes == self.proposed_tree.n_nodes:
            self.logger("Nothing to swap.")
            return

        n_children = 0
        while n_children == 0:
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey)
            # See in which subtree it lands
            subtree, _, u = self.proposed_tree.find_node(u)
            # Choose either couple of children to merge or a child to merge with parent
            parent = subtree.root
            n_children = len(parent['children'])

        # Merge parent-child
        source_idx = jax.random.choice(subkey, n_children)
        source = parent['children'][source_idx]
        target = parent

        slab = source['node'].label
        tlab = target['node'].label
        # self.proposed_tree.swap_nodes(source['node'], target['node'])
        self.proposed_tree.swap_root(source['node'].label, target['node'].label)
        subtree.merge_nodes(target, source, target)
        self.proposed_tree.compute_elbo(memoized=memoized)

        if self.proposed_tree.elbo > self.tree.elbo: # if ELBO improves even before optimizing
            self.tree = deepcopy(self.proposed_tree)
        else:
            # Update node parameters
            self.proposed_tree.learn_model(n_epochs, update_globals=False, update_roots=True, memoized=memoized, **learn_kwargs)
            self.proposed_tree.compute_elbo(memoized=memoized)
            if self.proposed_tree.elbo > self.tree.elbo:
                self.tree = deepcopy(self.proposed_tree)
            else:
                self.proposed_tree = deepcopy(self.tree)               

    def merge(self, key, moves_per_tssb=1, memoized=True, update_globals=False, n_epochs=10, **learn_kwargs):
        """
        Traverse the trees and propose and accept/reject merges as we go using local suff stats
        """
        n_merges = int(0.7 * self.proposed_tree.n_total_nodes * moves_per_tssb * 2)
        if update_globals:
            n_merges = int(0.7 * self.proposed_tree.n_total_nodes * moves_per_tssb)
        for _ in range(n_merges):
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey)
            parent = self.proposed_tree.get_node(u, key=subkey, uniform=True, include_leaves=False) # get non-leaf node, without accounting for weights
            tssb = parent['node'].tssb
            # Choose either couple of children to merge or a child to merge with parent
            n_children = len(parent['children'])
            if n_children == 0:
                continue
            if n_children > 1:
                # Choose 
                if jax.random.bernoulli(subkey, 0.5) == 1:
                    # Merge sibling-sibling
                    # Choose a child
                    source_idx, target_idx = jax.random.choice(subkey, n_children, shape=(2,), replace=False)
                    # Choose most similar sibling
                    source = parent['children'][source_idx]
                    target = parent['children'][target_idx]
                else:
                    # Merge parent-child
                    # Choose most similar child
                    source_idx = jax.random.choice(subkey, n_children)
                    source = parent['children'][source_idx]
                    target = parent
            else:
                # Merge parent-child
                source = parent['children'][0]
                target = parent

            source_label = source['node'].label
            target_label = target['node'].label
            # print(f"Will merge {source_label} to {target_label}")
            # Merge, updating suff stats
            tssb.merge_nodes(parent, source, target)
            # Update node sticks
            tssb.update_stick_params(parent)
            # Update pivot probs
            tssb.update_pivot_probs()
            # Compute ELBO of new tree
            self.proposed_tree.compute_elbo(memoized=memoized)
            # print(f"{self.tree.elbo} -> {self.proposed_tree.elbo}")
            # Update if ELBO improved
            if self.proposed_tree.elbo > self.tree.elbo:
                # print(f"Merged {source_label} to {target_label}")
                self.tree = deepcopy(self.proposed_tree)
            else:
                # Maybe update other locals 
                if update_globals:
                    # print("Inference")
                    # print(self.tree.elbo)
                    self.proposed_tree.learn_params(n_epochs, memoized=memoized, **learn_kwargs)
                    self.proposed_tree.compute_elbo(memoized=memoized)
                    # print(self.proposed_tree.elbo)
                    if self.proposed_tree.elbo > self.tree.elbo:
                        self.tree = deepcopy(self.proposed_tree)
                    else:
                        self.proposed_tree = deepcopy(self.tree)
                else:
                    self.proposed_tree = deepcopy(self.tree)


    def prune_reattach(self, moves_per_tssb=1):
        """
        Prune subtree and reattach somewhere else within the same TSSB
        """
        n_prs = self.proposed_tree.n_nodes * moves_per_tssb
        for _ in range(n_prs):
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey)
            parent, source, target = self.proposed_tree.get_nodes(u, n_nodes=2) # find two nodes in the same TSSB
            tssb = parent['node'].tssb
            tssb.prune_reattach(parent, source, target)
            # Update node stick parameters to account for changed mass distribution
            tssb.update_stick_params(parent)
            tssb.update_stick_params(target)

            # Optimize kernel parameters of root of moved subtree
            tssb.update_node_params(source)

            # Compute ELBO of new tree
            self.proposed_tree.compute_elbo()
            # Update if ELBO improved
            if self.proposed_tree.elbo > self.tree.elbo:
                self.tree = self.proposed_tree


    def plot_traces(
        self,
        keys=None,
        figsize=(16, 10),
        highlight_max=True,
        highlight_s=100,
        highlight_color="red",
    ):
        if keys is None:
            keys = list(self.traces.keys())
        if highlight_max:
            it_max_score = np.argmax(self.traces["elbo"])

        fig, ax_list = plt.subplots(len(keys), 1, sharex=True, figsize=figsize)
        for i, k in enumerate(keys):
            if len(keys) > 1:
                ax = ax_list[i]
            else:
                ax = ax_list
            ax.plot(self.traces[k])
            ax.set_title(k)
            if k == "n_nodes":
                ax.set_ylim(0, self.tree.max_nodes)
            if highlight_max:
                ax.scatter(
                    it_max_score,
                    self.traces[k][it_max_score],
                    s=highlight_s,
                    color=highlight_color,
                )
                ax.plot(
                    [0, it_max_score],
                    [self.traces[k][it_max_score], self.traces[k][it_max_score]],
                    ls="--",
                    color="black",
                )
                if (
                    type(self.traces[k][0]) is float
                    or type(self.traces[k][0]) is int
                    or k == "elbo"
                    or k == "score"
                ):
                    ax.plot(
                        [it_max_score, it_max_score],
                        [ax.get_ylim()[0], self.traces[k][it_max_score]],
                        ls="--",
                        color="black",
                    )
        plt.show()
