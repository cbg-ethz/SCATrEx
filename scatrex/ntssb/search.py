import numpy as np
from copy import deepcopy
from tqdm.auto import trange
from time import time
import matplotlib.pyplot as plt
import pandas as pd

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
        self.mcmc = dict()

    def run_mcmc(self, n_samples=100, n_burnin=10, n_thin=10, store_trees=False, memoized=True, n_opt_steps=50, seed=42, **pr_kwargs):
        """
        Run MCMC chain for each TSSB where we proposed prune-reattach moves, update the marginal likelihood of the tree and accept/reject with metropolis-hastings acceptance ratio. 
        Run each TSSB in parallel?
        """
        # Do not store the tree objects, just store the current and best tree objects, and keep the trace of number of times each node is parent of every other node, and the node_dict traces
        # For fixed roots and fixed node-subtree attachments...
        for ref_tssb in self.best_tree.get_tree_roots(): # Parallelize here across different processes... Maybe one GPU per chain for the param opts
            tssb = deepcopy(ref_tssb["node"])
            tssb.compute_elbo(memoized=memoized)
            proposed_tssb = deepcopy(tssb)
            if tssb.label not in self.mcmc:
                self.mcmc[tssb.label] = dict(
                    elbos=[],
                    node_dicts=[],
                    trees=[],
                    best_tree=deepcopy(tssb),
                    ar=0.
                )
                node_dict = tssb.get_node_dict() 
                nodes = node_dict.keys()
                self.mcmc[tssb.label]["posterior_counts"] = pd.DataFrame(index=nodes, columns=nodes, data=np.zeros((len(nodes), len(nodes))).astype(int)) #dict(zip(nodes, [dict(zip(nodes, [0] * len(nodes)))] * len(nodes)))
                self.mcmc[tssb.label]["posterior_freqs"] = pd.DataFrame(index=nodes, columns=nodes, data=np.zeros((len(nodes), len(nodes))).astype(float))

            if tssb.n_nodes <= 2:
                continue

            i = 0
            best_i = 0
            n_accepted = 0 
            key = jax.random.PRNGKey(seed)
            t = trange(n_samples, desc=f'Running MCMC on tree {tssb.label}', leave=True)
            while i < n_samples:
                # Sample
                key, valid, accepted, proposed_tssb, tssb = self.prune_reattach(key, proposed_tssb, tssb, update_names=False, memoized=memoized, n_steps=n_opt_steps, **pr_kwargs)
                if not valid:
                    continue

                # MCMC info
                if accepted:
                    n_accepted += 1
                node_dict = tssb.get_node_dict()
                self.mcmc[tssb.label]["elbos"].append(tssb.elbo)
                self.mcmc[tssb.label]["node_dicts"].append(node_dict)
                if tssb.elbo > self.mcmc[tssb.label]["best_tree"].elbo:    
                    best_i = i
                    self.mcmc[tssb.label]["best_tree"] = deepcopy(tssb)
                if store_trees:
                    self.mcmc[tssb.label]["trees"].append(deepcopy(tssb))
                
                if i > n_burnin:
                    self.mcmc[tssb.label]["ar"] = n_accepted / i
                    t.set_description(f'MCMC for {tssb.label}: iteration: {i}, acceptance ratio: {self.mcmc[tssb.label]["ar"]:0.4g}, best: {self.mcmc[tssb.label]["best_tree"].elbo:0.4g} (at {best_i})')
                    
                    # Compute tree statistics from sample
                    if i % n_thin == 0:
                        for node_src in node_dict:
                            for node_prt in node_dict:
                                if node_dict[node_src]['parent'] == node_prt:
                                    self.mcmc[tssb.label]["posterior_counts"].loc[node_src, node_prt] += 1

                i += 1
                t.update()
                t.refresh() 

            # Update names in table to use the ones in the best tree
            old_to_new = self.mcmc[tssb.label]["best_tree"].set_node_names(root_name=tssb.label, return_map=True)
            self.mcmc[tssb.label]["posterior_counts"] = self.mcmc[tssb.label]["posterior_counts"].rename(columns=old_to_new, index=old_to_new)

            # Normalize
            self.mcmc[tssb.label]["posterior_freqs"] = self.mcmc[tssb.label]["posterior_counts"]/np.sum(self.mcmc[tssb.label]["posterior_counts"],axis=1).values[:,None]

            logger.info(f"MCMC for {tssb.label} completed")

        # Re-assemble NTSSB from the best TSSBs
        def descend(root):
            for tssb_label in self.mcmc:
                if root["label"] == tssb_label:
                    root["node"] = deepcopy(self.mcmc[tssb_label]["best_tree"])
                    for child in root['children']:
                        descend(child)
                    break
        descend(self.best_tree.root)


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
                self.pr_merge(subkey, n_epochs=n_epochs, memoized=memoized, mc_samples=mc_samples, step_size=step_size, moves_per_tssb=moves_per_tssb, update_roots=update_roots, update_globals=update_globals,
                                 update_outer_ass=update_outer_ass)

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
        self.traces['tree'].append(deepcopy(self.tree))
        # Update parameters in n_epochs passes through the data, interleaving node updates with local batch updates
        self.tree.learn_params(int(n_epochs), update_roots=update_roots, mc_samples=mc_samples, 
                                step_size=step_size, memoized=memoized, update_outer_ass=update_outer_ass, ass_anneal=1.)
        # self.tree.learn_params(int(n_epochs/2), update_roots=update_roots, mc_samples=mc_samples, 
        #                         step_size=step_size, memoized=memoized, update_outer_ass=update_outer_ass, ass_anneal=1.)        
        self.tree.compute_elbo(memoized=memoized)
        self.traces['tree'].append(deepcopy(self.tree))
        self.proposed_tree = deepcopy(self.tree)
        
        # Merge: traverse the tree and propose merges and accept/reject based on their summary statistics (reliable)
        self.merge(key, moves_per_tssb=int(moves_per_tssb*10), memoized=memoized, update_globals=update_globals,
                n_epochs=n_epochs, mc_samples=mc_samples, step_size=step_size)
        self.traces['tree'].append(deepcopy(self.tree))

    def pr_merge(self, key, moves_per_tssb=1, n_epochs=100, update_roots=False, mc_samples=10, step_size=0.01, memoized=True, update_globals=False,
                    update_outer_ass=False):
        # PR: move nodes around and accept
        changed = self.prune_reattach(key, moves_per_tssb=moves_per_tssb, n_epochs=n_epochs, mc_samples=mc_samples, step_size=step_size)
        # if changed:
        #     self.traces['tree'].append(deepcopy(self.tree))
        #     # Update parameters in n_epochs passes through the data, interleaving node updates with local batch updates
        #     self.tree.learn_params(int(n_epochs), update_roots=update_roots, mc_samples=mc_samples, 
        #                             step_size=step_size, memoized=memoized, update_outer_ass=update_outer_ass, ass_anneal=1.)
        #     self.tree.compute_elbo(memoized=memoized)
        #     self.traces['tree'].append(deepcopy(self.tree))
        #     self.proposed_tree = deepcopy(self.tree)
            
        #     # Merge: traverse the tree and propose merges and accept/reject based on their summary statistics (reliable)
        #     self.merge(key, moves_per_tssb=int(moves_per_tssb*2), memoized=memoized, update_globals=update_globals,
        #             n_epochs=n_epochs, mc_samples=mc_samples, step_size=step_size)
        #     self.traces['tree'].append(deepcopy(self.tree))

    def birth(self, key, moves_per_tssb=1):
        """
        Spawn `moves_per_tssb=1` nodes. First select targets, and then add nodes. Gives a more local search
        """
        n_births = self.proposed_tree.n_nodes * moves_per_tssb
        targets = []
        for root in self.proposed_tree.get_tree_roots():
            tssb = root['node']
            for _ in range(moves_per_tssb):
                key, subkey = jax.random.split(key)
                _, _, target = tssb.find_node_uniform(subkey, include_leaves=True, return_parent=False) 
                # target = self.proposed_tree.get_node(u, key=subkey, uniform=True, variational=True)
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
        Traverse the trees and propose and accept/reject merges as we go using local suff stats. 
        Parallelize over TSSBs
        """
        proposed_tssbs = self.proposed_tree.get_tree_roots()
        for tssb_label in [a['label'] for a in proposed_tssbs]:
            for _ in range(moves_per_tssb):
                # Get tssb
                proposed_tssbs = self.proposed_tree.get_tree_roots()
                proposed_tssbs = dict(zip([a['label'] for a in proposed_tssbs], [a['node'] for a in proposed_tssbs]))
                tssb = proposed_tssbs[tssb_label]
                if tssb.n_nodes > 1:
                    key, subkey = jax.random.split(key)
                    # u = jax.random.uniform(subkey)
                    # parent = self.proposed_tree.get_node(u, key=subkey, uniform=True, include_leaves=False) # get non-leaf node, without accounting for weights
                    _, _, parent = tssb.find_node_uniform(subkey, include_leaves=False) 
                    # proposed_tssb = parent['node'].tssb
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

    def prune_reattach(self, key, proposed_tssb, tssb, n_tries=5, memoized=True, update_names=True, **learn_kwargs):
        changed = False
        accepted = False
        if tssb.n_nodes > 1:
            for _ in range(n_tries):
                key, subkey = jax.random.split(key)
                _, source_path, source, source_parent = proposed_tssb.find_node_uniform(subkey, include_leaves=True, return_parent=True) 
                if len(source_path) == 0: # Can't do root
                    continue
                
                key, subkey = jax.random.split(key)
                _, target_path, target = proposed_tssb.find_node_uniform(subkey, include_leaves=True) 
                if len(target_path) >= len(source_path):
                    if source_path == target_path[:len(source_path)]: # Can't swap parent-child
                        continue
                
                if target == source_parent: # Don't re-attach to same place
                    continue
            
                # print(source['node'].label, target['node'].label)
                proposed_tssb.prune_reattach(source_parent, source, target, update_names=update_names)
                
                # Quick parameter update
                proposed_tssb.update_stick_params(memoized=memoized)
                proposed_tssb.update_node_kernel_params(key, root=source, memoized=memoized, update_state=False, return_trace=False, **learn_kwargs)
                proposed_tssb.update_pivot_probs()

                proposed_tssb.compute_elbo(memoized=memoized)

                # MH acceptance probability
                key, subkey = jax.random.split(key)
                u = jax.random.uniform(key)

                if u < jnp.exp(proposed_tssb.elbo - tssb.elbo):
                    tssb = deepcopy(proposed_tssb)
                    accepted = True
                else:
                    proposed_tssb = deepcopy(tssb)

                changed = True

                break

        return key, changed, accepted, proposed_tssb, tssb


    # def prune_reattach(self, key, moves_per_tssb=1, memoized=True, n_epochs=10, **learn_kwargs):
    #     """
    #     Prune subtree and reattach somewhere else within the same TSSB. 
    #     """
    #     changed = False
    #     for _ in range(5):
    #         for root in self.proposed_tree.get_tree_roots():
    #             tssb = root['node']
    #             if tssb.n_nodes > 1:
    #                 for _ in range(moves_per_tssb):
    #                     key, subkey = jax.random.split(key)
    #                     _, source_path, source, source_parent = tssb.find_node_uniform(subkey, include_leaves=True, return_parent=True) 
    #                     if len(source_path) == 0: # Can't do root
    #                         continue
                        
    #                     key, subkey = jax.random.split(key)
    #                     _, target_path, target = tssb.find_node_uniform(subkey, include_leaves=True) 
    #                     if len(target_path) >= len(source_path):
    #                         if source_path == target_path[:len(source_path)]: # Can't swap parent-child
    #                             continue
                        
    #                     if target == source_parent: # Don't re-attach to same place
    #                         continue
                    
    #                     self.traces['tree'].append(deepcopy(self.proposed_tree))
    #                     print("Doing prune reattach!")
    #                     print(source['node'].label, target['node'].label)
    #                     tssb.prune_reattach(source_parent, source, target)
                        
    #                     self.traces['tree'].append(deepcopy(self.proposed_tree))

    #                     # Quick parameter update
    #                     tssb.update_stick_params(memoized=memoized)
    #                     tssb.update_node_kernel_params(key, root=source, memoized=memoized, n_steps=50, update_state=False, return_trace=False, **learn_kwargs)

    #                     # tssb.update_stick_params(memoized=memoized)
    #                     # for i in range(n_epochs):
    #                     #     key, subkey = jax.random.split(key)
    #                     #     tssb.update_node_params(key, root=target, memoized=memoized, i=i, **learn_kwargs)
    #                     self.proposed_tree.compute_elbo(memoized=memoized)

    #                     # MH acceptance probability
    #                     key, subkey = jax.random.split(key)
    #                     u = jax.random.uniform(key)

    #                     if u < jnp.exp(self.proposed_tree.elbo - self.tree.elbo):
    #                         self.tree = deepcopy(self.proposed_tree)
    #                         print("Accepted")
    #                     else:
    #                         self.proposed_tree = deepcopy(self.tree)

    #                     self.traces['tree'].append(deepcopy(self.tree))

    #                     changed = True
    #         if changed:
    #             break

    #     return changed

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
