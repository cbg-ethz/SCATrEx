import numpy as np
from copy import deepcopy
from tqdm import tqdm
from time import time
from ..util import *

def search_callback(inf):
    return

class StructureSearch(object):
    def __init__(self, ntssb):

        self.tree = deepcopy(ntssb)

        self.traces = dict()
        self.traces['tree'] = []
        self.traces['score'] = []
        self.traces['move'] = []
        self.traces['temperature'] = []
        self.traces['accepted'] = []
        self.traces['times'] = []
        self.traces['n_nodes'] = []
        self.best_elbo = self.tree.elbo
        self.best_tree = deepcopy(self.tree)

    def run_search(self, n_iters=1000, n_iters_elbo=1000, thin=10, local=True, num_samples=1, step_size=0.05, verbose=True, tol=1e-6, mb_size=100, max_nodes=5, debug=False, callback=None, alpha=0.01, Tmax=10, anneal=False, restart_step=10,
                    moves=['add', 'merge', 'pivot_reattach', 'swap', 'subtree_reattach', 'globals', 'full'], merge_n_tries=5, opt=None, search_callback=None, add_rule='accept'):
        print(f'Will search for the maximum marginal likelihood tree with the following moves: {moves}\n')
        T = Tmax
        if not anneal:
            T = 1.

        if len(self.traces['score']) == 0:
            # Compute score of initial tree
            self.tree.reset_variational_parameters()
            init_baseline = jnp.mean(self.tree.data, axis=0)
            init_log_baseline = jnp.log(init_baseline / init_baseline[0])[1:]
            self.tree.root['node'].root['node'].log_baseline_mean = init_log_baseline + np.random.normal(0, .5, size=self.tree.data.shape[1]-1)
            self.tree.optimize_elbo(root_node=None, num_samples=num_samples, n_iters=n_iters_elbo*10, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
            self.tree.update_ass_logits(variational=True)
            self.tree.assign_to_best()
            self.best_elbo = self.tree.elbo
            self.best_tree = deepcopy(self.tree)

        init_elbo = self.tree.elbo
        init_root = self.tree.root
        move_id = 'add'
        n_merge = 0
        # Search tree
        p = np.array([1.] * len(moves))
        try:
            # p[np.where(np.array(moves)=='add')[0][0]] = 1.5
            p[np.where(np.array(moves)=='globals')[0][0]] = 0.5
        except:
            pass
        p = p / np.sum(p)
        for i in tqdm(range(n_iters)):
            start = time()
            try:
                nodes, mixture = self.tree.get_node_mixture()
                n_nodes = len(nodes)
                if np.mod(i, 5) == 0:
                    if np.sum(mixture < 0.1*1./n_nodes) > .3*n_nodes:
                        # Reduce probability of adding, and only add if score improves
                        p = np.array([1.] * len(moves))
                        p[np.where(np.array(moves)=='add')[0][0]] = 0.25 * 1/len(moves)
                        add_rule = 'improve'
                    else:
                        # Keep uniform move probability and always accept adds
                        p = np.array([1.] * len(moves))
                        add_rule = 'accept'
            except:
                pass
            try:
                if i < int(0.5 * n_iters):
                    p[np.where(np.array(moves)=='subtree_reattach')[0][0]] *= 0.2
            except:
                pass
            p = p / np.sum(p)
            if move_id == 'add':
                move_id = 'merge' # always try to merge after adding
            if move_id == 'merge':
                if n_merge == merge_n_tries:
                    n_merge = 0
                    move_id = 'full' # always update parameters after merging to rebase
                    # move_id = np.random.choice(moves, p=p)
                else:
                    n_merge += 1
                    move_id = 'merge'
            else:
                move_id = np.random.choice(moves, p=p)

            if move_id == 'add':
                init_root, init_elbo = self.add_node(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback)
            elif move_id == 'merge':
                init_root, init_elbo = self.merge_nodes(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback)
            elif move_id == 'pivot_reattach':
                init_root, init_elbo = self.pivot_reattach(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback)
            elif move_id == 'add_reattach_pivot':
                init_root, init_elbo = self.add_reattach_pivot(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback)
            elif move_id == 'swap':
                init_root, init_elbo = self.swap_nodes(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback)
            elif move_id == 'subtree_reattach':
                init_root, init_elbo = self.subtree_reattach(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback)
            elif move_id == 'globals':
                init_root = deepcopy(self.tree.root)
                init_elbo = self.tree.elbo
                self.tree.optimize_elbo(root_node=None, global_only=True, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
            elif move_id == 'full':
                init_root = deepcopy(self.tree.root)
                init_elbo = self.tree.elbo
                self.tree.optimize_elbo(root_node=None, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)


            if np.isnan(self.tree.elbo):
                print("Got NaN!")
                self.tree.root = deepcopy(init_root)
                self.tree.elbo = init_elbo
                print("Proceeding with previous tree and reducing step size.")
                step_size = 1e-3
                continue
            else:
                step_size = 1e-2


            # if anneal:
            #     if i/thin >= 0 and np.mod(i, thin) == 0:
            #         idx = int(i/thin)
            #         if Tmax != 1:
            #             T = Tmax - alpha*idx
            #             T = T * (1 + (self.tree.elbo - self.best_elbo)/self.tree.elbo)

            accepted = True

            if move_id == 'add':
                if add_rule == 'accept':
                    if self.tree.n_nodes < self.tree.max_nodes:
                        print(f'*Move ({move_id}) accepted. ({init_elbo} -> {self.tree.elbo})*')
                        if self.tree.elbo > self.best_elbo:
                            self.best_elbo = self.tree.elbo
                            self.best_tree = deepcopy(self.tree)
                            print(f'New best! {self.best_elbo}')
                else:
                    if (-(init_elbo - self.tree.elbo)/T) < np.log(np.random.rand()):
                        self.tree.root = deepcopy(init_root)
                        self.tree.elbo = init_elbo
                        accepted = False
                    else:
                        if self.tree.n_nodes < self.tree.max_nodes:
                            print(f'*Move ({move_id}) accepted. ({init_elbo} -> {self.tree.elbo})*')
                            if self.tree.elbo > self.best_elbo:
                                self.best_elbo = self.tree.elbo
                                self.best_tree = deepcopy(self.tree)
                                print(f'New best! {self.best_elbo}')
            else:
                if move_id != 'full' and move_id != 'globals' and (-(init_elbo - self.tree.elbo)/T) < np.log(np.random.rand()) or self.tree.n_nodes >= self.tree.max_nodes: # Rejected
                    self.tree.root = deepcopy(init_root)
                    self.tree.elbo = init_elbo
                    accepted = False
                else:                                                         # Accepted
                    print(f'*Move ({move_id}) accepted. ({init_elbo} -> {self.tree.elbo})*')
                    if self.tree.elbo > self.best_elbo:
                        self.best_elbo = self.tree.elbo
                        self.best_tree = deepcopy(self.tree)
                        print(f'New best! {self.best_elbo}')

            self.tree.plot_tree(super_only=False)
            self.traces['score'].append(self.tree.elbo)
            self.traces['move'].append(move_id)
            self.traces['n_nodes'].append(self.tree.n_nodes)
            self.traces['temperature'].append(T)
            self.traces['accepted'].append(accepted)
            self.traces['times'].append(time() - start)

            if search_callback is not None:
                search_callback(self)

            if anneal:
                if i/restart_step > 0 and np.mod(i, restart_step) == 0:
                    self.tree.root = deepcopy(self.best_tree.root)
                    self.tree.elbo = self.best_elbo

            if T == 0:
                break


            # Cull tree
            if np.random.rand() > 0.67:
                self.tree.cull_subtrees()

        return self.best_tree

    def add_node(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        nodes, target_probs = self.tree.get_node_data_sizes(normalized=True)
        target_probs /= np.sum(target_probs)
        node_idx = np.random.choice(range(len(nodes)), p=np.array(target_probs))
        node = nodes[node_idx]

        if verbose:
            print(f"Trying to add node below {node.label}")
        # Use only data around the node
        # data_indices = np.where(root['node'].data_probs > 1/np.sqrt(len(nodes)))[0]
        new_node = self.tree.add_node_to(node, optimal_init=True)

        local_node = None
        if local:
            local_node = new_node
        self.tree.optimize_elbo(local_node=new_node, root_node=None, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def merge_nodes(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        # Choose a subtree
        _, subtrees = self.tree.get_mixture()

        n_nodes = []
        nodes = []
        for subtree in subtrees:
            nodes.append(subtree.get_mixture()[1])
            n_nodes.append(len(nodes[-1]))
        n_nodes = np.array(n_nodes)
        # Only proceed if there is at least one subtree with mergeable nodes
        if np.any(n_nodes > 1):
            probs = n_nodes - 1
            probs = probs / np.sum(probs)

            # Choose a subtree with more than 1 node
            idx = np.random.choice(range(len(subtrees)), p=probs)
            subtree = subtrees[idx]
            nodes = nodes[idx]

            # Uniformly choose a first node A (which can't be the root)
            node_idx = np.random.choice(range(len(nodes[1:])), p=[1./len(nodes[1:])]*len(nodes[1:]))
            nodeA = nodes[1:][node_idx]

            def descend(root, done, ntree):
                if not done:
                    for i, child in enumerate(root['children']):
                        if child['node'] != nodeA:
                            done = descend(child, done, ntree)
                            if done:
                                break
                        else:
                            # child['node'] is nodeA
                            nodes = [sibling for sibling in root['children'] if sibling['node'] != child['node']]
                            nodes.append(root)

                            # Get similarities to nodeA
                            sims = [1./(np.mean(np.abs(nodeA.node_mean - node['node'].node_mean)) + 1e-8) for node in nodes]

                            # Choose nodeB proportionally to similarities
                            nodeB_root = np.random.choice(nodes, p=sims/np.sum(sims))
                            nodeB = nodeB_root['node']

                            if verbose:
                                print(f"Trying to merge {nodeA.label} to {nodeB.label}...")

                            ntree.merge_nodes(nodeA, nodeB)
                            ntree.optimize_elbo(unique_node=None, root_node=nodeB, run=True, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
                            if verbose:
                                print(f"{init_elbo} -> {ntree.elbo}")

                            return True

            descend(subtree.root, False, self.tree)

        return init_root, init_elbo

    def pivot_reattach(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        # Uniformly pick a subtree
        subtrees = self.tree.get_mixture()[1][1:] # without the root
        subtree = np.random.choice(subtrees, p=[1./len(subtrees)]*len(subtrees))
        init_pivot_node = subtree.root['node'].parent()
        init_pivot = init_pivot_node.label

        # Choose a pivot node from the parent subtree that isn't the current one
        weights, nodes = init_pivot_node.tssb.get_fixed_weights()
        # Only proceed if parent subtree has more than 1 node
        if len(nodes) > 1:
            # weights = [weight for i, weight in enumerate(weights) if nodes[i] != init_pivot_node]
            # weights = np.array(weights) / np.sum(weights)
            # Also use the similarity of the parent subtree's nodes' unobserved factors with the subtree root
            sims = [1./(np.mean(np.abs(subtree.root['node'].variational_parameters['locals']['unobserved_factors_mean'] - node.variational_parameters['locals']['unobserved_factors_mean'])) + 1e-8) for node in nodes]
            log_weights = [np.log(weights[i]) + np.log(sims[i]) for i, node in enumerate(nodes) if node != init_pivot_node]
            weights = np.exp(np.array(log_weights))
            weights = weights / np.sum(weights)
            nodes = [node for node in nodes if node != init_pivot_node] # remove the current pivot
            node_idx = np.random.choice(range(len(nodes)), p=weights)
            node = nodes[node_idx]

            # Update pivot
            self.tree.pivot_reattach_to(subtree, node)

            if verbose:
                print(f"Trying to set {node.label} as pivot of {subtree.label}")

            root_node = None
            if local:
                root_node = subtree.root['node']
            self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
            if verbose:
                print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def add_reattach_pivot(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        # Add a node below a subtree with children subtrees
        subtrees = self.tree.get_subtrees(get_roots=True)
        nonleaf_subtrees = [subtree for subtree in subtrees if len(subtree[1]['children']) > 0]
        # Pick a subtree
        parent_subtree = nonleaf_subtrees[np.random.choice(len(nonleaf_subtrees), p=[1./len(nonleaf_subtrees)]*len(nonleaf_subtrees))]

        # Pick a node in the parent subtree
        _, nodes, roots = parent_subtree[0].get_mixture(get_roots=True)
        node = np.random.choice(nodes, p=[1./len(nodes)]*len(nodes))
        pivot_node = self.tree.add_node_to(node.label, optimal_init=False)
        self.tree.plot_tree(super_only=False)

        # Pick one of the children subtrees
        subtrees = [subtree for subtree in parent_subtree[1]['children']]
        subtree = np.random.choice(subtrees, p=[1./len(subtrees)]*len(subtrees))
        init_pivot = subtree['node'].root['node'].parent().label

        # Update pivot
        self.tree.pivot_reattach_to(subtree['node'].label, pivot_node.label)

        if verbose:
            print(f"Trying to add node {pivot_node.label} and setting it as pivot of {subtree['node'].label}")

        root_node = None
        if local:
            root_node = pivot_node
        self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def subtree_reattach(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None):
        """
        Move a subtree to a different clone
        """
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        subtrees = self.tree.get_subtrees(get_roots=True)

        # Pick a subtree with more than 1 node
        in_subtrees = [subtree[1] for subtree in subtrees if len(subtree[0].root['children']) > 0]

        # If there is any subtree with more than 1 node, proceed
        if len(in_subtrees) > 0:
            # Choose one subtree
            subtreeA = np.random.choice(in_subtrees, p=[1./len(in_subtrees)]*len(in_subtrees))

            # Choose one of its nodes uniformly which is not the root
            node_weights, nodes, roots = subtreeA['node'].get_mixture(get_roots=True)
            nodeA_idx = np.random.choice(len(roots[1:]), p=[1./len(roots[1:])]*len(roots[1:])) + 1
            nodeA_parent_idx = np.where(np.array(nodes) == nodes[nodeA_idx].parent())[0][0]
            def descend(root):
                ns = [root['node']]
                for child in root['children']:
                    child_node = descend(child)
                    ns.extend(child_node)
                return ns
            nodes_below_nodeA = descend(roots[nodeA_idx])

            # Check if there is a pivot here
            for subtree_child in subtreeA['children']:
                for n in nodes_below_nodeA:
                    if subtree_child['pivot_node'] == n:
                        subtree_child['node'].root['node'].set_parent(subtreeA['node'].root['node'])
                        subtree_child['pivot_node'] = subtreeA['node'].root['node']
                        break

            # Choose another subtree that's similar to the subtree's top node
            rem_subtrees = [s[1] for s in subtrees if s[1]['node'] != subtreeA['node']]
            sims = [1./(np.mean(np.abs(roots[nodeA_idx]['node'].node_mean - s['node'].root['node'].node_mean)) + 1e-8) for s in rem_subtrees]
            new_subtree = np.random.choice(rem_subtrees, p=sims/np.sum(sims))

            # Move subtree
            roots[nodeA_idx]['node'].set_parent(new_subtree['node'].root['node'])
            roots[nodeA_idx]['node'].set_mean(variational=True)
            roots[nodeA_idx]['node'].tssb = new_subtree['node']
            def descend(root):
                for child in root.children():
                    child.tssb = new_subtree['node']
            descend(roots[nodeA_idx]['node'])
            new_subtree['node'].root['children'].append(roots[nodeA_idx])
            new_subtree['node'].root['sticks'] = np.vstack([new_subtree['node'].root['sticks'], 1.])

            childnodes = np.array([n['node'] for n in roots[nodeA_parent_idx]['children']])
            tokeep = np.where(childnodes != roots[nodeA_idx]['node'])[0].astype(int).ravel()
            roots[nodeA_parent_idx]['sticks']   = roots[nodeA_parent_idx]['sticks'][tokeep]
            roots[nodeA_parent_idx]['children'] = list(np.array(roots[nodeA_parent_idx]['children'])[tokeep])

            if verbose:
                print(f"Trying to set {roots[nodeA_idx]['node'].label} below {new_subtree['node'].label}")

            # self.tree.reset_variational_parameters(variances_only=True)
            # init_baseline = jnp.mean(self.tree.data, axis=0)
            # init_log_baseline = jnp.log(init_baseline / init_baseline[0])[1:]
            # self.tree.root['node'].root['node'].log_baseline_mean = init_log_baseline + np.random.normal(0, .5, size=self.tree.data.shape[1]-1)
            self.tree.optimize_elbo(root_node=roots[nodeA_idx]['node'], num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)

            if verbose:
                print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def swap_nodes(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        def tssb_swap(tssb, children_trees, ntree, n_iters):
            weights, nodes = tssb.get_mixture();

            empty_root = False
            if len(nodes)>1:
                nodeA, nodeB = np.random.choice(nodes, replace=False, size=2)
                if len(nodes[0].data) == 0:
                    print('Swapping root')
                    empty_root = True
                    nodeA = nodes[0]
                    nodeB = np.random.choice(nodes[1:])

                if verbose:
                    print(f"Trying to swap {nodeA.label} with {nodeB.label}...")
                self.tree.swap_nodes(nodeA, nodeB)

                if empty_root:
                    # self.tree = deepcopy(ntree)
                    print(f"Swapped {nodeA.label} with {nodeB.label}")
                else:
                    # ntree = self.compute_expected_score(ntree, n_burnin=n_burnin, n_samples=n_samples, thin=thin, global_params=global_params, compound=compound)
                    # ntree.reset_variational_parameters(variances_only=True)
                    # init_baseline = jnp.mean(ntree.data, axis=0)
                    # init_log_baseline = jnp.log(init_baseline / init_baseline[0])[1:]
                    # ntree.root['node'].root['node'].log_baseline_mean = init_log_baseline + np.random.normal(0, .5, size=ntree.data.shape[1]-1)
                    root_node = nodes[0]
                    if nodeB == nodeA.parent():
                        root_node = nodeB
                    elif nodeA == nodeB.parent():
                        root_node = nodeA
                    if not local:
                        root_node = None
                    if root_node:
                        if root_node.parent() is None:
                            root_node = None # Update everything!
                            n_iters *= 12 # Big change, so give time to converge
                    ntree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
                    if verbose:
                        print(f"{init_elbo} -> {ntree.elbo}")

        def descend(root, subtree, ntree, done):
            if not done:
                if root['node'] == subtree:
                    tssb_swap(subtree, root['children'], ntree, n_iters)
                    return True
                else:
                    for index, child in enumerate(root['children']):
                        done = descend(child, subtree, ntree, done)
                        if done:
                            break


        # Randomly decide between within TSSB swap or unrestricted in ntssb
        within_tssb = np.random.binomial(1, 0.5)

        if within_tssb:
            # Uniformly pick a subtree with more than 1 node
            subtrees = self.tree.get_mixture()[1]
            subtrees = [subtree for subtree in subtrees if len(subtree.root['children']) > 0]
            if len(subtrees) > 0:
                subtree = np.random.choice(subtrees, p=[1./len(subtrees)]*len(subtrees))

                descend(self.tree.root, subtree, self.tree, False)
        else:
            nodes = self.tree.get_nodes()
            nodes = nodes[1:] # without root

            # Randomly decide between parent-child and unrestricted
            unrestricted = np.random.binomial(1, 0.1)
            if unrestricted:
                nodeA, nodeB = np.random.choice(nodes, replace=False, size=2)
            else:
                nodeA = np.random.choice(nodes)
                nodeB = nodeA.parent()

            if nodeB is not None:
                if verbose:
                    print(f"Trying to swap {nodeA.label} with {nodeB.label}...")
                self.tree.swap_nodes(nodeA, nodeB)
                root_node = nodeB
                if unrestricted:
                    if nodeA == nodeB.parent():
                        root_node = nodeA
                    elif nodeB == nodeA.parent():
                        root_node = nodeB
                    else:
                        root_node = self.tree.root['node'].root['node']
                if not local:
                    root_node = None
                self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=callback)
                if verbose:
                    print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo
