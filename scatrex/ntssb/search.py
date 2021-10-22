import numpy as np
from copy import deepcopy
from tqdm import tqdm
from time import time
from ..util import *
from jax.api import jit
from jax.experimental.optimizers import adam
import matplotlib.pyplot as plt

def search_callback(inf):
    return

MOVE_WEIGHTS = {'add':1,
                'merge':3,
                'prune_reattach':1,
                'pivot_reattach':1,
                'swap':1,
                'subtree_reattach':1,
                'push_subtree':1,
                'perturb_node':1,
                'perturb_globals':1}

class StructureSearch(object):
    def __init__(self, ntssb):

        self.tree = deepcopy(ntssb)

        self.traces = dict()
        self.traces['tree'] = []
        self.traces['elbo'] = []
        self.traces['score'] = []
        self.traces['move'] = []
        self.traces['temperature'] = []
        self.traces['accepted'] = []
        self.traces['times'] = []
        self.traces['n_nodes'] = []
        self.best_elbo = self.tree.elbo
        self.best_tree = deepcopy(self.tree)
        self.opt_triplet = None

    def init_optimizer(self, step_size=0.01, opt=adam):
        opt_init, opt_update, get_params = opt(step_size=step_size)
        get_params = jit(get_params)
        opt_update = jit(opt_update)
        opt_init = jit(opt_init)
        self.opt_triplet = (opt_init, opt_update, get_params)

    def plot_traces(self, keys=None, figsize=(16,10), highlight_max=True, highlight_s=100, highlight_color='red'):
        if keys is None:
            keys = list(self.traces['keys'])
        if highlight_max:
            it_max_score = np.argmax(self.traces['elbo'])

        fig, ax_list = plt.subplots(len(keys), 1, sharex=True, figsize=figsize)
        for i, k in enumerate(keys):
            ax_list[i].plot(self.traces[k])
            ax_list[i].set_title(k)
            if k == 'n_nodes':
                ax_list[i].set_ylim(0, self.tree.max_nodes)
            if highlight_max:
                ax_list[i].scatter(it_max_score, self.traces[k][it_max_score], s=highlight_s, color=highlight_color)
                ax_list[i].plot([0, it_max_score], [self.traces[k][it_max_score], self.traces[k][it_max_score]], ls='--', color='black')
                if type(self.traces[k][0]) is float or type(self.traces[k][0]) is int or k == 'elbo' or k == 'score':
                    ax_list[i].plot([it_max_score, it_max_score], [ax_list[i].get_ylim()[0], self.traces[k][it_max_score]], ls='--', color='black')
        plt.show()

    def run_search(self, n_iters=1000, n_iters_elbo=1000, factor_delay=0, posterior_delay=0, global_delay=0, joint_init=True, thin=10, local=True, num_samples=1, step_size=0.001, verbose=True, tol=1e-6, mb_size=100, max_nodes=5, debug=False, callback=None, alpha=0.01, Tmax=10, anneal=False, restart_step=10,
                    move_weights=None, merge_n_tries=5, opt=adam, search_callback=None, add_rule='accept', add_rule_thres=.5, **callback_kwargs):

        if move_weights is None:
            move_weights = MOVE_WEIGHTS
        moves = list(move_weights.keys())
        move_weights = list(move_weights.values())

        print(f'Will search for the maximum marginal likelihood tree with the following moves: {moves}\n')

        self.init_optimizer(step_size=step_size, opt=opt)

        self.tree.max_nodes = len(self.tree.input_tree_dict.keys()) * max_nodes # upper bound on number of nodes

        mb_size = min(mb_size, self.tree.data.shape[0])

        score_type = 'elbo'
        if posterior_delay > 0:
            score_type = 'll'

        main_step_size = step_size
        T = Tmax
        if not anneal:
            T = 1.

        if not local and global_delay > 0:
            local = True

        n_factors = self.tree.root['node'].root['node'].num_global_noise_factors

        init_baseline = np.median(self.tree.data, axis=0)
        init_baseline = init_baseline / np.mean(self.tree.input_tree.adata.X/2, axis=0)
        init_baseline = init_baseline / np.std(init_baseline)
        init_baseline[0] = 1.
        init_log_baseline = np.log(init_baseline[1:] + 1e-6)
        init_log_baseline = np.clip(init_log_baseline, -2, 2)

        if len(self.traces['score']) == 0:
            if n_factors > 0 and factor_delay > 0:
                self.tree.root['node'].root['node'].num_global_noise_factors = 0

            # Compute score of initial tree
            self.tree.reset_variational_parameters()
            self.tree.root['node'].root['node'].variational_parameters['globals']['log_baseline_mean'] = init_log_baseline
            self.tree.optimize_elbo(root_node=None, sticks_only=True, num_samples=num_samples, n_iters=n_iters_elbo*10, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback)

            # full update -- maybe without globals?
            root_node_init = self.tree.root['node'].root['node']
            if joint_init:
                root_node_init = None
            self.tree.optimize_elbo(root_node=root_node_init, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback)
            self.tree.plot_tree(super_only=False)
            self.tree.update_ass_logits(variational=True)
            self.tree.assign_to_best()
            self.best_elbo = self.tree.elbo
            self.best_tree = deepcopy(self.tree)

        init_score = self.tree.elbo if score_type == 'elbo' else self.tree.ll
        init_root = self.tree.root
        move_id = 'full'
        n_merge = 0
        # Search tree
        p = np.array(move_weights)
        p = p / np.sum(p)
        for i in tqdm(range(n_iters)):
            start = time()
            try:
                if np.mod(i, 5) == 0:
                    nodes, mixture = self.tree.get_node_mixture()
                    n_nodes = len(nodes)
                    if np.sum(mixture < 0.1*1./n_nodes) > np.ceil(n_nodes/3) or n_nodes > self.tree.max_nodes * add_rule_thres:
                        # Reduce probability of adding, and only add if score improves
                        # p = np.array(move_weights)
                        # p[np.where(np.array(moves)=='add')[0][0]] = 0.25 * 1/len(moves)
                        add_rule = 'improve'
                    else:
                        # Keep uniform move probability and always accept adds
                        p = np.array(move_weights)
                        add_rule = 'accept'
            except:
                pass

            p = p / np.sum(p)
            # if move_id == 'add':
            #    move_id = 'merge' # always try to merge after adding # not -- must give a chance for the noise factors to be updated too
            # else:
            move_id = np.random.choice(moves, p=p)

            if i == factor_delay and n_factors > 0 and self.tree.root['node'].root['node'].num_global_noise_factors == 0:
                self.tree = deepcopy(self.best_tree)
                self.tree.root['node'].root['node'].num_global_noise_factors = n_factors
                self.tree.root['node'].root['node'].init_noise_factors()
                move_id = 'full'

            if step_size < main_step_size:
                move_id = 'reset_globals'

            nits_check = int(np.max([10, .05*n_iters]))
            if i > 10 and score_type == 'elbo' and np.sum(np.array(self.traces['accepted'][-nits_check:]) == False) == nits_check:
                print(f"No moves accepted in {nits_check} iterations. Using ll for 10 iterations.")
                posterior_delay = i + 10

            if i < posterior_delay:
                score_type = 'll'
            elif i == posterior_delay:
                # Go back to best in terms of ELBO
                self.tree.root = deepcopy(self.best_tree.root)
                self.tree.elbo = self.best_elbo
                score_type = 'elbo'

            if global_delay > 0 and i > global_delay:
                local = False

            init_elbo = self.tree.elbo
            init_score = self.tree.elbo if score_type == 'elbo' else self.tree.ll

            self.tree.n_nodes = len(self.tree.get_nodes())

            if move_id == 'add' and self.tree.n_nodes < self.tree.max_nodes-1:
                init_root, init_elbo = self.add_node(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'merge':
                init_root, init_elbo = self.merge_nodes(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'prune_reattach':
                init_root, init_elbo = self.prune_reattach(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'pivot_reattach':
                init_root, init_elbo = self.pivot_reattach(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'add_reattach_pivot':
                init_root, init_elbo = self.add_reattach_pivot(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, mb_size=mb_size, max_nodes=max_nodes, debug=debug, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'swap':
                init_root, init_elbo = self.swap_nodes(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'subtree_reattach':
                init_root, init_elbo = self.subtree_reattach(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'subtree_pivot_reattach':
                init_root, init_elbo = self.subtree_pivot_reattach(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'push_subtree' and self.tree.n_nodes < self.tree.max_nodes-1:
                init_root, init_elbo = self.push_subtree(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'extract_pivot' and self.tree.n_nodes < self.tree.max_nodes-1:
                init_root, init_elbo = self.extract_pivot(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'perturb_node':
                init_root, init_elbo = self.perturb_node(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'clean_node':
                init_root, init_elbo = self.clean_node(local=local, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, step_size=step_size, verbose=verbose, tol=tol, debug=debug, mb_size=mb_size, max_nodes=max_nodes, opt=opt, callback=callback, **callback_kwargs)
            elif move_id == 'reset_globals':
                init_root = deepcopy(self.tree.root)
                init_elbo = self.tree.elbo
                self.tree.root['node'].root['node'].variational_parameters['globals']['cell_noise_mean'] *= 0. # shrink
                self.tree.root['node'].root['node'].variational_parameters['globals']['cell_noise_log_std'] *= 0. # shrink
                self.tree.root['node'].root['node'].variational_parameters['globals']['noise_factors_mean'] *= 0. # shrink
                self.tree.root['node'].root['node'].variational_parameters['globals']['noise_factors_log_std'] *= 0. # shrink
                self.tree.root['node'].root['node'].variational_parameters['globals']['log_baseline_mean'] = init_log_baseline # reset to initial
                self.tree.root['node'].root['node'].variational_parameters['globals']['log_baseline_log_std'] *= 0. # reset to initial
                self.tree.root['node'].root['node'].variational_parameters['locals']['unobserved_factors_mean'] *= 0. # reset to initial
                self.tree.root['node'].root['node'].variational_parameters['locals']['unobserved_factors_log_std'] *= 0. # reset to initial
                # self.tree.root['node'].root['node'].variational_parameters['globals']['log_baseline_log_std'] *= 0. # allow more variation
                self.tree.optimize_elbo(root_node=None, global_only=False, num_samples=num_samples, n_iters=n_iters_elbo*10, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=None, **callback_kwargs)
            elif move_id == 'globals':
                init_root = deepcopy(self.tree.root)
                init_elbo = self.tree.elbo
                self.tree.optimize_elbo(root_node=None, global_only=True, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=None, **callback_kwargs)
            elif move_id == 'full':
                init_root = deepcopy(self.tree.root)
                init_elbo = self.tree.elbo
                self.tree.optimize_elbo(root_node=None, num_samples=num_samples, n_iters=n_iters_elbo, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, callback=None, **callback_kwargs)

            if np.isnan(self.tree.elbo):
                print("Got NaN!")
                self.tree.root = deepcopy(init_root)
                self.tree.elbo = init_elbo
                print("Proceeding with previous tree, reducing step size and doing `reset_globals`.")
                step_size = step_size * 0.1
                if step_size < 1e-4:
                    print("Step size is becoming small. Fetching best tree with noise factors.")
                    self.tree.root = deepcopy(self.best_tree.root)
                    self.tree.elbo = self.best_elbo
                    if self.tree.root['node'].root['node'].num_global_noise_factors == 0 and n_factors > 0 and i > factor_delay:
                        self.tree.root['node'].root['node'].num_global_noise_factors = n_factors
                        self.tree.root['node'].root['node'].init_noise_factors()
                if step_size < 1e-6:
                    raise ValueError("Step size became too small due to too many NaNs!")
                self.init_optimizer(step_size=step_size)
                continue
            else:
                if step_size != main_step_size:
                    step_size = main_step_size
                    self.init_optimizer(step_size=step_size)


            # if anneal:
            #     if i/thin >= 0 and np.mod(i, thin) == 0:
            #         idx = int(i/thin)
            #         if Tmax != 1:
            #             T = Tmax - alpha*idx
            #             T = T * (1 + (self.tree.elbo - self.best_elbo)/self.tree.elbo)


            new_score = self.tree.elbo if score_type == 'elbo' else self.tree.ll

            accepted = True

            if move_id == 'full' or move_id == 'globals':
                accepted = False

            if self.tree.n_nodes >= self.tree.max_nodes:
                self.tree.root = deepcopy(init_root)
                self.tree.elbo = init_elbo
                accepted = False
            elif move_id == 'add':
                if add_rule == 'accept' and score_type == 'elbo': # only accept immediatly if using ELBO to score
                    print(f'*Move ({move_id}) accepted. ({init_elbo} -> {self.tree.elbo})*')
                    if self.tree.elbo > self.best_elbo:
                        self.best_elbo = self.tree.elbo
                        self.best_tree = deepcopy(self.tree)
                        print(f'New best! {self.best_elbo}')
                else:
                    if (-(init_score - new_score)/T) < np.log(np.random.rand()):
                        self.tree.root = deepcopy(init_root)
                        self.tree.elbo = init_elbo
                        accepted = False
                    else:
                        print(f'*Move ({move_id}) accepted. ({init_elbo} -> {self.tree.elbo})*')
                        if self.tree.elbo > self.best_elbo:
                            self.best_elbo = self.tree.elbo
                            self.best_tree = deepcopy(self.tree)
                            print(f'New best! {self.best_elbo}')
            elif move_id != 'full' and move_id != 'globals':
                if (-(init_score - new_score)/T) < np.log(np.random.rand()):
                    self.tree.root = deepcopy(init_root)
                    self.tree.elbo = init_elbo
                    accepted = False
                else: # Accepted
                    print(f'*Move ({move_id}) accepted. ({init_elbo} -> {self.tree.elbo})*')
                    if self.tree.elbo > self.best_elbo:
                        self.best_elbo = self.tree.elbo
                        self.best_tree = deepcopy(self.tree)
                        print(f'New best! {self.best_elbo}')

            if i == factor_delay and n_factors > 0:
                print("Setting current tree with complete number of factors as the best.")
                self.best_elbo = self.tree.elbo
                self.best_tree = deepcopy(self.tree)
                print(f'New best! {self.best_elbo}')

            score = self.tree.elbo if score_type == 'elbo' else self.tree.ll
            self.traces['elbo'].append(self.tree.elbo)
            self.traces['score'].append(score)
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

        self.tree.plot_tree(super_only=False)
        self.best_tree.plot_tree(super_only=False)
        self.best_tree.update_ass_logits(variational=True)
        self.best_tree.assign_to_best()
        return self.best_tree

    def add_node(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
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

        # Decide wether to initialize from a factor
        from_factor = False
        factor_idx = None
        if self.tree.root['node'].root['node'].num_global_noise_factors > 0:
            if len(nodes) < len(self.tree.input_tree_dict.keys()) * 2:
                p = 0.8
            else:
                p = 0.5
            from_factor = bool(np.random.binomial(1, p))

        if from_factor:
            if verbose:
                print(f"Initializing new node from noise factor")
            # Choose factor that the data in the node like
            cells_in_node = np.where(np.array(self.tree.assignments) == node)
            factor_idx = np.argmax(np.mean(np.abs(self.tree.root['node'].root['node'].variational_parameters['globals']['cell_noise_mean'][cells_in_node]), axis=0))
            # factor_idx = np.argmax(np.var(self.tree.root['node'].root['node'].variational_parameters['globals']['noise_factors_mean'], axis=1))

        new_node = self.tree.add_node_to(node, optimal_init=True, factor_idx=factor_idx)

        local_node = None
        if local:
            local_node = new_node

        if from_factor:
            # Remove factor
            self.tree.root['node'].root['node'].variational_parameters['globals']['factor_precision_log_means'][factor_idx] = np.log(self.tree.root['node'].root['node'].global_noise_factors_precisions_shape)
            self.tree.root['node'].root['node'].variational_parameters['globals']['factor_precision_log_stds'][factor_idx] = -1.
            self.tree.root['node'].root['node'].variational_parameters['globals']['noise_factors_mean'][factor_idx] *= 0.0
            self.tree.root['node'].root['node'].variational_parameters['globals']['noise_factors_log_std'][factor_idx] = -1
            self.tree.root['node'].root['node'].variational_parameters['globals']['cell_noise_mean'][:,factor_idx] = 0.0
            self.tree.root['node'].root['node'].variational_parameters['globals']['cell_noise_log_std'][:,factor_idx] = -1
            self.tree.optimize_elbo(local_node=None, root_node=None, num_samples=num_samples, n_iters=2*n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
        else:
            if node.parent() is None: # if root, need to update also global factors
                self.tree.optimize_elbo(local_node=None, root_node=None, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
            else:
                self.tree.optimize_elbo(local_node=local_node, root_node=None, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def merge_nodes(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        # merge_paris = self.tree.get_merge_pairs()
        # for pair in merge_pairs:
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo
            # self.tree.merge_nodes(pair[0], pair[1])
            # self.tree.optimize_elbo(unique_node=None, root_node=pair[1], run=True, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)

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

            # Get parent and siblings in the same subtree
            parent = nodeA.parent()
            nodes = parent.children()
            nodes = [s for s in nodes if s != nodeA and nodeA.tssb == s.tssb]
            nodes.append(parent)
            sims = [1./(np.mean(np.abs(nodeA.node_mean - node.node_mean)) + 1e-8) for node in nodes]

            # Choose nodeB proportionally to similarities
            nodeB = np.random.choice(nodes, p=sims/np.sum(sims))

            if verbose:
                print(f"Trying to merge {nodeA.label} to {nodeB.label}...")

            self.tree.merge_nodes(nodeA, nodeB)
            local_node = None
            if local:
                local_node = nodeB
            self.tree.optimize_elbo(unique_node=None, root_node=local_node, run=True, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
            if verbose:
                print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def prune_reattach(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
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
        # Only proceed if there is at least one subtree with pruneable nodes -- i.e., it needs to have at least 2 extra nodes
        if np.any(n_nodes > 2):
            probs = np.array(n_nodes)
            probs[probs <= 2] = 0.
            probs = probs / np.sum(probs)

            # Choose a subtree with more than 1 node
            idx = np.random.choice(range(len(subtrees)), p=probs)
            subtree = subtrees[idx]
            nodes = nodes[idx]

            # Get the nodes which can be reattached, use labels
            self.tree.plot_tree();
            possible_nodes = dict()
            node_label_dict = dict()
            for node in nodes:
                possible_nodes[node.label] = [n.label for n in nodes if node.label not in n.label and n != node.parent()]
                node_label_dict[node.label] = node

            # Choose a first node
            possible_nodesA = [node for node in possible_nodes if len(possible_nodes[node]) > 0]
            nodeA_label = np.random.choice(possible_nodesA, p=[1./len(possible_nodesA)]*len(possible_nodesA))
            nodeA = node_label_dict[nodeA_label]

            # Get nodes not below node A: use labels
            sims = [1./(np.mean(np.abs(nodeA.node_mean - node_label_dict[node_label].node_mean)) + 1e-8) for node_label in possible_nodes[nodeA_label]]

            # Choose nodeB proportionally to similarities
            nodeB_label = np.random.choice(possible_nodes[nodeA_label], p=sims/np.sum(sims))
            nodeB = node_label_dict[nodeB_label]

            if verbose:
                print(f"Trying to reattach {nodeA_label} to {nodeB_label}...")

            self.tree.prune_reattach(nodeA, nodeB)
            local_node = None
            if local:
                local_node = nodeA
            self.tree.optimize_elbo(unique_node=None, root_node=local_node, run=True, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
            if verbose:
                print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def pivot_reattach(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        # Uniformly pick a subtree
        subtrees = self.tree.get_mixture()[1][1:] # without the root
        subtree = np.random.choice(subtrees, p=[1./len(subtrees)]*len(subtrees))
        init_pivot_node = subtree.root['node'].parent()
        init_pivot = init_pivot_node.label
        init_pivot_node_parent = init_pivot_node.parent()

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

            removed_pivot = False
            if len(init_pivot_node.data) == 0 and init_pivot_node_parent is not None and len(init_pivot_node.children()) == 0:
                print(f"Also removing initial pivot ({init_pivot_node.label}) from tree")
                self.tree.merge_nodes(init_pivot_node, init_pivot_node_parent)
                # self.tree.optimize_elbo(sticks_only=True, root_node=init_pivot_node_parent, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
                removed_pivot = True

            root_node = None
            if local:
                root_node = subtree.root['node']
                if removed_pivot:
                    root_node = init_pivot_node_parent
            self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
            if verbose:
                print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def add_reattach_pivot(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        # Add a node below a subtree with children subtrees
        subtrees = self.tree.get_subtrees(get_roots=True)
        nonleaf_subtrees = [subtree for subtree in subtrees if len(subtree[1]['children']) > 0]
        # Pick a subtree
        parent_subtree = nonleaf_subtrees[np.random.choice(len(nonleaf_subtrees), p=[1./len(nonleaf_subtrees)]*len(nonleaf_subtrees))]

        # Pick a node in the parent subtree
        nodes, target_probs = self.tree.get_node_data_sizes(normalized=True)
        target_probs = [prob + 1e-8 for i, prob in enumerate(target_probs) if nodes[i].tssb == parent_subtree[0]]
        nodes = [node for node in nodes if node.tssb == parent_subtree[0]]
        target_probs /= np.sum(target_probs)
        node = np.random.choice(nodes, p=np.array(target_probs))
        pivot_node = self.tree.add_node_to(node, optimal_init=True)

        # Assign data to it
        self.tree.update_ass_logits(variational=True)
        self.tree.assign_to_best()

        # Pick one of the children subtrees
        subtrees = [subtree for subtree in parent_subtree[1]['children']]
        subtree = np.random.choice(subtrees, p=[1./len(subtrees)]*len(subtrees))
        init_pivot = subtree['node'].root['node'].parent()

        # Update pivot
        self.tree.pivot_reattach_to(subtree['node'], pivot_node)

        if verbose:
            print(f"Trying to add node {pivot_node.label} and setting it as pivot of {subtree['node'].label}")

        root_node = None
        if local:
            root_node = pivot_node
        self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def push_subtree(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        # Uniformly pick a subtree
        subtrees = self.tree.get_mixture()[1][1:] # without the root
        subtree = np.random.choice(subtrees, p=[1./len(subtrees)]*len(subtrees))

        # Push subtree down
        self.tree.push_subtree(subtree.root['node'])

        if verbose:
            print(f"Trying to push {subtree.label} down")

        root_node = None
        if local:
            root_node = subtree.root['node'].parent()
        self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def extract_pivot(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        # Uniformly pick a subtree
        subtrees = self.tree.get_mixture()[1][1:] # without the root
        subtree = np.random.choice(subtrees, p=[1./len(subtrees)]*len(subtrees))

        # Push subtree down
        new_node = self.tree.extract_pivot(subtree.root['node'])

        if verbose:
            print(f"Trying to extract pivot from {subtree.label}")

        root_node = None
        if local:
            root_node = new_node
        self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def subtree_reattach(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
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

            # Choose another subtree that's similar to the subtree's top node
            rem_subtrees = [s[1] for s in subtrees if s[1]['node'] != subtreeA['node']]
            sims = [1./(np.mean(np.abs(roots[nodeA_idx]['node'].node_mean - s['node'].root['node'].node_mean)) + 1e-8) for s in rem_subtrees]
            new_subtree = np.random.choice(rem_subtrees, p=sims/np.sum(sims))

            if verbose:
                print(f"Trying to set {roots[nodeA_idx]['node'].label} below {new_subtree['node'].label}")

            # Move subtree
            pivot_changed = self.tree.subtree_reattach_to(roots[nodeA_idx]['node'], new_subtree['node'])
            # self.tree.reset_variational_parameters(variances_only=True)
            # init_baseline = jnp.mean(self.tree.data, axis=0)
            # init_log_baseline = jnp.log(init_baseline / init_baseline[0])[1:]
            # self.tree.root['node'].root['node'].log_baseline_mean = init_log_baseline + np.random.normal(0, .5, size=self.tree.data.shape[1]-1)
            root_node = None
            if local and not pivot_changed:
                root_node = roots[nodeA_idx]['node']
            self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)

            if verbose:
                print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def subtree_pivot_reattach(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        """
        Reattach subtree of node A to clone B and use it as pivot of A
        """
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        subtrees = self.tree.get_subtrees(get_roots=True)

        # Pick a non-root subtree with more than 1 node
        in_subtrees = [subtree[1] for subtree in subtrees[1:] if len(subtree[0].root['children']) > 0]

        # If there is any subtree with more than 1 node, proceed
        if len(in_subtrees) > 0:
            # Choose one subtree
            subtreeA = np.random.choice(in_subtrees, p=[1./len(in_subtrees)]*len(in_subtrees))

            # Choose one of its nodes uniformly which is not the root
            node_weights, nodes, roots = subtreeA['node'].get_mixture(get_roots=True)
            nodeA_idx = np.random.choice(len(roots[1:]), p=[1./len(roots[1:])]*len(roots[1:])) + 1
            nodeA_parent_idx = np.where(np.array(nodes) == nodes[nodeA_idx].parent())[0][0]

            if verbose:
                print(f"Trying to set {roots[nodeA_idx]['node'].label} below {subtreeA['super_parent'].label} and use it as pivot of {subtreeA['node'].label}")

            # Move subtree to parent
            self.tree.subtree_reattach_to(roots[nodeA_idx]['node'], subtreeA['super_parent'].label) # Use label to avoid bugs with references

            # Set root of moved subtree as new pivot. TODO: Choose one node from the leaves instead
            self.tree.pivot_reattach_to(subtreeA['node'], roots[nodeA_idx]['node'])

            # And choose a leaf node of that subtree as the pivot of old subtree
            # self.tree.reset_variational_parameters(variances_only=True)
            # init_baseline = jnp.mean(self.tree.data, axis=0)
            # init_log_baseline = jnp.log(init_baseline / init_baseline[0])[1:]
            # self.tree.root['node'].root['node'].log_baseline_mean = init_log_baseline + np.random.normal(0, .5, size=self.tree.data.shape[1]-1)

            root_node = None
            if local:
                root_node = roots[nodeA_idx]['node']
            self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)

            if verbose:
                print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def swap_nodes(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
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
                    ntree.optimize_elbo(root_node=None, num_samples=num_samples, n_iters=n_iters*10, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
                    if verbose:
                        print(f"{init_elbo} -> {ntree.elbo}")
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
                    ntree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
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
                self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
                if verbose:
                    print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def perturb_node(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        # Move node towards the data in its neighborhood that is currently explained by nodes in its neighborhood
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        nodes = self.tree.get_nodes()
        node = np.random.choice(nodes)

        # Decide wether to move closer to parent, sibling or child
        parent = node.parent()
        if parent is not None:
            siblings = np.array([n for n in list(parent.children()) if n != node])
            parent = np.array([parent])
        else:
            parent = np.array([])
            siblings = np.array([])
        children = np.array(list(node.children()))
        if len(children) == 0:
            children = np.array([])
        possibilities = np.concatenate([parent, siblings, children])
        probs = np.array([1+node.num_local_data() for node in possibilities]) # the more data they have, the more likely it is that we decide to move towards them
        probs = probs / np.sum(probs)

        target = np.random.choice(possibilities, p=probs)

        if verbose:
            print(f"Trying to move {node.label} close to {target.label}...")

        self.tree.perturb_node(node, target)
        root_node = node
        if not local:
            root_node = None
        self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)

        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo

    def clean_node(self, local=False, num_samples=1, n_iters=100, thin=10, tol=1e-7, step_size=0.05, mb_size=100, max_nodes=5, verbose=True, debug=False, opt=None, callback=None, **callback_kwargs):
        # Get node with bad kernel and clean it up
        init_root = deepcopy(self.tree.root)
        init_elbo = self.tree.elbo

        nodes = self.tree.get_nodes()

        n_genes = nodes[0].observed_parameters.shape[0]

        frac_events = np.array([np.sum(np.exp(node.variational_parameters['locals']['unobserved_factors_kernel_log_mean']) > 0.2)/n_genes for node in nodes])

        probs = 1e-6 + frac_events # the more complex the kernel, the more likely it is to clean it
        probs = probs / np.sum(probs)

        node = np.random.choice(nodes, p=probs)

        # Get the number of nodes with too many events
        n_bad_nodes = np.sum(frac_events > 1/3)
        frac_bad_nodes = n_bad_nodes / len(nodes)
        if frac_bad_nodes > 1/3 or n_bad_nodes > 3:
            # Reset all unobserved_factors
            print(f"Trying to clean all nodes...")
            for node in nodes:
                node.variational_parameters['locals']['unobserved_factors_mean'] *= 0.
                node.variational_parameters['locals']['unobserved_factors_log_std'] = -2*np.ones((n_genes,))
                node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = np.log(node.unobserved_factors_kernel_concentration_caller())*np.ones((n_genes,))
                node.variational_parameters['locals']['unobserved_factors_kernel_log_std'] = -2*np.ones((n_genes,))

            root_node = nodes[0]
            if not local:
                root_node = None
            self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters*5, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)
        else:
            if verbose:
                print(f"Trying to clean {node.label}...")

            node.variational_parameters['locals']['unobserved_factors_mean'] *= 0.
            node.variational_parameters['locals']['unobserved_factors_log_std'] = -2*np.ones((n_genes,))
            node.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = np.log(node.unobserved_factors_kernel_concentration_caller())*np.ones((n_genes,))
            node.variational_parameters['locals']['unobserved_factors_kernel_log_std'] = -2*np.ones((n_genes,))
            root_node = node
            if not local:
                root_node = None
            self.tree.optimize_elbo(root_node=root_node, num_samples=num_samples, n_iters=n_iters, thin=thin, tol=tol, step_size=step_size, mb_size=mb_size, max_nodes=max_nodes, init=False, debug=debug, opt=opt, opt_triplet=self.opt_triplet, callback=callback, **callback_kwargs)

        if verbose:
            print(f"{init_elbo} -> {self.tree.elbo}")

        return init_root, init_elbo
