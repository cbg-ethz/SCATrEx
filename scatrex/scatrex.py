from .models import *
from .ntssb import NTSSB
from .ntssb import StructureSearch
from .plotting import scatterplot

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import scanpy as sc
from anndata import AnnData

class SCATrEx(object):
    def __init__(self,
                 model=cna,
                 model_args=dict(),
                 verbose=False):

        self.model = cna
        self.model_args = model_args
        self.observed_tree = None
        self.ntssb = None
        self.projector = None
        self.verbose = verbose
        self.search = None
        # self.data = None
        # self.normalized_data = None
        # self.labels = None
        self.adata = None

    def add_data(self, data):
        """
        Load raw annotated data.
        """
        self.adata = AnnData(data)
        self.adata.raw = self.adata
        if self.observed_tree is not None:
            self.adata.uns['obs_node_colors'] = [self.observed_tree.tree_dict[node]['color'] for node in self.observed_tree.tree_dict]

    def set_observed_tree(self, observed_tree):
        self.observed_tree = observed_tree
        if self.adata is not None:
            self.adata.uns['obs_node_colors'] = [self.observed_tree.tree_dict[node]['color'] for node in self.observed_tree.tree_dict]

    def simulate_tree(self, observed_tree=None, observed_tree_args=dict(), model_args=None, n_extra_per_observed=1, seed=None, copy=False):
        np.random.seed(seed)

        self.observed_tree = observed_tree

        if not self.observed_tree:
            if self.verbose:
                print(f"Generating an observed {self.model.__name__.split('.')[-1].upper()} tree")
                if len(observed_tree_args.keys()) > 0:
                    for arg in observed_tree_args:
                        print(f'{arg}: {observed_tree_args[arg]}')

            self.observed_tree = self.model.ObservedTree(**observed_tree_args)
            self.observed_tree.generate_tree()
            self.observed_tree.add_node_params()

        if self.verbose:
            print(f"Generating an augmented {self.model.__name__.split('.')[-1].upper()} tree")
            if len(self.model_args.keys()) > 0:
                for arg in self.model_args:
                    print(f'{arg}: {self.model_args[arg]}')

        self.ntssb = NTSSB(self.observed_tree, self.model.Node, node_hyperparams=self.model_args)
        self.ntssb.create_new_tree(n_extra_per_observed=n_extra_per_observed)

        if self.verbose:
            print('Tree is stored in `self.observed_tree` and `self.ntssb`')

        return self.ntssb if copy else None

    def simulate_data(self, n_cells=100, seed=None, copy=False):
        np.random.seed(seed)
        self.ntssb.put_data_in_nodes(n_cells)
        self.ntssb.root['node'].root['node'].generate_data_params()

        # Sample observations
        observations = []
        assignments = []
        assignments_labels = []
        assignments_obs_labels = []
        for obs in range(len(self.ntssb.assignments)):
            sample = self.ntssb.assignments[obs]['node'].sample_observation(obs).reshape(1, -1)
            observations.append(sample)
            assignments.append(self.ntssb.assignments[obs]['node'])
            assignments_labels.append(self.ntssb.assignments[obs]['node'].label)
            assignments_obs_labels.append(self.ntssb.assignments[obs]['subtree'].label)
        assignments = np.array(assignments)
        assignments_labels = np.array(assignments_labels)
        assignments_obs_labels = np.array(assignments_obs_labels)
        observations = np.concatenate(observations)

        self.ntssb.data = observations
        self.ntssb.num_data = observations.shape[0]

        if self.verbose:
            print('Labeled data are stored in `self.adata`')

        self.adata = AnnData(observations)
        self.adata.obs['node'] = assignments_labels
        self.adata.obs['obs_node'] = assignments_obs_labels
        self.adata.uns['obs_node_colors'] = [self.observed_tree.tree_dict[node]['color'] for node in self.observed_tree.tree_dict]
        self.adata.raw = self.adata

        return (observations, assignments_labels) if copy else None

    def learn_tree(self, observed_tree=None, reset=True, search_kwargs=dict()):
        if not self.observed_tree and observed_tree is None:
            raise ValueError("No observed tree available. Please pass an observed tree object.")

        if self.adata is None:
            raise ValueError("No data available. Please add data to the SCATrEx object via `self.add_data()`")

        if observed_tree:
            self.observed_tree = observed_tree

        if reset:
            self.ntssb = NTSSB(self.observed_tree, self.model.Node, node_hyperparams=self.model_args)
            self.ntssb.add_data(self.adata.raw.X, to_root=True)
            self.ntssb.root['node'].root['node'].reset_data_parameters()
            self.ntssb.reset_variational_parameters()
            self.ntssb.update_ass_logits(variational=True)
            self.ntssb.assign_to_best()
            self.search = StructureSearch(self.ntssb)
        else:
            if self.verbose:
                print('Will continue search from where it left off.')

        self.ntssb = self.search.run_search(**search_kwargs)

        self.adata.obs['node'] = np.array([assignment['node'].label for assignment in self.ntssb.assignments])
        self.adata.obs['obs_node'] = np.array([assignment['subtree'].label for assignment in self.ntssb.assignments])

    def normalize_data(self, target_sum=1e4, log=True, copy=False):
        sc.pp.normalize_total(self.adata, target_sum=target_sum)
        if log:
            sc.pp.log1p(self.adata)

        if self.verbose:
            print('Normalized data are stored in `self.adata`')

        return self.adata if copy else None

    def project_data(self, n_dim=2, copy=False):
        self.pca_obj = PCA(n_components=n_dim)
        self.pca = self.pca_obj.fit_transform(self.adata.X)

        if self.verbose:
            print(f'{n_dim}-projected data are stored in `self.pca` and projection matrix in `self.pca_obj`')

        return (self.pca, self.pca_obj) if copy else None

    def assign_new_data(self, data):
        raise NotImplementedError
        # Learn posterior of local parameters of new data

        # Evaluate likelihood of data at each node

        # Assign to best
        return

    def plot_tree(self, **kwargs):
        """
        The nodes will be coloured according to the average normalized counts of the feature indicated in `color`
        """
        kwargs.setdefault('var_names', self.adata.var_names)
        self.n_tssb.plot_tree(**kwargs)

    def plot_tree_proj(self, figsize=(4,4), title='', lw=0.001, hw=0.003, s=10, fc='k', ec='k', fs=22, lfs=16, save=None):
        scatterplot.plot_tree_proj(self.pca, self.ntssb, pca_obj=self.pca_obj, title=title,
                                    line_width=lw, head_width=hw, s=s, fc=fc, ec=ec, fontsize=fs,
                                    legend_fontsize=lfs, figsize=figsize, save=save)

    def plot_unobserved_parameters(self, figsize=(4,4), lw=4, alpha=0.7, title='', fontsize=18, step=4, estimated=False, save=None):
        nodes, _ = self.ntssb.get_node_mixture()
        plt.figure(figsize=figsize)
        ticklabs = []
        tickpos = []
        for i, node in enumerate(nodes):
            ls = '-'
            tickpos.append(- step*i)
            ticklabs.append(fr"{node.label.replace('-', '')}")
            unobs_factors = node.unobserved_factors
            if estimated:
                unobs_factors = node.variational_parameters['locals']['unobserved_factors_mean']
            plt.plot(unobs_factors - step*i, label=node.label, color=node.tssb.color, lw=4, alpha=0.7, ls=ls)
        plt.yticks(tickpos, labels=ticklabs, fontsize=fontsize)
        plt.xticks([])
        plt.title(title, fontsize=fontsize)
        if save is not None:
            plt.savefig(save, bbox_inches='tight')
        plt.show()

    def compute_smoothed_expression(self, var_names=None, copy=False):
        smoothed_mat = []
        if isinstance(var_names, dict):
            self.var_names = var_names
            for region in var_names:
                smoothed_mat.append(self.__smooth_expression(self.adata, var_names[region]))

        self.adata.obsm["smoothed"] = np.concatenate(smoothed_mat, axis=1)

        if self.verbose:
            print(f'Smoothed gene expression is stored in `self.adata.obsm[\"smoothed\"]`')

        return self.adata.obsm["smoothed"] if copy else None

    def __smooth_expression(self, var_names=None):
        pass

    def compute_pathway_enrichments(self, db='kegg', method='gsva'):
        pass
