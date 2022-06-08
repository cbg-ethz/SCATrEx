from .models import *
from .ntssb import NTSSB
from .ntssb import StructureSearch
from .plotting import scatterplot, constants

import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import pickle
from copy import deepcopy
import os

import scanpy as sc
from anndata import AnnData
import gseapy as gp
from tqdm.auto import tqdm

import logging

logger = logging.getLogger(__name__)


class SCATrEx(object):
    def __init__(
        self,
        model=cna,
        model_args=dict(),
        verbosity=logging.INFO,
        temppath="./temppath",
    ):

        self.model = cna
        self.model_args = model_args
        self.observed_tree = None
        self.ntssb = None
        self.projector = None
        self.verbosity = verbosity
        self.search = None
        self.adata = None
        self.temppath = temppath
        if not os.path.exists(temppath):
            os.makedirs(temppath, exist_ok=True)

        logger.setLevel(verbosity)

    def add_data(self, data):
        """
        Load raw annotated data.
        """
        if isinstance(data, AnnData):
            self.adata = data.copy()
        else:
            self.adata = AnnData(data)
        self.adata.raw = self.adata
        if self.observed_tree is not None:
            if self.observed_tree.adata is None:
                self.observed_tree.create_adata()
            self.adata.uns["obs_node_colors"] = [
                self.observed_tree.tree_dict[node]["color"]
                for node in np.unique(self.observed_tree.adata.obs["obs_node"])
            ]
            # self.adata.uns['obs_node_colors'] = [self.observed_tree.tree_dict[node]['color'] for node in self.observed_tree.tree_dict]

    def set_observed_tree(self, observed_tree):
        if isinstance(observed_tree, str):
            with open(observed_tree, "rb") as f:
                self.observed_tree = pickle.load(f)
        else:
            self.observed_tree = observed_tree

        if self.observed_tree.adata is None:
            self.observed_tree.create_adata()

        if self.adata is not None:
            self.adata.uns["obs_node_colors"] = [
                self.observed_tree.tree_dict[node]["color"]
                for node in self.observed_tree.tree_dict
            ]
            try:
                labels = list(self.observed_tree.tree_dict.keys())
                sizes = [
                    self.observed_tree.tree_dict[node]["size"]
                    for node in self.observed_tree.tree_dict
                ]
                self.adata.uns["observed_frequencies"] = dict(zip(labels, sizes))
            except KeyError:
                pass

    def simulate_tree(
        self,
        observed_tree=None,
        observed_tree_args=dict(),
        observed_tree_params=dict(),
        model_args=None,
        n_genes=50,
        n_extra_per_observed=1,
        seed=None,
        copy=False,
    ):
        np.random.seed(seed)

        self.observed_tree = observed_tree

        if not self.observed_tree:
            logger.info(
                f"Generating an observed {self.model.__name__.split('.')[-1].upper()} tree"
            )
            if len(observed_tree_args.keys()) > 0:
                for arg in observed_tree_args:
                    logger.info(f"{arg}: {observed_tree_args[arg]}")

            self.observed_tree = self.model.ObservedTree(**observed_tree_args)
            self.observed_tree.generate_tree()
            self.observed_tree.add_node_params(n_genes=n_genes, **observed_tree_params)

        logger.info(
            f"Generating an augmented {self.model.__name__.split('.')[-1].upper()} tree"
        )
        if len(self.model_args.keys()) > 0:
            for arg in self.model_args:
                logger.info(f"{arg}: {self.model_args[arg]}")

        self.ntssb = NTSSB(
            self.observed_tree, self.model.Node, node_hyperparams=self.model_args
        )
        self.ntssb.create_new_tree(n_extra_per_observed=n_extra_per_observed)

        logger.info("Tree is stored in `self.observed_tree` and `self.ntssb`")

        return self.ntssb if copy else None

    def simulate_data(self, n_cells=100, seed=None, copy=False):
        np.random.seed(seed)
        self.ntssb.put_data_in_nodes(n_cells)
        self.ntssb.root["node"].root["node"].generate_data_params()

        # Sample observations
        observations = []
        assignments = []
        assignments_labels = []
        assignments_obs_labels = []
        for obs in range(len(self.ntssb.assignments)):
            sample = self.ntssb.assignments[obs].sample_observation(obs).reshape(1, -1)
            observations.append(sample)
            assignments.append(self.ntssb.assignments[obs])
            assignments_labels.append(self.ntssb.assignments[obs].label)
            assignments_obs_labels.append(self.ntssb.assignments[obs].tssb.label)
        assignments = np.array(assignments)
        assignments_labels = np.array(assignments_labels)
        assignments_obs_labels = np.array(assignments_obs_labels)
        observations = np.concatenate(observations)

        self.ntssb.data = observations
        self.ntssb.num_data = observations.shape[0]

        logger.info("Labeled data are stored in `self.adata`")

        self.adata = AnnData(observations)
        self.adata.obs["node"] = assignments_labels
        self.adata.obs["obs_node"] = assignments_obs_labels
        self.adata.uns["obs_node_colors"] = [
            self.observed_tree.tree_dict[node]["color"]
            for node in self.observed_tree.tree_dict
        ]
        self.adata.raw = self.adata

        return (observations, assignments_labels) if copy else None

    def learn_tree(
        self,
        observed_tree=None,
        reset=True,
        cell_filter=None,
        filter_genes=False,
        max_genes=1000,
        search_kwargs=dict(),
    ):
        if not self.observed_tree and observed_tree is None:
            raise ValueError(
                "No observed tree available. Please pass an observed tree object."
            )

        if self.adata is None:
            raise ValueError(
                "No data available. Please add data to the SCATrEx object via `self.add_data()`"
            )

        if observed_tree:
            self.observed_tree = observed_tree

        clones = np.array(
            [
                self.observed_tree.tree_dict[clone]["params"]
                for clone in self.observed_tree.tree_dict
            ]
        )

        cell_idx = np.arange(self.adata.shape[0])
        others_idx = np.array([])
        if cell_filter:
            cell_idx = np.where(
                np.array(
                    [
                        cell_filter in celltype
                        for celltype in self.adata.obs["celltype_major"]
                    ]
                )
            )[0]
            others_idx = np.where(
                np.array(
                    [
                        cell_filter not in celltype
                        for celltype in self.adata.obs["celltype_major"]
                    ]
                )
            )[0]

        adata = self.adata.raw.to_adata()
        adata = adata[cell_idx]
        clones_filtered = np.array(clones)
        rna_filtered = np.array(adata.X)
        observed_tree_filtered = deepcopy(self.observed_tree)

        retained_genes = adata.var_names
        retained_genes_pos = np.arange(retained_genes.size)

        if filter_genes:
            adata.raw = adata.copy()
            rna_filtered = np.array(adata.X)

            # Subset the data to the highest variable genes
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(
                adata, n_top_genes=np.min([max_genes, adata.shape[1]])
            )

            hvgenes = np.where(np.array(adata.var.highly_variable).ravel())[0]
            retained_genes = adata.var.index[hvgenes]
            retained_genes_pos = np.array(hvgenes)
            clones_filtered = clones_filtered[:, hvgenes]
            rna_filtered = np.array(rna_filtered[:, hvgenes])

            for node in observed_tree_filtered.tree_dict:
                observed_tree_filtered.tree_dict[node][
                    "params"
                ] = observed_tree_filtered.tree_dict[node]["params"][hvgenes]

        self.adata.uns["scatrex_retained_genes"] = retained_genes

        logger.debug(f"Filtered scRNA data for clonemap shape: {rna_filtered.shape}")

        if reset:
            self.ntssb = NTSSB(
                observed_tree_filtered,
                self.model.Node,
                node_hyperparams=self.model_args,
                verbosity=self.verbosity,
            )
            self.ntssb.add_data(np.array(rna_filtered), to_root=True)
            self.ntssb.root["node"].root["node"].reset_data_parameters()
            self.ntssb.reset_variational_parameters()
            self.ntssb.update_ass_logits(variational=True)
            self.ntssb.assign_to_best()
            self.search = StructureSearch(self.ntssb)
        else:
            logger.info("Will continue search from where it left off.")

        self.ntssb = self.search.run_search(**search_kwargs)
        self.ntssb.create_augmented_tree_dict()

        node_assignments = [
            self.ntssb.root["node"].root["node"].label
        ] * self.adata.shape[0]
        for i, idx in enumerate(cell_idx):
            node_assignments[idx] = self.ntssb.assignments[i].label

        obs_node_assignments = np.array(
            [self.ntssb.root["node"].label] * self.adata.shape[0]
        )
        obs_node_assignments[cell_idx] = np.array(
            [assignment.tssb.label for assignment in self.ntssb.assignments]
        )

        self.adata.obs["scatrex_node"] = node_assignments
        self.adata.obs["scatrex_obs_node"] = obs_node_assignments.astype(str)

        labels = list(self.observed_tree.tree_dict.keys())
        sizes = [
            np.count_nonzero(self.adata.obs["scatrex_obs_node"] == label)
            for label in labels
        ]
        self.adata.uns["scatrex_estimated_frequencies"] = dict(zip(labels, sizes))

        cnv_mat = np.ones(self.adata.shape) * 2
        for clone_id in np.unique(self.adata.obs["scatrex_obs_node"][cell_idx]):
            cells = np.where(self.adata.obs["scatrex_obs_node"][cell_idx] == clone_id)[
                0
            ]
            clone_idx = np.where(np.array(labels).astype(str) == str(clone_id))[0]
            cnv_mat[cells] = np.array(clones)[clone_idx]
        self.adata.layers["scatrex_cnvs"] = cnv_mat

        xi_mat = np.zeros(self.adata.shape)
        om_mat = np.zeros(self.adata.shape)
        mean_mat = np.zeros(self.adata.shape)
        nodes = np.array(self.ntssb.get_nodes())
        nodes_labels = np.array([node.label for node in nodes])
        for node_id in np.unique(self.adata.obs["scatrex_node"][cell_idx]):
            cells = np.where(self.adata.obs["scatrex_node"][cell_idx] == node_id)[0]
            node = nodes[np.where(node_id == nodes_labels)[0][0]]
            pos = np.meshgrid(cells, retained_genes_pos)
            xi_mat[tuple(pos)] = (
                np.array(
                    node.variational_parameters["locals"]["unobserved_factors_mean"]
                ).reshape(-1, 1)
                * np.ones((len(cells), len(retained_genes_pos))).T
            )
            om_mat[tuple(pos)] = (
                np.array(
                    np.exp(
                        node.variational_parameters["locals"][
                            "unobserved_factors_kernel_log_mean"
                        ]
                    )
                ).reshape(-1, 1)
                * np.ones((len(cells), len(retained_genes_pos))).T
            )
            mean_mat[tuple(pos)] = (
                np.array(node.get_mean(norm=False)).reshape(-1, 1)
                * np.ones((len(cells), len(retained_genes_pos))).T
            )
        self.adata.layers["scatrex_xi"] = xi_mat
        self.adata.layers["scatrex_om"] = om_mat
        self.adata.layers["scatrex_mean"] = mean_mat

        self.ntssb.initialize_gene_node_colormaps()

    def learn_clonemap_corr(
        self,
        observed_tree=None,
        cell_filter=None,
        layer="scaled",
        dna_diploid_threshold=0.95,
    ):
        if not self.observed_tree and observed_tree is None:
            raise ValueError(
                "No observed tree available. Please pass an observed tree object."
            )

        if self.adata is None:
            raise ValueError(
                "No data available. Please add data to the SCATrEx object via `self.add_data()`"
            )

        if observed_tree:
            self.observed_tree = observed_tree

        cell_idx = np.arange(self.adata.shape[0])
        others_idx = np.array([])
        if cell_filter:
            cell_idx = np.where(
                np.array(
                    [
                        cell_filter in celltype
                        for celltype in self.adata.obs["celltype_major"]
                    ]
                )
            )[0]
            others_idx = np.where(
                np.array(
                    [
                        cell_filter not in celltype
                        for celltype in self.adata.obs["celltype_major"]
                    ]
                )
            )[0]

        labels = [
            self.observed_tree.tree_dict[clone]["label"]
            for clone in self.observed_tree.tree_dict
        ]
        ids = [clone for clone in self.observed_tree.tree_dict]
        clones = [
            self.observed_tree.tree_dict[clone]["params"]
            for clone in self.observed_tree.tree_dict
        ]
        clones = np.array(clones)

        # Diploid clone
        dna_is_diploid = np.array(
            (np.sum(clones == 2, axis=1) / clones.shape[1]) > dna_diploid_threshold
        )
        diploid_clone_indices = np.where(dna_is_diploid)[0]
        malignant_indices = np.arange(clones.shape[0])
        if len(diploid_clone_indices) == 0:
            logger.warning("No diploid clones in scDNA-seq data.")
        else:
            diploid_clone_idx = diploid_clone_indices[0]
            malignant_indices_mask = np.ones(clones.shape[0], dtype=bool)
            malignant_indices_mask[diploid_clone_idx] = 0
            malignant_indices = malignant_indices[malignant_indices_mask]
            diploid_labels = np.array(labels)[diploid_clone_idx].tolist()
            diploid_ids = np.array(ids)[diploid_clone_idx].tolist()
            malignant_labels = np.array(labels)[malignant_indices].tolist()
            malignant_ids = np.array(ids)[malignant_indices].tolist()
            logger.info(f"Diploid clones: labels: {diploid_labels}, ids: {diploid_ids}")
            logger.info(
                f"Malignant clones: labels: {malignant_labels}, ids: {malignant_ids}"
            )

        # Subset the data to the highest variable genes
        adata = deepcopy(self.adata)
        adata = adata[cell_idx]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=np.min([500, adata.shape[1]]))
        hvgenes = np.where(np.array(adata.var.highly_variable).ravel())[0]
        clones_filtered = clones[:, hvgenes]
        rna_filtered = np.array(adata.raw.X[:, hvgenes])

        # Subset the data to the genes with varying copy number across malignant clones
        var_genes = np.where(np.var(clones_filtered[malignant_indices], axis=0) > 0)[0]
        clones_filtered = clones_filtered[:, var_genes]
        rna_filtered = rna_filtered[:, var_genes]

        assignments = [0] * self.adata.shape[0]
        for i, cell in enumerate(cell_idx):
            corrs = []
            for clone_idx in range(len(clones)):
                if clone_idx != diploid_clone_idx:
                    correlation, _ = stats.spearmanr(
                        rna_filtered[i], clones_filtered[clone_idx]
                    )
                else:
                    correlation = -1.0
                corrs.append(correlation)
            assignments[cell] = ids[np.argmax(corrs)]
        if len(others_idx) > 0:
            if len(diploid_clone_indices) > 0:
                assignments[others_idx] = ids[diploid_clone_idx]

        assignments = np.array(assignments)
        self.adata.obs["node"] = assignments.astype(str)
        self.adata.obs["obs_node"] = assignments.astype(str)

        labels = list(self.observed_tree.tree_dict.keys())
        sizes = [
            np.count_nonzero(self.adata.obs["obs_node"] == label) for label in labels
        ]
        self.adata.uns["corr_estimated_frequencies"] = dict(zip(labels, sizes))

        cnv_mat = np.ones(self.adata.shape) * 2
        for clone_id in np.unique(assignments):
            cells = np.where(assignments == clone_id)[0]
            clone_idx = np.where(np.array(ids).astype(str) == str(clone_id))[0]
            cnv_mat[cells] = np.array(clones[clone_idx])
        self.adata.layers["corr_cnvs"] = np.array(cnv_mat)

    def learn_clonemap(
        self,
        observed_tree=None,
        cell_filter=None,
        dna_diploid_threshold=0.95,
        filter_genes=True,
        filter_diploid_cells=False,
        max_genes=500,
        **optimize_kwargs,
    ):
        """Provides an equivalent of clonealign (for CNV nodes) or cardelino (for SNV nodes)"""
        if not self.observed_tree and observed_tree is None:
            raise ValueError(
                "No observed tree available. Please pass an observed tree object."
            )

        if self.adata is None:
            raise ValueError(
                "No data available. Please add data to the SCATrEx object via `self.add_data()`"
            )

        if observed_tree:
            self.observed_tree = observed_tree

        # import numpy as np
        # import scatrex
        # from copy import deepcopy
        # import scanpy as sc
        # self = sca
        # cell_filter='Melanoma'
        # dna_diploid_threshold=0.95
        # filter_genes=True
        # filter_diploid_cells=True

        cell_idx = np.arange(self.adata.shape[0])
        others_idx = np.array([])
        if cell_filter:
            logger.info(f"Selecting {cell_filter} cells")
            cell_idx = np.where(
                np.array(
                    [
                        cell_filter in celltype
                        for celltype in self.adata.obs["celltype_major"]
                    ]
                )
            )[0]
            others_idx = np.where(
                np.array(
                    [
                        cell_filter not in celltype
                        for celltype in self.adata.obs["celltype_major"]
                    ]
                )
            )[0]

        labels = [
            self.observed_tree.tree_dict[clone]["label"]
            for clone in self.observed_tree.tree_dict
        ]
        ids = [clone for clone in self.observed_tree.tree_dict]
        clones = [
            self.observed_tree.tree_dict[clone]["params"]
            for clone in self.observed_tree.tree_dict
        ]
        clones = np.array(clones)

        # Diploid clone
        dna_is_diploid = np.array(
            (np.sum(clones == 2, axis=1) / clones.shape[1]) > dna_diploid_threshold
        )
        diploid_clone_indices = np.where(dna_is_diploid)[0]
        malignant_indices = np.arange(clones.shape[0])
        if len(diploid_clone_indices) == 0:
            logger.warning("No diploid clones in scDNA-seq data.")
        else:
            diploid_clone_idx = diploid_clone_indices[0]
            malignant_indices_mask = np.ones(clones.shape[0], dtype=bool)
            malignant_indices_mask[diploid_clone_idx] = 0
            malignant_indices = malignant_indices[malignant_indices_mask]
            diploid_labels = np.array(labels)[diploid_clone_idx].tolist()
            diploid_ids = np.array(ids)[diploid_clone_idx].tolist()
            malignant_labels = np.array(labels)[malignant_indices].tolist()
            malignant_ids = np.array(ids)[malignant_indices].tolist()
            logger.info(f"Diploid clones: labels: {diploid_labels}, ids: {diploid_ids}")
            logger.info(
                f"Malignant clones: labels: {malignant_labels}, ids: {malignant_ids}"
            )

        adata = self.adata.raw.to_adata()
        adata = adata[cell_idx]
        clones_filtered = np.array(clones)
        rna_filtered = np.array(adata.X)
        observed_tree_filtered = deepcopy(self.observed_tree)

        if filter_genes:
            # Subset the data to the genes with varying copy number across malignant clones
            # var_genes = np.where(np.var(clones_filtered[malignant_indices], axis=0) > 0)[0]
            var_genes = np.where(np.any(clones[malignant_indices] != 2, axis=0))[0]

            clones_filtered = clones[:, var_genes]
            adata = adata[:, var_genes]
            adata.raw = adata.copy()
            rna_filtered = np.array(adata.X)
            for node in observed_tree_filtered.tree_dict:
                observed_tree_filtered.tree_dict[node][
                    "params"
                ] = observed_tree_filtered.tree_dict[node]["params"][var_genes]

            # Subset the data to the highest variable genes
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(
                adata, n_top_genes=np.min([max_genes, adata.shape[1]])
            )

            hvgenes = np.where(np.array(adata.var.highly_variable).ravel())[0]
            clones_filtered = clones_filtered[:, hvgenes]
            rna_filtered = np.array(rna_filtered[:, hvgenes])

            for node in observed_tree_filtered.tree_dict:
                observed_tree_filtered.tree_dict[node][
                    "params"
                ] = observed_tree_filtered.tree_dict[node]["params"][hvgenes]

        logger.debug(f"Filtered scRNA data for clonemap shape: {rna_filtered.shape}")

        if filter_diploid_cells:
            if len(diploid_clone_indices) > 0:
                # Set weights -- no diploid cells allowed
                for node in diploid_ids:
                    observed_tree_filtered.tree_dict[node]["size"] = 0
                observed_tree_filtered.update_weights(uniform=False)
                logger.info(
                    f"Assigning no weight to diploid clones: {diploid_labels} ({diploid_ids})"
                )

        self.ntssb = NTSSB(
            observed_tree_filtered, self.model.Node, node_hyperparams=self.model_args
        )
        self.ntssb.add_data(np.array(rna_filtered), to_root=True)
        self.ntssb.root["node"].root["node"].reset_data_parameters()
        self.ntssb.reset_variational_parameters()
        init_baseline = np.mean(
            self.ntssb.data
            / np.sum(self.ntssb.data, axis=1).reshape(-1, 1)
            * self.ntssb.data.shape[1],
            axis=0,
        )
        init_baseline = init_baseline / init_baseline[0]
        init_log_baseline = np.log(init_baseline[1:] + 1e-6)
        self.ntssb.root["node"].root["node"].variational_parameters["globals"][
            "log_baseline_mean"
        ] = np.clip(init_log_baseline, -1, 1)
        optimize_kwargs.setdefault(
            "sticks_only", True
        )  # ignore other node-specific parameters
        optimize_kwargs.setdefault("mb_size", adata.shape[0])  # use all cells in batch
        elbos = self.ntssb.optimize_elbo(max_nodes=1, **optimize_kwargs)

        self.ntssb.plot_tree(counts=True)

        assignments = np.array([ids[0]] * self.adata.shape[0])
        # assignments[cell_idx] = np.array([self.observed_tree.tree_dict[assignment.tssb.label]['label'] for assignment in self.ntssb.assignments])
        assignments[cell_idx] = np.array(
            [assignment.tssb.label for assignment in self.ntssb.assignments]
        )
        if len(others_idx) > 0:
            if len(diploid_clone_indices) > 0:
                assignments[others_idx] = ids[diploid_clone_idx]

        self.adata.obs["node"] = assignments.astype(str)
        self.adata.obs["obs_node"] = assignments.astype(str)

        # scdna_labels = [self.observed_tree.tree_dict[node]['label'] for node in self.observed_tree.tree_dict if self.observed_tree.tree_dict[node]['size'] > 0]
        scdna_labels = list(self.observed_tree.tree_dict.keys())
        sizes = [
            np.count_nonzero(self.adata.obs["obs_node"] == label)
            for label in scdna_labels
        ]
        self.adata.uns["clonemap_estimated_frequencies"] = dict(
            zip(scdna_labels, sizes)
        )

        cnv_mat = np.ones(self.adata.shape) * 2
        for clone_id in np.unique(assignments):
            cells = np.where(assignments == clone_id)[0]
            clone_idx = np.where(np.array(ids).astype(str) == str(clone_id))[0]
            cnv_mat[cells] = np.array(clones[clone_idx])
        self.adata.layers["clonemap_cnvs"] = np.array(cnv_mat)

        # Initialize colormaps and account for filtered genes
        # node_obs = dict(zip([self.observed_tree.tree_dict[node]['label'] for node in self.observed_tree.tree_dict], [self.observed_tree.tree_dict[node]['params'] for node in self.observed_tree.tree_dict]))
        node_obs = dict(
            zip(
                scdna_labels,
                [
                    self.observed_tree.tree_dict[node]["params"]
                    for node in self.observed_tree.tree_dict
                ],
            )
        )
        nodes = self.ntssb.get_nodes()
        avgs = []
        for node in nodes:
            idx = np.array(list(node.data))
            if len(idx) > 0:
                avgs.append(np.mean(self.adata.X[idx], axis=0))
            else:
                avgs.append(np.nan * np.zeros(self.adata.X.shape[1]))
        node_avg_exp = dict(zip([node.label for node in nodes], avgs))
        self.ntssb.initialize_gene_node_colormaps(
            node_obs=node_obs, node_avg_exp=node_avg_exp
        )

        return elbos

    def normalize_data(self, target_sum=1e4, log=True, copy=False):
        sc.pp.normalize_total(self.adata, target_sum=target_sum)
        if log:
            sc.pp.log1p(self.adata)

        mat = sc.pp.scale(self.adata.X, copy=True)
        self.adata.layers["scaled"] = mat

        logger.info("Normalized data are stored in `self.adata`")

        return self.adata if copy else None

    def project_data(self, n_dim=2, copy=False):
        self.pca_obj = PCA(n_components=n_dim)
        self.pca = self.pca_obj.fit_transform(self.adata.X)

        logger.info(
            f"{n_dim}-projected data are stored in `self.pca` and projection matrix in `self.pca_obj`"
        )

        return (self.pca, self.pca_obj) if copy else None

    def assign_new_data(self, data):
        raise NotImplementedError
        # Learn posterior of local parameters of new data

        # Evaluate likelihood of data at each node

        # Assign to best

        return

    def set_node_event_strings(self, **kwargs):
        self.ntssb.set_node_event_strings(var_names=sim_sca.adata.var_names, **kwargs)

    def plot_tree(
        self,
        pathway=None,
        ax=None,
        figsize=(6, 6),
        dpi=80,
        tree_dpi=300,
        cbtitle="",
        title="",
        show_colorbar=True,
        **kwargs,
    ):
        """
        The nodes will be coloured according to the average normalized counts of the feature indicated in `color`
        """
        # Add events
        events = False
        if "events" in kwargs:
            events = kwargs["events"]

        # Deal with colorbars
        gene = None
        if "gene" in kwargs:
            gene = kwargs["gene"]

        if self.adata is not None:
            if gene is not None:
                gene_pos = np.where(self.adata.var_names == gene)[0][0]
                kwargs["gene"] = gene_pos

        # Make pathway colormap
        if pathway is not None:
            pathway_node_cmap = dict()
            for node in self.enrichments:
                try:
                    val = -np.log10(self.enrichments[node][pathway]["Adjusted P-value"])
                except:
                    val = 0.0
                    pass
                color = matplotlib.colors.to_hex(constants.PATHWAY_CMAPPER.to_rgba(val))
                pathway_node_cmap[node] = color
            kwargs["node_color_dict"] = pathway_node_cmap

        g = self.ntssb.plot_tree(**kwargs)

        if gene is not None or pathway is not None:
            # Plot colorbar
            g.attr(dpi=str(tree_dpi))
            g.render("temptree", directory=self.temppath, format="png")
            im = plt.imread(os.path.join(self.temppath, "temptree.png"))
            if ax is not None:
                plt.sca(ax)
            else:
                plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(im, interpolation="bilinear")
            plt.axis("off")
            plt.title(title)
            g = plt.gca()
            if show_colorbar:
                if gene is not None:
                    mapper = self.ntssb.gene_node_colormaps[kwargs["genemode"]][
                        "mapper"
                    ]
                    if isinstance(mapper, list):
                        mapper = self.ntssb.gene_node_colormaps[kwargs["genemode"]][
                            "mapper"
                        ][gene_pos]
                    cbar = plt.colorbar(mapper, label=cbtitle)
                    if kwargs["genemode"] == "observed":
                        n_discrete_levels = self.ntssb.gene_node_colormaps["observed"][
                            "mapper"
                        ].cmap.N
                        tick_locs = (
                            (np.arange(n_discrete_levels) + 0.5)
                            * (n_discrete_levels - 1)
                            / n_discrete_levels
                        )
                        cbar.set_ticks(tick_locs)
                        cbar.set_ticklabels(np.arange(n_discrete_levels))
                elif pathway is not None:
                    plt.colorbar(constants.PATHWAY_CMAPPER, label=cbtitle)
            os.remove(os.path.join(self.temppath, "temptree.png"))
            os.remove(os.path.join(self.temppath, "temptree"))

        return g

    def plot_tree_proj(
        self,
        project=True,
        figsize=(4, 4),
        title="",
        lw=0.001,
        hw=0.003,
        s=10,
        fc="k",
        ec="k",
        fs=22,
        lfs=16,
        save=None,
    ):
        if project:
            pca_obj = self.pca_obj
        else:
            pca_obj = None
        scatterplot.plot_tree_proj(
            self.pca,
            self.ntssb,
            pca_obj=pca_obj,
            title=title,
            line_width=lw,
            head_width=hw,
            s=s,
            fc=fc,
            ec=ec,
            fontsize=fs,
            legend_fontsize=lfs,
            figsize=figsize,
            save=save,
        )

    def plot_observed_parameters(
        self, figsize=(4, 4), lw=4, alpha=0.7, title="", fontsize=18, step=4, save=None
    ):
        nodes, _ = self.ntssb.get_node_mixture()
        plt.figure(figsize=figsize)
        ticklabs = []
        tickpos = []
        for i, node in enumerate(nodes):
            ls = "-"
            tickpos.append(-step * i)
            ticklabs.append(rf"{node.label.replace('-', '')}")
            obs = node.observed_parameters
            plt.plot(
                obs - step * i,
                label=node.label,
                color=node.tssb.color,
                lw=4,
                alpha=0.7,
                ls=ls,
            )
            plt.xticks([])
        plt.yticks(tickpos, labels=ticklabs, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        if save is not None:
            plt.savefig(save, bbox_inches="tight")
        plt.show()

    def plot_unobserved_parameters(
        self,
        node_names=None,
        gene=None,
        gene_names=None,
        ax=None,
        figsize=(4, 4),
        lw=4,
        alpha=0.7,
        title="",
        fontsize=18,
        step=4,
        estimated=False,
        x_max=1,
        name="unobserved_factors",
        show_names=False,
        save=None,
    ):
        nodes, _ = self.ntssb.get_node_mixture()

        if node_names is not None:
            nodes = [node for node in nodes if node.label in node_names]

        genes = None
        if gene_names is not None:
            # Transform gene names into gene indices
            genes = np.array([self.adata.var_names.get_loc(g) for g in gene_names])

        if self.search is not None:
            if len(self.search.traces["elbo"]) > 0:
                estimated = True

        if ax is None:
            plt.figure(figsize=figsize)
        else:
            plt.gca(ax)
        ticklabs = []
        tickpos = []
        for i, node in enumerate(nodes):
            if node.parent() is not None:
                ls = "-"
                unobs = node.__getattribute__(name)
                if estimated:
                    try:
                        mean = node.variational_parameters["locals"][name]
                    except KeyError:
                        try:
                            unobs = node.variational_parameters["locals"][
                                name + "_mean"
                            ]
                            std = np.exp(
                                node.variational_parameters["locals"][name + "_log_std"]
                            )
                        except KeyError:
                            unobs = np.exp(
                                node.variational_parameters["locals"][
                                    name + "_log_mean"
                                ]
                            )
                            std = np.exp(
                                node.variational_parameters["locals"][name + "_log_std"]
                            )
                if estimated and gene is not None:
                    gene_pos = np.where(self.adata.var_names == gene)[0][0]
                    mean = unobs
                    # Plot the variational distribution
                    xx = np.arange(-x_max, x_max, 0.001)
                    # plt.scatter(xx, stats.norm.pdf(xx, mean[14], std[14]), c=np.abs(xx))
                    density = stats.norm.pdf(xx, mean[gene_pos], std[gene_pos])
                    normed_density = density / np.max(density)
                    # step = 1.5
                    plt.plot(
                        xx,
                        density - step * i,
                        label=node.label,
                        color=node.tssb.color,
                        lw=4,
                        alpha=0.7,
                        ls=ls,
                    )
                else:
                    plt.plot(
                        unobs[genes].ravel() - step * i,
                        label=node.label,
                        color=node.tssb.color,
                        lw=4,
                        alpha=0.7,
                        ls=ls,
                    )
                    if gene_names is not None and show_names:
                        plt.xticks(np.arange(len(gene_names)), labels=gene_names)
                    else:
                        plt.xticks([])
                tickpos.append(-step * i)
                ticklabs.append(rf"{node.label.replace('-', '')}")
        plt.yticks(tickpos, labels=ticklabs, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        if save is not None:
            plt.savefig(save, bbox_inches="tight")
        if ax is None:
            plt.show()

    def bulkify(self):
        self.adata.var["raw_bulk"] = np.mean(self.adata.X, axis=0)
        try:
            self.adata.var["scaled_bulk"] = np.mean(self.adata.layers["scaled"], axis=0)
        except KeyError:
            pass
        try:
            self.adata.var["smoothed_bulk"] = np.mean(
                self.adata.layers["smoothed"], axis=0
            )
        except KeyError:
            pass

    def compute_smoothed_expression(
        self, var_names=None, window_size=10, clip=3, copy=False
    ):
        mat = sc.pp.scale(self.adata.X, copy=True)
        self.adata.layers["scaled"] = mat

        if isinstance(var_names, dict):
            smoothed_mat = []
            for region in var_names:
                smoothed_mat.append(
                    self.__smooth_expression(
                        mat=self.adata[:, var_names[region]].layers["scaled"],
                        var_names=None,
                        window_size=window_size,
                        clip=clip,
                    )
                )
            self.adata.layers["smoothed"] = np.concatenate(smoothed_mat, axis=1)
        else:
            self.adata.layers["smoothed"] = self.__smooth_expression(
                mat=mat, window_size=window_size, clip=clip
            )

        logger.info(
            f'Smoothed gene expression is stored in `self.adata.layers["smoothed"]`'
        )

        return self.adata.obsm["smoothed"] if copy else None

    def __smooth_expression(self, mat=None, var_names=None, window_size=10, clip=3):
        if mat is None:
            if var_names is None:
                var_names = np.arange(self.adata.shape[1])
            mat = self.adata[:, var_names].X

        mat = np.clip(mat, -np.abs(clip), np.abs(clip))

        half_window = int(window_size / 2)
        smoothed = np.zeros(mat.shape)
        for ii in range(mat.shape[1]):
            left = max(0, ii - half_window)
            right = min(ii + half_window, mat.shape[1] - 1)
            if left != right:
                smoothed[:, ii] = np.mean(mat[:, left:right], axis=1)

        return smoothed

    def get_rankings(self, genemode="unobserved", threshold=0.5):
        term_names = self.adata.var_names
        if genemode == "unobserved":
            nodes, term_scores = self.ntssb.get_node_unobs()
        elif genemode == "observed":
            nodes, term_scores = self.ntssb.get_node_obs()
        else:
            nodes, term_scores = self.ntssb.get_avg_node_exp()
        term_scores = np.abs(np.array(term_scores))
        top_terms = []
        for k, node in enumerate(nodes):
            top_terms_idx = (term_scores[k]).argsort()[::-1]
            top_terms_idx = top_terms_idx[
                np.where(term_scores[k][top_terms_idx] >= threshold)
            ]
            top_terms_list = [term_names[i] for i in top_terms_idx]
            top_terms.append(top_terms_list)
        return dict(zip([node.label for node in nodes], top_terms))

    def compute_pathway_enrichments(
        self,
        threshold=0.5,
        cutoff=0.05,
        genemode="unobserved",
        libs=["MSigDB_Hallmark_2020"],
    ):
        enrichments = []
        gene_rankings = self.get_rankings(genemode=genemode, threshold=threshold)
        for node in tqdm(gene_rankings):
            enr = []
            if len(gene_rankings[node]) > 0:
                enr = gp.enrichr(
                    gene_list=gene_rankings[node],
                    gene_sets=libs,
                    organism="Human",
                    outdir="test/enrichr",
                    cutoff=cutoff,
                ).results
                enr = enr[["Term", "Adjusted P-value"]].set_index("Term").T
            enrichments.append(enr)
        self.enrichments = dict(zip([node for node in gene_rankings], enrichments))

    def compute_pivot_likelihoods(self, clone="B", normalized=True):
        """
        For the given clone, compute the tree ELBO for each possible pivot and
        return a dictionary of pivots and their ELBOs.
        """
        if clone == "A":
            raise ValueError(
                "The root clone was selected, which by definition \
            does not have parent nodes. Please select a non-root clone."
            )

        tssbs = ntssb.get_subtrees()
        labels = [tssb.label for tssb in tssbs]
        tssb = subtrees[np.where(np.array(labels) == clone)[0]]

        parent_tssb = tssb.root["node"].parent().tssb
        possible_pivots = parent_tssb.get_nodes()

        pivot_likelihoods = dict()
        for pivot in possible_pivots:
            if len(possible_pivots) == 1:
                possible_pivots[pivot] = self.ntssb.elbo
                logger.warning(f"Clone {clone} has only one possible parent node.")
                break
            ntssb = deepcopy(self.ntssb)
            ntssb.pivot_reattach_to(clone, pivot.label)
            ntssb.optimize_elbo()
            pivot_likelihoods[pivot.label] = ntssb.elbo

        if normalize:
            labels = list(pivot_likelihoods.get_keys())
            vals = list(pivot_likelihoods.get_values())
            vals = np.array(vals) / np.sum(vals)
            pivot_likelihoods = dict(zip(labels, vals.tolist()))

        return pivot_likelihoods

    def get_cnv_exp(self, max_level=4, method="scatrex"):
        cnv_levels = np.unique(self.observed_tree.adata.X)
        exp_levels = []
        for cnv in cnv_levels:
            gene_avg = []
            for gene in range(self.adata.X.shape[1]):
                cells = np.where(self.adata.layers[f"{method}_cnvs"][:, gene] == cnv)[0]
                if len(cells) > 0:
                    gene_avg.append(np.mean(self.adata.X[cells, gene]))
            exp_levels.append(np.array(gene_avg))

        try:
            max_level_pos = np.where(cnv_levels >= max_level)[0][0]
        except IndexError:
            logger.warning(f"{max_level} not present in {cnv_levels}.")
            max_level_pos = np.argmax(cnv_levels)
        exp_levels[max_level_pos] = np.concatenate(exp_levels[max_level_pos:])
        exp_levels = exp_levels[: max_level_pos + 1]
        cnv_levels = list(cnv_levels[: max_level_pos + 1].astype(int))
        cnv_levels_labels = list(cnv_levels)
        cnv_levels_labels[max_level_pos] = f"{cnv_levels[max_level_pos]}+"

        d = dict()
        for i in range(len(cnv_levels)):
            d[cnv_levels[i]] = dict(label=cnv_levels_labels[i], exp=exp_levels[i])

        return d

    def get_cnv_vs_state(self):
        node_dict = dict()
        nodes, locs = np.unique(self.adata.obs["scatrex_node"], return_index=True)
        for i, node in enumerate(nodes):
            node_dict[node] = dict()
            cnv = self.adata.layers["scatrex_cnvs"][locs[i]]
            xi = self.adata.layers["scatrex_xi"][locs[i]]
            cnv_levels = np.unique(cnv).astype(int)
            for cnv_level in cnv_levels:
                genes_in_level = np.where(cnv == cnv_level)[0]
                node_dict[node][cnv_level] = dict()
                node_dict[node][cnv_level]["state"] = xi[genes_in_level]
                node_dict[node][cnv_level]["genes"] = genes_in_level

        return node_dict

    def get_concordances(self):
        node_dict = dict()
        nodes, locs = np.unique(self.adata.obs["scatrex_node"], return_index=True)
        for i, node in enumerate(nodes):
            cnv = self.adata.layers["scatrex_cnvs"][locs[i]]
            xi = self.adata.layers["scatrex_xi"][locs[i]]
            node_dict[node] = np.zeros(cnv.shape)
            non_2 = np.where(cnv != 2)[0]
            node_dict[node][non_2] = xi[non_2] / np.log(cnv[non_2] / 2)
        return node_dict

    def get_discordant_genes_node(self, node_concordances):
        discordant_genes = np.where(node_concordances < 0)[0]
        sorted_discordant_genes = discordant_genes[
            np.argsort(node_concordances[discordant_genes])
        ]
        sorted_concordances = node_concordances[sorted_discordant_genes]
        return sorted_discordant_genes, sorted_concordances

    def get_discordant_genes(self, concordances):
        dg = []
        sc = []
        nodes = []
        for node in np.unique(self.adata.obs["scatrex_node"])[1:]:
            a, b = self.get_discordant_genes_node(concordances[node])
            dg.append(a)
            sc.append(b)
            nodes.append(len(a) * [node])
        dg = np.concatenate(dg)
        sc = np.concatenate(sc)
        nodes = np.concatenate(nodes)

        sorted_discordant_genes = dg[np.argsort(sc)]
        sorted_concordances = sc[np.argsort(sc)]
        sorted_nodes = nodes[np.argsort(sc)]

        unique, indices = np.unique(sorted_discordant_genes, return_index=True)

        unique_discordant_genes = sorted_discordant_genes[indices]
        unique_concordances = sorted_concordances[indices]
        unique_nodes = sorted_nodes[indices]

        sorted_discordant_genes = unique_discordant_genes[
            np.argsort(unique_concordances)
        ]
        sorted_concordances = unique_concordances[np.argsort(unique_concordances)]
        sorted_nodes = unique_nodes[np.argsort(unique_concordances)]

        return sorted_discordant_genes, sorted_concordances, sorted_nodes

    def plot_discordant_genes(
        self,
        sorted_discordant_genes,
        sorted_concordances,
        sorted_nodes=None,
        gene_annots=None,
        top=20,
        figsize=None,
    ):
        sorted_discordant_genes = np.array(sorted_discordant_genes)[:top]
        sorted_concordances = np.array(sorted_concordances)[:top]
        if sorted_nodes is not None:
            sorted_nodes = np.array(sorted_nodes)[:top]
        if gene_annots is not None:
            gene_annots = gene_annots[sorted_discordant_genes]

        plt.figure(figsize=figsize)
        plt.axhline(1, color="gray", alpha=0.6, ls="--", label="Perfect discordance")
        if sorted_nodes is not None:
            for node in np.unique(sorted_nodes):
                idx = np.where(sorted_nodes == node)[0]
                plt.scatter(
                    np.arange(len(sorted_concordances))[idx],
                    -sorted_concordances[idx],
                    label=node,
                    color=self.ntssb.node_dict[node]["node"].tssb.color,
                )
        elif gene_annots is not None:
            for gene_annot in np.unique(gene_annots):
                idx = np.where(gene_annots == gene_annot)[0]
                plt.scatter(
                    np.arange(len(sorted_concordances))[idx],
                    -sorted_concordances[idx],
                    label=gene_annot,
                )
        else:
            plt.scatter(np.arange(len(sorted_concordances)), -sorted_concordances)
        if sorted_nodes is not None or gene_annots is not None:
            plt.legend()
        plt.xticks(range(len(sorted_concordances)), labels=sorted_discordant_genes)
        plt.xlabel("Discordant genes")
        plt.ylabel("Negative concordance score")
        maximum_discordance = np.max(-sorted_concordances)
        plt.ylim([0, np.max([maximum_discordance, 1]) + 0.2])
        plt.show()

    def _plot_cnv_vs_state_node(
        self,
        node,
        mapping=None,
        concordances=None,
        state_range=[-1, 1],
        cnv_range=[1, 2, 3, 4],
        ax=None,
        figsize=None,
        ylabel="Cell state",
        xlabel="Copy number",
        alpha=1,
        colorbar=False,
    ):
        if mapping is None:
            mapping = self.get_cnv_vs_state()

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        norm = matplotlib.colors.Normalize(vmin=state_range[0], vmax=state_range[1])
        if concordances is not None:
            norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        plt.sca(ax)
        for cnv in cnv_range:
            if cnv in list(mapping[node].keys()):
                c = mapping[node][cnv]["state"]
                if concordances is not None:
                    c = concordances[node][mapping[node][cnv]["genes"]]
                plt.scatter(
                    cnv * np.ones((len(mapping[node][cnv]["state"]))),
                    mapping[node][cnv]["state"],
                    c=c,
                    norm=norm,
                    alpha=alpha,
                )
        plt.axhline(0, alpha=0.6, color="gray")
        plt.xticks(cnv_range)
        plt.xlim([cnv_range[0] - 0.5, cnv_range[-1] + 0.5])
        plt.ylim([state_range[0] - 0.2, state_range[-1] + 0.2])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(node)
        if concordances:
            if colorbar:
                cbar = plt.colorbar(label="Concordance")
                cbar.set_ticks([-1, 0, 1])
                cbar.set_ticklabels(["<= -1", "0", ">= +1"])

    def plot_cnv_vs_state(
        self,
        nodes=None,
        mapping=None,
        concordances=None,
        state_range=[-1, 1],
        cnv_range=[1, 2, 3, 4],
        figsize=None,
        alpha=1,
    ):
        if mapping is None:
            mapping = self.get_cnv_vs_state()

        if nodes is None:
            nodes = sorted(list(mapping.keys()))
        elif isinstance(nodes, str):
            self._plot_cnv_vs_state_node(
                nodes,
                mapping=mapping,
                concordances=concordances,
                state_range=state_range,
                cnv_range=cnv_range,
                alpha=alpha,
            )
            plt.show()
            return

        fig, axes = plt.subplots(1, len(nodes), figsize=figsize, sharey=True)
        for i, node in enumerate(nodes):
            node = nodes[i]
            ylabel = ""
            colorbar = False
            if i == 0:
                ylabel = "Cell state"
            if i == len(nodes) - 1:
                colorbar = True
            self._plot_cnv_vs_state_node(
                node,
                mapping=mapping,
                concordances=concordances,
                state_range=state_range,
                cnv_range=cnv_range,
                ax=axes[i],
                ylabel=ylabel,
                colorbar=colorbar,
                alpha=alpha,
            )

        plt.show()

    def get_gene_inheritance_scores_in_lineage(
        self,
        lineage,
        xi_threshold=0.1,
        chi_threshold=0.1,
        prob=True,
        direction=True,
        window=1,
        neutral=False,
    ):
        lineage_root_idx = np.where(self.adata.obs["scatrex_node"] == lineage[0])[0]

        genes = np.array(
            list(
                set(
                    np.where(
                        self.adata.layers["scatrex_om"][lineage_root_idx[0]]
                        > chi_threshold
                    )[0]
                ).intersection(
                    set(
                        np.where(
                            np.abs(self.adata.layers["scatrex_xi"][lineage_root_idx[0]])
                            > xi_threshold
                        )[0]
                    )
                )
            )
        ).astype(int)

        if len(genes) == 0:
            raise ValueError(
                f"No genes had a cell state event in node {lineage[0]} with indicated thresholds."
            )

        # Filter: get only CNV-constant events in the lineage
        lineage_idx = np.where(self.adata.obs["scatrex_node"].isin(lineage))[0]
        if neutral:
            genes = genes[
                np.where(
                    np.all(
                        self.adata.layers["scatrex_cnvs"][lineage_idx][:, genes] == 2,
                        axis=0,
                    )
                )[0]
            ]
        else:
            genes = genes[
                np.where(
                    np.var(
                        self.adata.layers["scatrex_cnvs"][lineage_idx][:, genes], axis=0
                    )
                    == 0
                )[0]
            ]

        gene_names = self.adata.var_names[genes]

        gene_scores = dict()
        for gene_idx, gene in enumerate(genes):
            parent_concordances = []
            for idx, node in enumerate(lineage):
                if idx != 0:
                    parent_mean = self.ntssb.node_dict[lineage[idx - 1]][
                        "node"
                    ].variational_parameters["locals"]["unobserved_factors_mean"][gene]
                    node_mean = self.ntssb.node_dict[lineage[idx]][
                        "node"
                    ].variational_parameters["locals"]["unobserved_factors_mean"][gene]
                    if prob:
                        parent_std = np.exp(
                            self.ntssb.node_dict[lineage[idx - 1]][
                                "node"
                            ].variational_parameters["locals"][
                                "unobserved_factors_log_std"
                            ][
                                gene
                            ]
                        )
                        node_std = np.exp(
                            self.ntssb.node_dict[lineage[idx]][
                                "node"
                            ].variational_parameters["locals"][
                                "unobserved_factors_log_std"
                            ][
                                gene
                            ]
                        )
                        if direction:
                            if parent_mean > 0:
                                prob_score = 1 - stats.norm.cdf(
                                    parent_mean - parent_std, node_mean, node_std
                                )
                            else:
                                prob_score = stats.norm.cdf(
                                    parent_mean + parent_std, node_mean, node_std
                                )
                        else:
                            prob_score = stats.norm.cdf(
                                parent_mean + parent_std, node_mean, node_std
                            ) - stats.norm.cdf(
                                parent_mean - parent_std, node_mean, node_std
                            )
                        parent_concordances.append(prob_score)
                    else:  # negative absolute distance
                        parent_concordances.append(-np.abs(parent_mean - node_mean))
            gene_name = gene_names[gene_idx]
            gene_scores[gene_name] = dict(idx=gene, score=np.mean(parent_concordances))

        return gene_scores

    def plot_state_inheritance(
        self, lineage, gene_scores=None, figsize=None, ax=None, **gene_scores_kwargs
    ):
        # Plots the variational distributions of cell states across a lineage
        # Useful for identifying heritable effects

        if gene_scores is None:
            gene_scores = self.get_gene_inheritance_scores_in_lineage(
                lineage, **gene_scores_kwargs
            )

        # Sort by highest scoring to lowest scoring
        scores = np.array(list(gene["score"] for gene in gene_scores.values()))
        order = np.argsort(scores)[::-1]
        scores = scores[order]
        genes = np.array(list(gene_scores.keys()))[order]
        indices = np.array(list(gene["idx"] for gene in gene_scores.values()))[order]

        if ax is None:
            plt.figure(figsize=figsize)
        else:
            plt.sca(ax)
        plt.axhline(0, alpha=0.5, ls="--", color="gray")
        gene_offset = 0
        node_step = (
            1 + 0.5
        )  # each node's dist is normalized to height 1, plus some small offset
        gene_length = len(lineage) * node_step
        gene_ticks = []
        for gene_idx, gene in enumerate(genes):
            gene_offset = gene_idx * (gene_length + 0.5 * gene_length)
            gene_ticks.append(gene_offset + 0.5 * gene_length)
            node_offset = 0
            for node_idx, node in enumerate(lineage):
                node_offset = node_idx * node_step
                mean = self.ntssb.node_dict[node]["node"].variational_parameters[
                    "locals"
                ]["unobserved_factors_mean"][indices[gene_idx]]
                std = np.exp(
                    self.ntssb.node_dict[node]["node"].variational_parameters["locals"][
                        "unobserved_factors_log_std"
                    ][indices[gene_idx]]
                )
                yy = np.arange(mean - 3 * std, mean + 3 * std, 0.001)
                label = None
                if gene_idx == len(genes) - 1:
                    label = node
                pdf = stats.norm.pdf(yy, mean, std)
                pdf = pdf / np.max(pdf)
                plt.plot(
                    pdf + gene_offset + node_offset,
                    yy,
                    label=label,
                    color=self.ntssb.node_dict[node]["node"].tssb.color,
                    lw=4,
                    alpha=0.7,
                )
        if ax is None:
            plt.xticks(gene_ticks, labels=genes)
            plt.title(" -> ".join(lineage))
            plt.ylabel("Cell state factor")
        ax = plt.gca()
        return ax, gene_ticks

    def plot_state_inheritance_scores(
        self, lineage, gene_scores=None, figsize=None, **gene_scores_kwargs
    ):
        # Includes actual scores as a subplot
        if gene_scores is None:
            gene_scores = self.get_gene_inheritance_scores_in_lineage(
                lineage, **gene_scores_kwargs
            )

        # Sort by highest scoring to lowest scoring
        scores = np.array(list(gene["score"] for gene in gene_scores.values()))
        order = np.argsort(scores)[::-1]
        scores = scores[order]
        genes = np.array(list(gene_scores.keys()))[order]

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
        ax, gene_ticks = self.plot_state_inheritance(
            lineage, gene_scores=gene_scores, ax=axes[0]
        )

        ax.set_title(" -> ".join(lineage))
        ax.set_ylabel("Cell state factor")

        plt.sca(axes[1])
        plt.bar(gene_ticks, scores)
        plt.ylabel("Heritability score")
        plt.xlabel(f"State-affected genes in {lineage[0]}")
        plt.xticks(gene_ticks, labels=genes)
        plt.show()

    def plot_cnv_inheritance(self):
        # Plots the combined effect of each CNV across a lineage
        # Can be useful to check if the provided CNV tree is true
        raise NotImplementedError

    def plot_proportions(self, dna=True, rna=True, show=True):
        if dna:
            dna_props = np.array(
                [
                    self.observed_tree.tree_dict[node]["weight"]
                    for node in self.observed_tree.tree_dict
                ]
            )
            nodes_labels = np.array([node for node in self.observed_tree.tree_dict])
            colors = np.array(
                [
                    self.observed_tree.tree_dict[node]["color"]
                    for node in self.observed_tree.tree_dict
                ]
            )
            s = np.argsort(np.array(nodes_labels))
            dna_nodes_labels = nodes_labels[s]
            dna_colors = colors[s]
            dna_props = dna_props[s]

        if rna:
            rna_nodes, rna_props = self.ntssb.get_node_data_sizes(
                normalized=True, super_only=True
            )
            nodes_labels = [node.label for node in nodes]
            s = np.argsort(np.array(nodes_labels))
            rna_nodes = np.array(rna_nodes)[s]
            rna_nodes_labels = np.array(nodes_labels)[s]
            rna_props = np.array(rna_props)[s]
            rna_colors = [node.tssb.color for node in nodes]

        if dna and rna:
            if set(dna_nodes_labels) != set(rna_nodes_labels):
                raise ValueError(
                    f"DNA and RNA nodes are not the same! DNA: {dna_nodes_labels}, RNA: {rna_nodes_labels}"
                )
            handles = []
            for i, node in enumerate(rna_nodes):
                dna_bottom = np.sum(dna_props[:i])
                rna_bottom = np.sum(rna_props[:i])
                h = plt.bar(
                    ["DNA", "RNA"],
                    [dna_props[i], rna_props[i]],
                    color=[dna_colors[i], rna_colors[i]],
                    bottom=[dna_bottom, rna_bottom],
                )
                handles.append(h[0])
                plt.legend(handles, nodes_labels)
        else:
            if dna:
                plt.bar(dna_nodes_labels, dna_props, color=dna_colors)
            if rna:
                plt.bar(rna_nodes_labels, rna_props, color=rna_colors)

        if show:
            plt.show()
