from numpy import *
import numpy as np
from numpy.random import *

import jax
import jax.numpy as jnp
from jax.api import jit, grad, vmap
from jax import random, ops
from jax.experimental import optimizers
from jax.scipy.stats import norm
from jax.scipy.stats import gamma
from jax.scipy.stats import poisson
from jax.scipy.special import logit

from ...util import *
from ...ntssb.node import *
from ...ntssb.tree import *

MIN_CNV = 1e-6
MAX_XI = 1


class Node(AbstractNode):
    def __init__(
        self,
        is_observed,
        observed_parameters,
        log_lib_size_mean=7.1,
        log_lib_size_std=0.6,
        num_global_noise_factors=4,
        global_noise_factors_precisions_shape=2.0,
        cell_global_noise_factors_weights_scale=1.0,
        unobserved_factors_root_kernel=0.1,
        unobserved_factors_kernel=1.0,
        unobserved_factors_kernel_concentration=0.01,
        unobserved_factors_kernel_rate=1.0,
        frac_dosage=1,
        baseline_shape=0.7,
        num_batches=0,
        **kwargs,
    ):
        super(Node, self).__init__(is_observed, observed_parameters, **kwargs)

        # The observed parameters are the CNVs of all genes
        self.cnvs = np.array(self.observed_parameters)
        self.cnvs[np.where(self.cnvs == 0)[0]] = MIN_CNV
        self.observed_parameters = np.array(self.cnvs)

        self.n_genes = self.cnvs.size

        # Node hyperparameters
        if self.parent() is None:
            self.node_hyperparams = dict(
                log_lib_size_mean=log_lib_size_mean,
                log_lib_size_std=log_lib_size_std,
                num_global_noise_factors=num_global_noise_factors,
                global_noise_factors_precisions_shape=global_noise_factors_precisions_shape,
                cell_global_noise_factors_weights_scale=cell_global_noise_factors_weights_scale,
                unobserved_factors_root_kernel=unobserved_factors_root_kernel,
                unobserved_factors_kernel=unobserved_factors_kernel,
                unobserved_factors_kernel_concentration=unobserved_factors_kernel_concentration,
                unobserved_factors_kernel_rate=unobserved_factors_kernel_rate,
                frac_dosage=frac_dosage,
                baseline_shape=baseline_shape,
                num_batches=num_batches,
            )
        else:
            self.node_hyperparams = self.node_hyperparams_caller()

        self.reset_parameters(**self.node_hyperparams)

    def inherit_parameters(self):
        if not self.is_observed:
            # Make sure we use the right observed parameters
            self.cnvs = self.parent().cnvs

    def get_node_mean(self, log_baseline, unobserved_factors, noise, cnvs):
        node_mean = jnp.exp(
            log_baseline + unobserved_factors + noise + jnp.log(cnvs / 2)
        )
        sum = jnp.sum(node_mean, axis=1).reshape(self.tssb.ntssb.num_data, 1)
        node_mean = node_mean / sum
        return node_mean

    def init_noise_factors(self):
        # Noise
        self.variational_parameters["globals"]["cell_noise_mean"] = np.zeros(
            (self.tssb.ntssb.num_data, self.num_global_noise_factors)
        )
        self.variational_parameters["globals"]["cell_noise_log_std"] = -np.ones(
            (self.tssb.ntssb.num_data, self.num_global_noise_factors)
        )
        self.variational_parameters["globals"]["noise_factors_mean"] = np.zeros(
            (self.num_global_noise_factors, self.n_genes)
        )
        self.variational_parameters["globals"]["noise_factors_log_std"] = -np.ones(
            (self.num_global_noise_factors, self.n_genes)
        )
        self.variational_parameters["globals"]["factor_precision_log_means"] = np.log(
            self.global_noise_factors_precisions_shape
        ) * np.ones((self.num_global_noise_factors))
        self.variational_parameters["globals"]["factor_precision_log_stds"] = -np.ones(
            (self.num_global_noise_factors)
        )

        # Batch effects
        self.variational_parameters["globals"]["batch_effects_mean"] = np.zeros(
            (self.num_batches, self.n_genes)
        )
        self.variational_parameters["globals"]["batch_effects_log_std"] = -np.ones(
            (self.num_batches, self.n_genes)
        )

    def reset_variational_parameters(self, means=True, variances=True):
        if self.parent() is None:
            # Baseline: first value is 1
            if means:
                self.variational_parameters["globals"]["log_baseline_mean"] = np.array(
                    normal_sample(0, 1, self.n_genes - 1)
                )  # np.zeros((self.n_genes-1,))
                if self.tssb.ntssb.data is not None:
                    self.full_data = jnp.array(self.tssb.ntssb.data)
                    # init_baseline = np.mean(self.tssb.ntssb.data, axis=0)
                    # init_log_baseline = np.log(init_baseline / init_baseline[0])[1:]
                    init_baseline = np.mean(
                        self.tssb.ntssb.data
                        / np.sum(self.tssb.ntssb.data, axis=1).reshape(-1, 1)
                        * self.n_genes,
                        axis=0,
                    )
                    init_baseline = init_baseline / init_baseline[0]
                    init_log_baseline = np.log(init_baseline[1:] + 1e-6)
                    self.variational_parameters["globals"][
                        "log_baseline_mean"
                    ] = np.clip(init_log_baseline, -1, 1)
            if variances:
                self.variational_parameters["globals"][
                    "log_baseline_log_std"
                ] = np.array(np.zeros((self.n_genes - 1,)))

            # Overdispersion
            # self.log_od_mean = np.zeros(1)
            # self.log_od_log_std = np.zeros(1)

            if self.tssb.ntssb.num_data is not None:
                # Noise
                if means:
                    self.variational_parameters["globals"][
                        "cell_noise_mean"
                    ] = np.random.normal(
                        0,
                        0.1,
                        (self.tssb.ntssb.num_data, self.num_global_noise_factors),
                    )
                if variances:
                    self.variational_parameters["globals"][
                        "cell_noise_log_std"
                    ] = -np.ones(
                        (self.tssb.ntssb.num_data, self.num_global_noise_factors)
                    )
                if means:
                    self.variational_parameters["globals"][
                        "noise_factors_mean"
                    ] = np.random.normal(
                        0, 0.1, (self.num_global_noise_factors, self.n_genes)
                    )
                if variances:
                    self.variational_parameters["globals"][
                        "noise_factors_log_std"
                    ] = -np.ones((self.num_global_noise_factors, self.n_genes))
                if means:
                    self.variational_parameters["globals"][
                        "factor_precision_log_means"
                    ] = np.log(self.global_noise_factors_precisions_shape) * np.ones(
                        (self.num_global_noise_factors)
                    )
                if variances:
                    self.variational_parameters["globals"][
                        "factor_precision_log_stds"
                    ] = -np.ones((self.num_global_noise_factors))
                if means:
                    self.variational_parameters["globals"][
                        "batch_effects_mean"
                    ] = np.random.normal(0, 0.1, (self.num_batches, self.n_genes))
                if variances:
                    self.variational_parameters["globals"][
                        "batch_effects_log_std"
                    ] = -np.ones((self.num_batches, self.n_genes))

        self.data_ass_logits = np.array([])
        if self.tssb.ntssb.num_data is not None:
            self.data_ass_logits = np.zeros((self.tssb.ntssb.num_data))

        # Sticks
        if means:
            self.variational_parameters["locals"]["nu_log_mean"] = np.array(
                np.log(1.0 * self.tssb.dp_alpha) + np.random.normal(0, 0.01)
            )
        if variances:
            self.variational_parameters["locals"]["nu_log_std"] = np.array(
                np.log(1.0 * self.tssb.dp_alpha)
            )
        if means:
            self.variational_parameters["locals"]["psi_log_mean"] = np.array(
                np.log(1.0 * self.tssb.dp_gamma) + np.random.normal(0, 0.01)
            )
        if variances:
            self.variational_parameters["locals"]["psi_log_std"] = np.array(
                np.log(1.0 * self.tssb.dp_gamma)
            )

        # Unobserved factors
        if means:
            try:
                self.variational_parameters["locals"][
                    "unobserved_factors_mean"
                ] = np.array(
                    self.parent().variational_parameters["locals"][
                        "unobserved_factors_mean"
                    ]
                )
            except AttributeError:
                self.variational_parameters["locals"][
                    "unobserved_factors_mean"
                ] = np.zeros((self.n_genes,))
        if variances:
            self.variational_parameters["locals"][
                "unobserved_factors_log_std"
            ] = -2.0 * np.ones((self.n_genes,))
        if means:
            try:
                kernel_means = np.clip(
                    self.unobserved_factors_kernel_concentration_caller()
                    / (
                        np.exp(
                            (self.parent() != None)
                            * (self.parent().parent() != None)
                            * self.unobserved_factors_kernel_rate_caller()
                            * np.abs(
                                self.parent().variational_parameters["locals"][
                                    "unobserved_factors_mean"
                                ]
                            )
                        )
                    ),
                    self.unobserved_factors_kernel_concentration_caller() / 10,
                    1e2,
                )
                self.variational_parameters["locals"][
                    "unobserved_factors_kernel_log_mean"
                ] = (np.log(kernel_means) - (np.exp(-2) ** 2) / 2)
            except AttributeError:
                self.variational_parameters["locals"][
                    "unobserved_factors_kernel_log_mean"
                ] = np.log(
                    self.unobserved_factors_kernel_concentration_caller()
                ) * np.ones(
                    (self.n_genes,)
                )
        if variances:
            self.variational_parameters["locals"][
                "unobserved_factors_kernel_log_std"
            ] = -2.0 * np.ones((self.n_genes,))
        self.set_mean(
            self.get_mean(
                baseline=np.append(1, np.exp(self.log_baseline_caller())),
                unobserved_factors=self.variational_parameters["locals"][
                    "unobserved_factors_mean"
                ],
            )
        )

    # ========= Functions to initialize node. =========
    def reset_data_parameters(self):
        self.full_data = jnp.array(self.tssb.ntssb.data)
        self.lib_sizes = np.sum(self.tssb.ntssb.data, axis=1).reshape(
            self.tssb.ntssb.num_data, 1
        )
        self.cell_global_noise_factors_weights = normal_sample(
            0,
            self.cell_global_noise_factors_weights_scale,
            size=[self.tssb.ntssb.num_data, self.num_global_noise_factors],
        )
        self.cell_covariates = jnp.array(self.tssb.ntssb.covariates)
        self.num_batches = self.cell_covariates.shape[1]

    def generate_data_params(self):
        self.lib_sizes = 20.0 + np.exp(
            normal_sample(
                self.log_lib_size_mean,
                self.log_lib_size_std,
                size=self.tssb.ntssb.num_data,
            )
        ).reshape(self.tssb.ntssb.num_data, 1)
        self.lib_sizes = np.ceil(self.lib_sizes)
        self.cell_global_noise_factors_weights = normal_sample(
            0,
            self.cell_global_noise_factors_weights_scale,
            size=[self.tssb.ntssb.num_data, self.num_global_noise_factors],
        )
        n_cells = self.tssb.ntssb.num_data
        self.cell_covariates = np.zeros((n_cells, self.num_batches))
        if self.num_batches > 1:
            batches = np.random.choice(self.num_batches, size=n_cells)
            self.cell_covariates[range(n_cells), batches] = 1
        else:
            self.cell_covariates = np.zeros((n_cells, 0))
            self.num_batches = 0

    def reset_parameters(
        self,
        root_params=True,
        down_params=True,
        log_lib_size_mean=6,
        log_lib_size_std=0.8,
        num_global_noise_factors=4,
        global_noise_factors_precisions_shape=2.0,
        cell_global_noise_factors_weights_scale=1.0,
        unobserved_factors_root_kernel=0.1,
        unobserved_factors_kernel=1.0,
        unobserved_factors_kernel_concentration=0.01,
        unobserved_factors_kernel_rate=1.0,
        frac_dosage=1.0,
        baseline_shape=0.1,
        num_batches=0,
    ):
        parent = self.parent()

        if parent is None:  # this is the root
            self.node_hyperparams = dict(
                log_lib_size_mean=log_lib_size_mean,
                log_lib_size_std=log_lib_size_std,
                num_global_noise_factors=num_global_noise_factors,
                global_noise_factors_precisions_shape=global_noise_factors_precisions_shape,
                cell_global_noise_factors_weights_scale=cell_global_noise_factors_weights_scale,
                unobserved_factors_root_kernel=unobserved_factors_root_kernel,
                unobserved_factors_kernel=unobserved_factors_kernel,
                unobserved_factors_kernel_concentration=unobserved_factors_kernel_concentration,
                unobserved_factors_kernel_rate=unobserved_factors_kernel_rate,
                frac_dosage=frac_dosage,
                baseline_shape=baseline_shape,
                num_batches=num_batches,
            )

            if root_params:
                # The root is used to store global parameters:  mu
                # self.log_baseline = normal_sample(0, 1., size=self.n_genes)
                # self.log_baseline[0] = 0.
                # self.baseline = np.exp(self.log_baseline)
                self.baseline_shape = baseline_shape
                self.baseline = np.random.gamma(
                    self.baseline_shape, 1, size=self.n_genes
                )
                # self.baseline = np.concatenate([1, self.baseline])
                self.log_baseline = np.log(self.baseline)

                self.overdispersion = np.exp(normal_sample(0, 1))
                self.log_lib_size_mean = log_lib_size_mean
                self.log_lib_size_std = log_lib_size_std

                # Structured noise: keep all cells' noise factors at the root
                self.num_global_noise_factors = num_global_noise_factors
                self.global_noise_factors_precisions_shape = (
                    global_noise_factors_precisions_shape
                )
                self.cell_global_noise_factors_weights_scale = (
                    cell_global_noise_factors_weights_scale
                )

                self.global_noise_factors_precisions = gamma_sample(
                    global_noise_factors_precisions_shape,
                    1.0,
                    size=self.num_global_noise_factors,
                )
                self.global_noise_factors = normal_sample(
                    0,
                    1.0 / np.sqrt(self.global_noise_factors_precisions),
                    size=[self.n_genes, self.num_global_noise_factors],
                ).T  # K x P

                # Batch effects
                self.num_batches = num_batches
                self.batch_effects_factors = normal_sample(
                    0,
                    1.0,
                    size=[self.n_genes, self.num_batches],
                ).T  # K x P

                self.unobserved_factors_root_kernel = unobserved_factors_root_kernel
                self.unobserved_factors_kernel_concentration = (
                    unobserved_factors_kernel_concentration
                )
                self.unobserved_factors_kernel_rate = unobserved_factors_kernel_rate

                self.frac_dosage = frac_dosage

            self.inert_genes = np.random.choice(
                self.n_genes,
                size=int(self.n_genes * (1.0 - self.frac_dosage)),
                replace=False,
            )

            # Root should not have unobserved factors
            self.unobserved_factors_kernel = 0 * np.array(
                [self.unobserved_factors_root_kernel] * self.n_genes
            )
            self.unobserved_factors = 0 * normal_sample(
                0.0, self.unobserved_factors_kernel
            )

            self.set_mean()

            self.depth = 0.0
        else:  # Non-root node: inherits everything from upstream node
            self.node_hyperparams = self.node_hyperparams_caller()
            if down_params:
                self.unobserved_factors_kernel = gamma_sample(
                    self.unobserved_factors_kernel_concentration_caller(),
                    np.exp(np.abs(parent.unobserved_factors)),
                    size=self.n_genes,
                )
                # Make sure some genes are affected in unobserved nodes
                if not self.is_observed:
                    top_genes = np.argsort(self.unobserved_factors_kernel)[::-1][:5]
                    self.unobserved_factors_kernel[top_genes] = np.max(
                        [5.0, np.max(self.unobserved_factors_kernel)]
                    )
                self.unobserved_factors = normal_sample(
                    parent.unobserved_factors, self.unobserved_factors_kernel
                )
                self.unobserved_factors = np.clip(
                    self.unobserved_factors, -MAX_XI, MAX_XI
                )
                if not self.is_observed:
                    self.unobserved_factors[
                        top_genes
                    ] = MAX_XI  # just force an amplification

            # Observation mean
            self.set_mean()

            self.depth = parent.depth + 1

    def set_mean(self, node_mean=None, variational=False):
        if node_mean is not None:
            self.node_mean = node_mean
        else:
            if variational:
                self.node_mean = (
                    np.append(1, np.exp(self.log_baseline_caller()))
                    * self.cnvs
                    / 2
                    * np.exp(
                        (self.parent() is not None)
                        * self.variational_parameters["locals"][
                            "unobserved_factors_mean"
                        ]
                    )
                )
            else:
                self.node_mean = (
                    self.baseline_caller()
                    * self.cnvs
                    / 2
                    * np.exp(self.unobserved_factors)
                )
            self.node_mean = self.node_mean / np.sum(self.node_mean)

    def get_mean(
        self,
        baseline=None,
        unobserved_factors=None,
        noise=0.0,
        batch_effects=0.0,
        cell_factors=None,
        global_factors=None,
        cnvs=None,
        norm=True,
        inert_genes=None,
    ):
        baseline = (
            np.append(1, np.exp(self.log_baseline_caller()))
            if baseline is None
            else baseline
        )
        unobserved_factors = (
            self.variational_parameters["locals"]["unobserved_factors_mean"]
            if unobserved_factors is None
            else unobserved_factors
        )
        unobserved_factors *= self.parent() is not None
        cnvs = self.cnvs if cnvs is None else cnvs
        if inert_genes is not None:
            cnvs = np.array(cnvs)
            inert_genes = np.array(inert_genes)
            zero_genes = np.where(cnvs == MIN_CNV)[0]
            cnvs[
                inert_genes
            ] = 2.0  # set these genes to 2, i.e., act like they have no CNV
            cnvs[zero_genes] = MIN_CNV  # (except if they are zero)
        node_mean = (
            baseline * cnvs / 2 * np.exp(unobserved_factors + noise + batch_effects)
        )
        if norm:
            if len(node_mean.shape) == 1:
                sum = np.sum(node_mean)
            else:
                sum = np.sum(node_mean, axis=1)
            if len(sum.shape) > 0:
                sum = sum.reshape(-1, 1)
            node_mean = node_mean / sum
        return node_mean

    def get_noise(self, variational=False):
        return self.cell_global_noise_factors_weights_caller(
            variational=variational
        ).dot(self.global_noise_factors_caller(variational=variational))

    def get_batch_effects(self, variational=False):
        return self.cell_covariates_caller().dot(
            self.batch_effects_factors_caller(variational=variational)
        )

    # ========= Functions to take samples from node. =========
    def sample_observation(self, n):
        noise = self.cell_global_noise_factors_weights_caller()[n].dot(
            self.global_noise_factors_caller()
        )
        batch_effects = self.cell_global_noise_factors_weights_caller()[n].dot(
            self.global_noise_factors_caller()
        )
        node_mean = self.get_mean(
            unobserved_factors=self.unobserved_factors,
            baseline=self.baseline_caller(),
            noise=noise,
            batch_effects=batch_effects,
            inert_genes=self.inert_genes_caller(),
        )
        s = multinomial_sample(self.lib_sizes_caller()[n], node_mean)
        # s = negative_binomial_sample(self.lib_sizes_caller()[n] * node_mean, 0.01)
        return s

    # ========= Functions to evaluate node's parameters. =========
    def log_baseline_logprior(self, x=None):
        if x is None:
            x = np.log(self.baseline)
        return normal_lpdf(x, 0, 1)

    def log_overdispersion_logprior(self, x=None):
        if x is None:
            x = np.log(self.overdispersion)
        return normal_lpdf(x, 0, 1)

    def global_noise_factors_logprior(self):
        return normal_lpdf(
            self.global_noise_factors,
            0.0,
            1.0 / np.sqrt(self.global_noise_factors_precisions),
        )

    def cell_global_noise_factors_logprior(self):
        return normal_lpdf(
            self.cell_global_noise_factors_weights,
            0.0,
            self.cell_global_noise_factors_weights_scale,
        )

    def unobserved_factors_logprior(self, x=None):
        if x is None:
            x = self.unobserved_factors

        if self.parent() is not None:
            if self.is_observed:
                llp = normal_lpdf(
                    x,
                    self.parent().unobserved_factors,
                    self.unobserved_factors_root_kernel,
                )
            else:
                llp = normal_lpdf(
                    x, self.parent().unobserved_factors, self.unobserved_factors_kernel
                )
        else:
            llp = normal_lpdf(x, 0.0, self.unobserved_factors_root_kernel)

        return llp

    def logprior(self):
        # Prior of current node
        llp = self.unobserved_factors_logprior()
        if self.parent() is None:
            llp = (
                llp
                + self.global_noise_factors_logprior()
                + self.cell_global_noise_factors_logprior()
            )
            llp = llp + self.log_baseline_logprior()

        return llp

    def loglh(self, n, variational=False, axis=None):
        noise = self.get_noise(variational=variational)[n]
        batch_effects = self.get_batch_effects(variational=variational)[n]
        unobs_factors = self.unobserved_factors
        baseline = self.baseline_caller()
        if variational:
            unobs_factors = self.variational_parameters["locals"][
                "unobserved_factors_mean"
            ]
            baseline = np.append(1, np.exp(self.log_baseline_caller(variational=True)))
        node_mean = self.get_mean(
            baseline=baseline,
            unobserved_factors=unobs_factors,
            noise=noise,
            batch_effects=batch_effects,
        )
        lib_sizes = self.lib_sizes_caller()[n]
        return jax.partial(jit, static_argnums=2)(poisson_lpmf)(
            jnp.array(self.tssb.ntssb.data[n]), lib_sizes * node_mean, axis=axis
        )

    def complete_loglh(self):
        return self.loglh(list(self.data))

    def logprob(self, n):
        # Add prior
        l = self.loglh(n) + self.logprior()

        # Add prob of children nodes given current node's parameters
        for child in self.children():
            l = l + child.unobserved_factors_logprior()

        return l

    # Sum over all data points attached to this node
    def complete_logprob(self):
        return self.logprob(list(self.data))

    # Sum over all data
    def full_logprob(self):
        return self.logprob(list(range(len(self.tssb.ntssb.data))))

    # ========= Functions to acess root's parameters. =========
    def node_hyperparams_caller(self):
        if self.parent() is None:
            return self.node_hyperparams
        else:
            return self.parent().node_hyperparams_caller()

    def global_noise_factors_precisions_shape_caller(self):
        if self.parent() is None:
            return self.global_noise_factors_precisions_shape
        else:
            return self.parent().global_noise_factors_precisions_shape_caller()

    def cell_covariates_caller(self):
        if self.parent() is None:
            return self.cell_covariates
        else:
            return self.parent().cell_covariates_caller()

    def lib_sizes_caller(self):
        if self.parent() is None:
            return self.lib_sizes
        else:
            return self.parent().lib_sizes_caller()

    def log_kernel_mean_caller(self):
        if self.parent() is None:
            return self.logkernel_means
        else:
            return self.parent().log_kernel_mean_caller()

    def kernel_caller(self):
        if self.parent() is None:
            return self.kernel
        else:
            return self.parent().kernel_caller()

    def node_std_caller(self):
        if self.parent() is None:
            return self.node_std
        else:
            return self.parent().node_std_caller()

    def inert_genes_caller(self):
        if self.parent() is None:
            return self.inert_genes
        else:
            return self.parent().inert_genes_caller()

    def log_baseline_caller(self, variational=True):
        if self.parent() is None:
            if variational:
                return self.variational_parameters["globals"]["log_baseline_mean"]
            else:
                return self.log_baseline
        else:
            return self.parent().log_baseline_caller(variational=variational)

    def baseline_caller(self):
        if self.parent() is None:
            return self.baseline
        else:
            return self.parent().baseline_caller()

    def log_overdispersion_caller(self):
        if self.parent() is None:
            return self.log_od_mean
        else:
            return self.parent().log_overdispersion_caller()

    def overdispersion_caller(self):
        if self.parent() is None:
            return self.overdispersion
        else:
            return self.parent().overdispersion_caller()

    def unobserved_factors_damp_caller(self):
        if self.parent() is None:
            return self.unobserved_factors_damp
        else:
            return self.parent().unobserved_factors_damp_caller()

    def unobserved_factors_kernel_concentration_caller(self):
        if self.parent() is None:
            return self.unobserved_factors_kernel_concentration
        else:
            return self.parent().unobserved_factors_kernel_concentration_caller()

    def unobserved_factors_kernel_rate_caller(self):
        if self.parent() is None:
            return self.unobserved_factors_kernel_rate
        else:
            return self.parent().unobserved_factors_kernel_rate_caller()

    def unobserved_factors_root_kernel_caller(self):
        if self.parent() is None:
            return self.unobserved_factors_root_kernel
        else:
            return self.parent().unobserved_factors_root_kernel_caller()

    def unobserved_factors_kernel_caller(self):
        if self.parent() is None:
            return self.unobserved_factors_kernel
        else:
            return self.parent().unobserved_factors_kernel_caller()

    def cell_global_noise_factors_weights_caller(self, variational=False):
        if self.parent() is None:
            if variational:
                return self.variational_parameters["globals"]["cell_noise_mean"]
            else:
                return self.cell_global_noise_factors_weights
        else:
            return self.parent().cell_global_noise_factors_weights_caller(
                variational=variational
            )

    def global_noise_factors_caller(self, variational=False):
        if self.parent() is None:
            if variational:
                return self.variational_parameters["globals"]["noise_factors_mean"]
            else:
                return self.global_noise_factors
        else:
            return self.parent().global_noise_factors_caller(variational=variational)

    def batch_effects_factors_caller(self, variational=False):
        if self.parent() is None:
            if variational:
                return self.variational_parameters["globals"]["batch_effects_mean"]
            else:
                return self.batch_effects_factors
        else:
            return self.parent().batch_effects_factors_caller(variational=variational)

    def set_event_string(
        self,
        var_names=None,
        estimated=True,
        unobs_threshold=1.0,
        kernel_threshold=1.0,
        max_len=5,
        event_fontsize=14,
    ):
        if var_names is None:
            var_names = np.arange(self.n_genes).astype(int).astype(str)

        unobserved_factors = self.unobserved_factors
        unobserved_factors_kernel = self.unobserved_factors_kernel
        if estimated:
            unobserved_factors = self.variational_parameters["locals"][
                "unobserved_factors_mean"
            ]
            unobserved_factors_kernel = np.exp(
                self.variational_parameters["locals"][
                    "unobserved_factors_kernel_log_mean"
                ]
            )

        # Up-regulated
        up_color = "red"
        up_list = np.where(
            np.logical_and(
                unobserved_factors > unobs_threshold,
                unobserved_factors_kernel > kernel_threshold,
            )
        )[0]
        sorted_idx = np.argsort(unobserved_factors[up_list])[:max_len]
        up_list = up_list[sorted_idx]
        up_str = ""
        if len(up_list) > 0:
            up_str = (
                f'<font point-size="{event_fontsize}" color="{up_color}">+</font>'
                + ",".join(var_names[up_list])
            )

        # Down-regulated
        down_color = "blue"
        down_list = np.where(
            np.logical_and(
                unobserved_factors < -unobs_threshold,
                unobserved_factors_kernel > kernel_threshold,
            )
        )[0]
        sorted_idx = np.argsort(-unobserved_factors[down_list])[:max_len]
        down_list = down_list[sorted_idx]
        down_str = ""
        if len(down_list) > 0:
            down_str = (
                f'<font point-size="{event_fontsize}" color="{down_color}">-</font>'
                + ",".join(var_names[down_list])
            )

        self.event_str = up_str
        sep_str = ""
        if len(up_list) > 0 and len(down_list) > 0:
            sep_str = "<br/><br/>"
        self.event_str = self.event_str + sep_str + down_str
