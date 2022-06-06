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
        global_noise_factors_precisions_shape=100.0,
        cell_global_noise_factors_weights_scale=1.0,
        unobserved_factors_root_kernel=0.1,
        unobserved_factors_kernel=1.0,
        unobserved_factors_kernel_concentration=0.01,
        unobserved_factors_kernel_rate=100.0,
        frac_dosage=1,
        baseline_shape=0.7,
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
                    ] = np.zeros(
                        (self.tssb.ntssb.num_data, self.num_global_noise_factors)
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
                    ] = np.zeros((self.num_global_noise_factors, self.n_genes))
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

        self.data_ass_logits = np.zeros((self.tssb.ntssb.num_data))

        # Sticks
        if means:
            self.variational_parameters["locals"]["nu_log_mean"] = np.array(
                np.log(1.0 * self.tssb.dp_alpha)
            )
        if variances:
            self.variational_parameters["locals"]["nu_log_std"] = np.array(
                np.log(1.0 * self.tssb.dp_alpha)
            )
        if means:
            self.variational_parameters["locals"]["psi_log_mean"] = np.array(
                np.log(1.0 * self.tssb.dp_gamma)
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
                ] = self.parent().variational_parameters["locals"][
                    "unobserved_factors_mean"
                ]
            except AttributeError:
                self.variational_parameters["locals"][
                    "unobserved_factors_mean"
                ] = np.zeros((self.n_genes,))
        if variances:
            self.variational_parameters["locals"][
                "unobserved_factors_log_std"
            ] = -2.0 * np.ones((self.n_genes,))
        if means:
            self.variational_parameters["locals"][
                "unobserved_factors_kernel_log_mean"
            ] = np.log(self.unobserved_factors_kernel_concentration_caller()) * np.ones(
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
        noise=None,
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
        node_mean = None
        if noise is not None:
            node_mean = baseline * cnvs / 2 * np.exp(unobserved_factors + noise)
        else:
            node_mean = baseline * cnvs / 2 * np.exp(unobserved_factors)
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

    # ========= Functions to take samples from node. =========
    def sample_observation(self, n):
        noise = self.cell_global_noise_factors_weights_caller()[n].dot(
            self.global_noise_factors_caller()
        )
        node_mean = self.get_mean(
            unobserved_factors=self.unobserved_factors,
            baseline=self.baseline_caller(),
            noise=noise,
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
        unobs_factors = self.unobserved_factors
        baseline = self.baseline_caller()
        if variational:
            unobs_factors = self.variational_parameters["locals"][
                "unobserved_factors_mean"
            ]
            baseline = np.append(1, np.exp(self.log_baseline_caller(variational=True)))
        node_mean = self.get_mean(
            baseline=baseline, unobserved_factors=unobs_factors, noise=noise
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

    # ========= Functions to define the tree ELBO. ============

    def compute_elbo(
        self,
        rng,
        cnvs,
        parent_vector,
        children_vector,
        ancestor_nodes_indices,
        tssb_indices,
        previous_branches_indices,
        tssb_weights,
        dp_alphas,
        dp_gammas,
        node_mask,
        data_mask_subset,
        indices,
        do_global,
        global_only,
        sticks_only,
        nu_sticks_log_means,
        nu_sticks_log_stds,
        psi_sticks_log_means,
        psi_sticks_log_stds,
        unobserved_means,
        unobserved_log_stds,
        log_unobserved_factors_kernel_means,
        log_unobserved_factors_kernel_log_stds,
        log_baseline_mean,
        log_baseline_log_std,
        cell_noise_mean,
        cell_noise_log_std,
        noise_factors_mean,
        noise_factors_log_std,
        factor_precision_log_means,
        factor_precision_log_stds,
    ):
        elbo, ll, kl, node_kl = self._compute_elbo(
            rng,
            cnvs,
            parent_vector,
            children_vector,
            ancestor_nodes_indices,
            tssb_indices,
            previous_branches_indices,
            tssb_weights,
            dp_alphas,
            dp_gammas,
            node_mask,
            data_mask_subset,
            indices,
            do_global,
            global_only,
            sticks_only,
            nu_sticks_log_means,
            nu_sticks_log_stds,
            psi_sticks_log_means,
            psi_sticks_log_stds,
            unobserved_means,
            unobserved_log_stds,
            log_unobserved_factors_kernel_means,
            log_unobserved_factors_kernel_log_stds,
            log_baseline_mean,
            log_baseline_log_std,
            cell_noise_mean,
            cell_noise_log_std,
            noise_factors_mean,
            noise_factors_log_std,
            factor_precision_log_means,
            factor_precision_log_stds,
        )
        return elbo

    # @partial(jit, static_argnums=(0,1,2,3,4,5))
    def _compute_elbo(
        self,
        rng,
        cnvs,
        parent_vector,
        children_vector,
        ancestor_nodes_indices,
        tssb_indices,
        previous_branches_indices,
        tssb_weights,
        dp_alphas,
        dp_gammas,
        node_mask,
        data_mask_subset,
        indices,
        do_global,
        global_only,
        sticks_only,
        nu_sticks_log_means,
        nu_sticks_log_stds,
        psi_sticks_log_means,
        psi_sticks_log_stds,
        unobserved_means,
        unobserved_log_stds,
        log_unobserved_factors_kernel_means,
        log_unobserved_factors_kernel_log_stds,
        log_baseline_mean,
        log_baseline_log_std,
        cell_noise_mean,
        cell_noise_log_std,
        noise_factors_mean,
        noise_factors_log_std,
        factor_precision_log_means,
        factor_precision_log_stds,
    ):

        # single-sample Monte Carlo estimate of the variational lower bound
        mb_size = len(indices)

        def stop_global(globals):
            (
                log_baseline_mean,
                log_baseline_log_std,
                noise_factors_mean,
                noise_factors_log_std,
                factor_precision_log_means,
                factor_precision_log_stds,
            ) = (globals[0], globals[1], globals[2], globals[3], globals[4], globals[5])
            log_baseline_mean = jax.lax.stop_gradient(log_baseline_mean)
            log_baseline_log_std = jax.lax.stop_gradient(log_baseline_log_std)
            noise_factors_mean = jax.lax.stop_gradient(noise_factors_mean)
            noise_factors_log_std = jax.lax.stop_gradient(noise_factors_log_std)
            factor_precision_log_means = jax.lax.stop_gradient(
                factor_precision_log_means
            )
            factor_precision_log_stds = jax.lax.stop_gradient(factor_precision_log_stds)
            return (
                log_baseline_mean,
                log_baseline_log_std,
                noise_factors_mean,
                noise_factors_log_std,
                factor_precision_log_means,
                factor_precision_log_stds,
            )

        def alt_global(globals):
            (
                log_baseline_mean,
                log_baseline_log_std,
                noise_factors_mean,
                noise_factors_log_std,
                factor_precision_log_means,
                factor_precision_log_stds,
            ) = (globals[0], globals[1], globals[2], globals[3], globals[4], globals[5])
            return (
                log_baseline_mean,
                log_baseline_log_std,
                noise_factors_mean,
                noise_factors_log_std,
                factor_precision_log_means,
                factor_precision_log_stds,
            )

        (
            log_baseline_mean,
            log_baseline_log_std,
            noise_factors_mean,
            noise_factors_log_std,
            factor_precision_log_means,
            factor_precision_log_stds,
        ) = jax.lax.cond(
            do_global,
            alt_global,
            stop_global,
            (
                log_baseline_mean,
                log_baseline_log_std,
                noise_factors_mean,
                noise_factors_log_std,
                factor_precision_log_means,
                factor_precision_log_stds,
            ),
        )

        def stop_node_grad(i):
            return (
                jax.lax.stop_gradient(nu_sticks_log_means[i]),
                jax.lax.stop_gradient(nu_sticks_log_stds[i]),
                jax.lax.stop_gradient(psi_sticks_log_means[i]),
                jax.lax.stop_gradient(psi_sticks_log_stds[i]),
                jax.lax.stop_gradient(log_unobserved_factors_kernel_log_stds[i]),
                jax.lax.stop_gradient(log_unobserved_factors_kernel_means[i]),
                jax.lax.stop_gradient(unobserved_log_stds[i]),
                jax.lax.stop_gradient(unobserved_means[i]),
            )

        def stop_node_grads(i):
            return jax.lax.cond(
                node_mask[i] != 1,
                stop_node_grad,
                lambda i: (
                    nu_sticks_log_means[i],
                    nu_sticks_log_stds[i],
                    psi_sticks_log_means[i],
                    psi_sticks_log_stds[i],
                    log_unobserved_factors_kernel_log_stds[i],
                    log_unobserved_factors_kernel_means[i],
                    unobserved_log_stds[i],
                    unobserved_means[i],
                ),
                i,
            )  # Sample all

        (
            nu_sticks_log_means,
            nu_sticks_log_stds,
            psi_sticks_log_means,
            psi_sticks_log_stds,
            log_unobserved_factors_kernel_log_stds,
            log_unobserved_factors_kernel_means,
            unobserved_log_stds,
            unobserved_means,
        ) = vmap(stop_node_grads)(jnp.arange(len(cnvs)))

        def stop_node_params_grads(locals):
            return (
                jax.lax.stop_gradient(log_unobserved_factors_kernel_log_stds),
                jax.lax.stop_gradient(log_unobserved_factors_kernel_means),
                jax.lax.stop_gradient(unobserved_log_stds),
                jax.lax.stop_gradient(unobserved_means),
            )

        def alt_node_params(locals):
            return (
                log_unobserved_factors_kernel_log_stds,
                log_unobserved_factors_kernel_means,
                unobserved_log_stds,
                unobserved_means,
            )

        (
            log_unobserved_factors_kernel_log_stds,
            log_unobserved_factors_kernel_means,
            unobserved_log_stds,
            unobserved_means,
        ) = jax.lax.cond(
            sticks_only,
            stop_node_params_grads,
            alt_node_params,
            (
                log_unobserved_factors_kernel_log_stds,
                log_unobserved_factors_kernel_means,
                unobserved_log_stds,
                unobserved_means,
            ),
        )

        def stop_non_global(locals):
            (
                log_unobserved_factors_kernel_log_stds,
                log_unobserved_factors_kernel_means,
                unobserved_log_stds,
                unobserved_means,
                psi_sticks_log_stds,
                psi_sticks_log_means,
                nu_sticks_log_stds,
                nu_sticks_log_means,
            ) = (
                locals[0],
                locals[1],
                locals[2],
                locals[3],
                locals[4],
                locals[5],
                locals[6],
                locals[7],
            )
            log_unobserved_factors_kernel_log_stds = jax.lax.stop_gradient(
                log_unobserved_factors_kernel_log_stds
            )
            log_unobserved_factors_kernel_means = jax.lax.stop_gradient(
                log_unobserved_factors_kernel_means
            )
            unobserved_log_stds = jax.lax.stop_gradient(unobserved_log_stds)
            unobserved_means = jax.lax.stop_gradient(unobserved_means)
            psi_sticks_log_stds = jax.lax.stop_gradient(psi_sticks_log_stds)
            psi_sticks_log_means = jax.lax.stop_gradient(psi_sticks_log_means)
            nu_sticks_log_stds = jax.lax.stop_gradient(nu_sticks_log_stds)
            nu_sticks_log_means = jax.lax.stop_gradient(nu_sticks_log_means)
            return (
                log_unobserved_factors_kernel_log_stds,
                log_unobserved_factors_kernel_means,
                unobserved_log_stds,
                unobserved_means,
                psi_sticks_log_stds,
                psi_sticks_log_means,
                nu_sticks_log_stds,
                nu_sticks_log_means,
            )

        def alt_non_global(locals):
            (
                log_unobserved_factors_kernel_log_stds,
                log_unobserved_factors_kernel_means,
                unobserved_log_stds,
                unobserved_means,
                psi_sticks_log_stds,
                psi_sticks_log_means,
                nu_sticks_log_stds,
                nu_sticks_log_means,
            ) = (
                locals[0],
                locals[1],
                locals[2],
                locals[3],
                locals[4],
                locals[5],
                locals[6],
                locals[7],
            )
            return (
                log_unobserved_factors_kernel_log_stds,
                log_unobserved_factors_kernel_means,
                unobserved_log_stds,
                unobserved_means,
                psi_sticks_log_stds,
                psi_sticks_log_means,
                nu_sticks_log_stds,
                nu_sticks_log_means,
            )

        (
            log_unobserved_factors_kernel_log_stds,
            log_unobserved_factors_kernel_means,
            unobserved_log_stds,
            unobserved_means,
            psi_sticks_log_stds,
            psi_sticks_log_means,
            nu_sticks_log_stds,
            nu_sticks_log_means,
        ) = jax.lax.cond(
            global_only,
            stop_non_global,
            alt_non_global,
            (
                log_unobserved_factors_kernel_log_stds,
                log_unobserved_factors_kernel_means,
                unobserved_log_stds,
                unobserved_means,
                psi_sticks_log_stds,
                psi_sticks_log_means,
                nu_sticks_log_stds,
                nu_sticks_log_means,
            ),
        )

        def keep_cell_grad(i):
            return cell_noise_mean[indices][i], cell_noise_log_std[indices][i]

        def stop_cell_grad(i):
            return jax.lax.stop_gradient(
                cell_noise_mean[indices][i]
            ), jax.lax.stop_gradient(cell_noise_log_std[indices][i])

        def stop_cell_grads(i):
            return jax.lax.cond(
                data_mask_subset[i] != 1, stop_cell_grad, keep_cell_grad, i
            )

        cell_noise_mean, cell_noise_log_std = vmap(stop_cell_grads)(jnp.arange(mb_size))

        def has_children(i):
            return jnp.any(ancestor_nodes_indices.ravel() == i)

        def has_next_branch(i):
            return jnp.any(previous_branches_indices.ravel() == i)

        ones_vec = jnp.ones(cnvs[0].shape)
        zeros_vec = jnp.log(ones_vec)

        log_baseline = diag_gaussian_sample(
            rng, log_baseline_mean, log_baseline_log_std
        )

        # noise
        factor_precision_log_means = jnp.clip(
            factor_precision_log_means, a_min=jnp.log(1e-3), a_max=jnp.log(1e2)
        )
        factor_precision_log_stds = jnp.clip(
            factor_precision_log_stds, a_min=jnp.log(1e-2), a_max=jnp.log(1e2)
        )
        log_factors_precisions = diag_gaussian_sample(
            rng, factor_precision_log_means, factor_precision_log_stds
        )
        noise_factors_mean = jnp.clip(noise_factors_mean, a_min=-10.0, a_max=10.0)
        noise_factors_log_std = jnp.clip(
            noise_factors_log_std, a_min=jnp.log(1e-2), a_max=jnp.log(1e2)
        )
        noise_factors = diag_gaussian_sample(
            rng, noise_factors_mean, noise_factors_log_std
        )
        cell_noise_mean = jnp.clip(cell_noise_mean, a_min=-10.0, a_max=10.0)
        cell_noise_log_std = jnp.clip(
            cell_noise_log_std, a_min=jnp.log(1e-2), a_max=jnp.log(1e2)
        )
        cell_noise = diag_gaussian_sample(rng, cell_noise_mean, cell_noise_log_std)
        noise = jnp.dot(cell_noise, noise_factors)

        log_unobserved_factors_kernel_means = jnp.clip(
            log_unobserved_factors_kernel_means, a_min=jnp.log(1e-6), a_max=jnp.log(1e2)
        )
        log_unobserved_factors_kernel_log_stds = jnp.clip(
            log_unobserved_factors_kernel_log_stds,
            a_min=jnp.log(1e-6),
            a_max=jnp.log(1e2),
        )

        def sample_unobs_kernel(i):
            return jnp.clip(
                diag_gaussian_sample(
                    rng,
                    log_unobserved_factors_kernel_means[i],
                    log_unobserved_factors_kernel_log_stds[i],
                ),
                a_min=jnp.log(1e-6),
            )

        def sample_all_unobs_kernel(i):
            return jax.lax.cond(
                node_mask[i] >= 0, sample_unobs_kernel, lambda i: zeros_vec, i
            )

        nodes_log_unobserved_factors_kernels = vmap(sample_all_unobs_kernel)(
            jnp.arange(len(cnvs))
        )

        unobserved_means = jnp.clip(unobserved_means, a_min=jnp.log(1e-3), a_max=10)
        unobserved_log_stds = jnp.clip(
            unobserved_log_stds, a_min=jnp.log(1e-6), a_max=jnp.log(1e2)
        )

        def sample_unobs(i):
            return diag_gaussian_sample(
                rng, unobserved_means[i], unobserved_log_stds[i]
            )

        def sample_all_unobs(i):
            return jax.lax.cond(node_mask[i] >= 0, sample_unobs, lambda i: zeros_vec, i)

        nodes_unobserved_factors = vmap(sample_all_unobs)(jnp.arange(len(cnvs)))

        nu_sticks_log_means = jnp.clip(nu_sticks_log_means, -4.0, 4.0)
        nu_sticks_log_stds = jnp.clip(nu_sticks_log_stds, -4.0, 4.0)

        def sample_nu(i):
            return jnp.clip(
                diag_gaussian_sample(
                    rng, nu_sticks_log_means[i], nu_sticks_log_stds[i]
                ),
                -4.0,
                4.0,
            )

        def sample_valid_nu(i):
            return jax.lax.cond(
                has_children(i), sample_nu, lambda i: jnp.array([logit(1 - 1e-6)]), i
            )  # Sample all valid

        def sample_all_nus(i):
            return jax.lax.cond(
                cnvs[i][0] >= 0, sample_valid_nu, lambda i: jnp.array([logit(1e-6)]), i
            )  # Sample all

        log_nu_sticks = vmap(sample_all_nus)(jnp.arange(len(cnvs)))

        def sigmoid_nus(i):
            return jax.lax.cond(
                cnvs[i][0] >= 0,
                lambda i: jnn.sigmoid(log_nu_sticks[i]),
                lambda i: jnp.array([1e-6]),
                i,
            )  # Sample all

        nu_sticks = jnp.clip(vmap(sigmoid_nus)(jnp.arange(len(cnvs))), 1e-6, 1 - 1e-6)

        psi_sticks_log_means = jnp.clip(psi_sticks_log_means, -4.0, 4.0)
        psi_sticks_log_stds = jnp.clip(psi_sticks_log_stds, -4.0, 4.0)

        def sample_psi(i):
            return jnp.clip(
                diag_gaussian_sample(
                    rng, psi_sticks_log_means[i], psi_sticks_log_stds[i]
                ),
                -4.0,
                4.0,
            )

        def sample_valid_psis(i):
            return jax.lax.cond(
                has_next_branch(i),
                sample_psi,
                lambda i: jnp.array([logit(1 - 1e-6)]),
                i,
            )  # Sample all valid

        def sample_all_psis(i):
            return jax.lax.cond(
                cnvs[i][0] >= 0,
                sample_valid_psis,
                lambda i: jnp.array([logit(1e-6)]),
                i,
            )  # Sample all

        log_psi_sticks = vmap(sample_all_psis)(jnp.arange(len(cnvs)))

        def sigmoid_psis(i):
            return jax.lax.cond(
                cnvs[i][0] >= 0,
                lambda i: jnn.sigmoid(log_psi_sticks[i]),
                lambda i: jnp.array([1e-6]),
                i,
            )  # Sample all

        psi_sticks = jnp.clip(vmap(sigmoid_psis)(jnp.arange(len(cnvs))), 1e-6, 1 - 1e-6)

        lib_sizes = jnp.array(self.lib_sizes)[indices]
        data = jnp.array(self.full_data)[indices]
        baseline = jnp.exp(jnp.append(0, log_baseline))

        def compute_node_ll(i):
            unobserved_factors = nodes_unobserved_factors[i] * (parent_vector[i] != -1)

            node_mean = baseline * cnvs[i] / 2 * jnp.exp(unobserved_factors + noise)
            sum = jnp.sum(node_mean, axis=1).reshape(mb_size, 1)
            node_mean = node_mean / sum
            node_mean = node_mean * lib_sizes
            pll = vmap(jax.scipy.stats.poisson.logpmf)(data, node_mean)
            ll = jnp.sum(pll, axis=1)  # N-vector

            # TSSB prior
            nu_stick = nu_sticks[i]
            psi_stick = psi_sticks[i]

            def prev_branches_psi(idx):
                return (idx != -1) * jnp.log(1.0 - psi_sticks[idx])

            def ancestors_nu(idx):
                _log_phi = jnp.log(psi_sticks[idx]) + jnp.sum(
                    vmap(prev_branches_psi)(previous_branches_indices[idx])
                )
                _log_1_nu = jnp.log(1.0 - nu_sticks[idx])
                total = _log_phi + _log_1_nu
                return (idx != -1) * total

            log_phi = jnp.log(psi_stick) + jnp.sum(
                vmap(prev_branches_psi)(previous_branches_indices[i])
            )
            log_node_weight = (
                jnp.log(nu_stick)
                + log_phi
                + jnp.sum(vmap(ancestors_nu)(ancestor_nodes_indices[i]))
            )
            log_node_weight = log_node_weight + jnp.log(tssb_weights[i])
            ll = ll + log_node_weight  # N-vector

            return ll

        small_ll = -1e10 * jnp.ones((mb_size))

        def get_node_ll(i):
            return jnp.where(
                node_mask[i] == 1,
                compute_node_ll(jnp.where(node_mask[i] == 1, i, 0)),
                small_ll,
            )

        out = jnp.array(vmap(get_node_ll)(jnp.arange(len(parent_vector))))
        l = jnp.sum(jnn.logsumexp(out, axis=0) * data_mask_subset)

        log_rate = jnp.log(self.unobserved_factors_kernel_rate)
        log_concentration = jnp.log(self.unobserved_factors_kernel_concentration)
        log_kernel = jnp.log(self.unobserved_factors_root_kernel)
        broadcasted_concentration = log_concentration * ones_vec
        broadcasted_rate = log_rate * ones_vec

        def compute_node_kl(i):
            kl = 0.0
            # # unobserved_factors_kernel -- USING JUST A SPARSE GAMMA DOES NOT PENALIZE KERNELS EQUAL TO ZERO
            pl = diag_gamma_logpdf(
                jnp.clip(jnp.exp(nodes_log_unobserved_factors_kernels[i]), a_min=1e-6),
                broadcasted_concentration,
                log_rate
                + (parent_vector[i] != -1)
                * (parent_vector[i] != 0)
                * (jnp.abs(nodes_unobserved_factors[parent_vector[i]])),
            )
            ent = -diag_loggaussian_logpdf(
                jnp.clip(jnp.exp(nodes_log_unobserved_factors_kernels[i]), a_min=1e-6),
                log_unobserved_factors_kernel_means[i],
                log_unobserved_factors_kernel_log_stds[i],
            )
            kl += (parent_vector[i] != -1) * (pl + ent)

            # # Penalize copies in unobserved nodes
            # pl = diag_gamma_logpdf(1e-6 * jnp.ones(broadcasted_concentration.shape), broadcasted_concentration,
            #                         (parent_vector[i] != -1)*(log_rate + jnp.abs(nodes_unobserved_factors[parent_vector[i]])))
            # ent = - diag_gaussian_logpdf(jnp.log(1e-6 * jnp.ones(broadcasted_concentration.shape)), log_unobserved_factors_kernel_means[i], log_unobserved_factors_kernel_log_stds[i])
            # kl -= (parent_vector[i] != -1) * jnp.all(tssb_indices[i] == tssb_indices[parent_vector[i]]) * (pl + 0*ent)

            # unobserved_factors
            is_root_subtree = jnp.all(tssb_indices[i] == tssb_indices[0])
            pl = diag_gaussian_logpdf(
                nodes_unobserved_factors[i],
                (parent_vector[i] != -1)
                * (parent_vector[i] != 0)
                * nodes_unobserved_factors[parent_vector[i]]
                + (parent_vector[i] != -1)
                * is_root_subtree
                * (jnp.exp(nodes_log_unobserved_factors_kernels[i]) > 0.1)
                * 0.2,
                jnp.clip(nodes_log_unobserved_factors_kernels[i], a_min=jnp.log(1e-6))
                * (parent_vector[i] != -1),
            )
            ent = -diag_gaussian_logpdf(
                nodes_unobserved_factors[i], unobserved_means[i], unobserved_log_stds[i]
            )
            kl += (parent_vector[i] != -1) * (pl + ent)

            # # Penalize copied unobserved_factors
            # pl = diag_gaussian_logpdf(nodes_unobserved_factors[parent_vector[i]],
            #             (parent_vector[i] != -1) * nodes_unobserved_factors[parent_vector[i]],
            #             jnp.clip(nodes_log_unobserved_factors_kernels[i], a_min=jnp.log(1e-6))*(parent_vector[i] != -1) + log_kernel*(parent_vector[i] == -1))
            # ent = - diag_gaussian_logpdf(nodes_unobserved_factors[parent_vector[i]], unobserved_means[i], unobserved_log_stds[i])
            # kl -= (parent_vector[i] != -1) * jnp.all(tssb_indices[i] == tssb_indices[parent_vector[i]]) * (pl + 0*ent)

            # sticks
            nu_pl = has_children(i) * beta_logpdf(
                nu_sticks[i],
                jnp.log(jnp.array([1.0])),
                jnp.log(jnp.array([dp_alphas[i]])),
            )
            nu_ent = has_children(i) * -diag_gaussian_logpdf(
                log_nu_sticks[i], nu_sticks_log_means[i], nu_sticks_log_stds[i]
            )
            kl += nu_pl + nu_ent

            psi_pl = has_next_branch(i) * beta_logpdf(
                psi_sticks[i],
                jnp.log(jnp.array([1.0])),
                jnp.log(jnp.array([dp_gammas[i]])),
            )
            psi_ent = has_next_branch(i) * -diag_gaussian_logpdf(
                log_psi_sticks[i], psi_sticks_log_means[i], psi_sticks_log_stds[i]
            )
            kl += psi_pl + psi_ent

            return kl

        def get_node_kl(i):
            return jnp.where(
                node_mask[i] == 1,
                compute_node_kl(jnp.where(node_mask[i] == 1, i, 0)),
                0.0,
            )

        node_kls = vmap(get_node_kl)(jnp.arange(len(parent_vector)))
        node_kl = jnp.sum(node_kls)

        # Global vars KL
        baseline_kl = diag_gaussian_logpdf(
            log_baseline, zeros_vec[1:], zeros_vec[1:]
        ) - diag_gaussian_logpdf(log_baseline, log_baseline_mean, log_baseline_log_std)
        ones_mat = jnp.ones(log_factors_precisions.shape)
        zeros_mat = jnp.zeros(log_factors_precisions.shape)
        factor_precision_kl = diag_gamma_logpdf(
            jnp.exp(log_factors_precisions),
            jnp.log(self.global_noise_factors_precisions_shape) * ones_mat,
            zeros_mat,
        ) - diag_loggaussian_logpdf(
            log_factors_precisions,
            factor_precision_log_means,
            factor_precision_log_stds,
        )
        noise_factors_kl = diag_gaussian_logpdf(
            noise_factors,
            jnp.zeros(noise_factors.shape),
            jnp.log(jnp.sqrt(1.0 / jnp.exp(log_factors_precisions)).reshape(-1, 1))
            * jnp.ones(noise_factors.shape),
        ) - diag_gaussian_logpdf(
            noise_factors, noise_factors_mean, noise_factors_log_std
        )
        total_kl = node_kl + baseline_kl + factor_precision_kl + noise_factors_kl

        # Scale the KL by the data size
        total_kl = total_kl * jnp.sum(data_mask_subset != 0) / self.tssb.ntssb.num_data

        zeros_mat = jnp.zeros(cell_noise.shape)
        cell_noise_kl = diag_gaussian_logpdf(
            cell_noise, zeros_mat, zeros_mat, axis=1
        ) - diag_gaussian_logpdf(
            cell_noise, cell_noise_mean, cell_noise_log_std, axis=1
        )
        cell_noise_kl = jnp.sum(cell_noise_kl * data_mask_subset)
        total_kl = total_kl + cell_noise_kl

        elbo_val = l + total_kl

        return elbo_val, l, total_kl, node_kls

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
