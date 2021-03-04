from numpy        import *
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

from ...util import *
from ...ntssb.node import *
from ...ntssb.tree import *

class Node(AbstractNode):
    def __init__(self, is_observed, observed_parameters, log_lib_size_mean=2, log_lib_size_std=1,
                        num_global_noise_factors=4, global_noise_factors_scale=1.,
                        cell_global_noise_factors_weights_scale=1.,
                        unobserved_factors_root_kernel=0.1, unobserved_factors_kernel=1.,
                        unobserved_factors_kernel_concentration=0.1, **kwargs):
        super(Node, self).__init__(is_observed, observed_parameters, **kwargs)

        # The observed parameters are the CNVs of all genes
        self.cnvs = self.observed_parameters
        self.n_genes = self.cnvs.size

        # Node hyperparameters
        if self.parent() is None:
            self.node_hyperparams = dict(
                log_lib_size_mean=log_lib_size_mean, log_lib_size_std=log_lib_size_std,
                num_global_noise_factors=num_global_noise_factors,
                global_noise_factors_scale=global_noise_factors_scale,
                cell_global_noise_factors_weights_scale=cell_global_noise_factors_weights_scale,
                unobserved_factors_root_kernel=unobserved_factors_root_kernel,
                unobserved_factors_kernel=unobserved_factors_kernel,
                unobserved_factors_kernel_concentration=unobserved_factors_kernel_concentration,
            )
        else:
            self.node_hyperparams = self.node_hyperparams_caller()

        self.reset_parameters(**self.node_hyperparams)

    def get_node_mean(self, log_baseline, unobserved_factors, noise, cnvs):
        node_mean = jnp.exp(log_baseline + unobserved_factors + noise + jnp.log(cnvs/2))
        sum = jnp.sum(node_mean, axis=1).reshape(self.tssb.ntssb.num_data, 1)
        node_mean = node_mean / sum
        return node_mean

    def reset_variational_parameters(self):
        if self.parent() is None:
            # Baseline: first value is 1
            self.variational_parameters['globals']['log_baseline_mean'] = np.array(normal_sample(0, 1, self.n_genes-1)) #np.zeros((self.n_genes-1,))
            if self.tssb.ntssb.data is not None:
                self.full_data = jnp.array(self.tssb.ntssb.data)
                init_baseline = np.mean(self.tssb.ntssb.data, axis=0)
                init_log_baseline = np.log(init_baseline / init_baseline[0])[1:]
                self.variational_parameters['globals']['log_baseline_mean'] = init_log_baseline
            self.variational_parameters['globals']['log_baseline_log_std'] = np.array(np.zeros((self.n_genes-1,)))

            # Overdispersion
            # self.log_od_mean = np.zeros(1)
            # self.log_od_log_std = np.zeros(1)

            if self.tssb.ntssb.num_data is not None:
                # Noise
                self.variational_parameters['globals']['cell_noise_mean'] = np.zeros((self.tssb.ntssb.num_data, self.num_global_noise_factors))
                self.variational_parameters['globals']['cell_noise_log_std'] = np.zeros((self.tssb.ntssb.num_data, self.num_global_noise_factors))
                self.variational_parameters['globals']['noise_factors_mean'] = np.zeros((self.num_global_noise_factors, self.n_genes))
                self.variational_parameters['globals']['noise_factors_log_std'] = np.zeros((self.num_global_noise_factors, self.n_genes))
                self.variational_parameters['globals']['factor_precision_log_means'] = np.zeros((self.num_global_noise_factors))
                self.variational_parameters['globals']['factor_precision_log_stds'] = np.zeros((self.num_global_noise_factors))

        self.data_ass_logits = np.zeros((self.tssb.ntssb.num_data))

        # Sticks
        self.variational_parameters['locals']['nu_log_alpha'] = np.array(-1.)
        self.variational_parameters['locals']['nu_log_beta'] = np.array(np.log(1. * self.tssb.dp_alpha))
        self.variational_parameters['locals']['psi_log_alpha'] = np.array(-1)
        self.variational_parameters['locals']['psi_log_beta'] = np.array(np.log(1. * self.tssb.dp_alpha))

        # Unobserved factors
        self.variational_parameters['locals']['unobserved_factors_mean'] = np.zeros((self.n_genes,))
        self.variational_parameters['locals']['unobserved_factors_log_std'] = -2.*np.ones((self.n_genes,))
        self.variational_parameters['locals']['unobserved_factors_kernel_log_mean'] = -1.*np.ones((self.n_genes,))
        self.variational_parameters['locals']['unobserved_factors_kernel_log_std'] = -np.ones((self.n_genes,))

        self.set_mean(self.get_mean(baseline=np.append(1, np.exp(self.log_baseline_caller())), unobserved_factors=self.variational_parameters['locals']['unobserved_factors_mean']))

    # ========= Functions to initialize node. =========
    def reset_data_parameters(self):
        self.full_data = jnp.array(self.tssb.ntssb.data)
        self.lib_sizes = np.sum(self.tssb.ntssb.data, axis=1).reshape(self.tssb.ntssb.num_data, 1)
        self.cell_global_noise_factors_weights = normal_sample(0, self.cell_global_noise_factors_weights_scale,
                                                    size=[self.tssb.ntssb.num_data, self.num_global_noise_factors])

    def generate_data_params(self):
        self.lib_sizes = np.exp(normal_sample(self.log_lib_size_mean, self.log_lib_size_std, size=self.tssb.ntssb.num_data)).reshape(self.tssb.ntssb.num_data, 1)
        self.lib_sizes = np.ceil(self.lib_sizes)
        self.cell_global_noise_factors_weights = normal_sample(0, self.cell_global_noise_factors_weights_scale,
                                                    size=[self.tssb.ntssb.num_data, self.num_global_noise_factors])

    def reset_parameters(self, root_params=True, down_params=True,
                        log_lib_size_mean=2, log_lib_size_std=1,
                        num_global_noise_factors=4, global_noise_factors_scale=1.,
                        cell_global_noise_factors_weights_scale=1.,
                        unobserved_factors_root_kernel=0.1, unobserved_factors_kernel=1.,
                        unobserved_factors_kernel_concentration=.1):
        parent = self.parent()

        if parent is None: # this is the root
            self.node_hyperparams = dict(
                log_lib_size_mean=log_lib_size_mean, log_lib_size_std=log_lib_size_std,
                num_global_noise_factors=num_global_noise_factors,
                global_noise_factors_scale=global_noise_factors_scale,
                cell_global_noise_factors_weights_scale=cell_global_noise_factors_weights_scale,
                unobserved_factors_root_kernel=unobserved_factors_root_kernel,
                unobserved_factors_kernel=unobserved_factors_kernel,
                unobserved_factors_kernel_concentration=unobserved_factors_kernel_concentration,
            )

            if root_params:
                # The root is used to store global parameters:  mu
                self.baseline = np.exp(normal_sample(0, .1, size=self.n_genes))
                self.baseline[0] = 1.
                self.overdispersion = np.exp(normal_sample(0, 1))
                self.log_lib_size_mean = log_lib_size_mean
                self.log_lib_size_std = log_lib_size_std

                # Structured noise: keep all cells' noise factors at the root
                self.num_global_noise_factors = num_global_noise_factors
                self.global_noise_factors_scale = global_noise_factors_scale
                self.cell_global_noise_factors_weights_scale = cell_global_noise_factors_weights_scale

                self.global_noise_factors_precisions = gamma_sample(5., 1., size=self.num_global_noise_factors)
                self.global_noise_factors = normal_sample(0, 1./np.sqrt(self.global_noise_factors_precisions), size=[self.n_genes, self.num_global_noise_factors]).T # K x P

                self.unobserved_factors_root_kernel = unobserved_factors_root_kernel
                self.unobserved_factors_kernel_concentration = unobserved_factors_kernel_concentration

            self.unobserved_factors_kernel = np.array([self.unobserved_factors_root_kernel] * self.n_genes)
            self.unobserved_factors = normal_sample(0., self.unobserved_factors_kernel)

            self.set_mean()

            self.depth  = 0.0
        else: # Non-root node: inherits everything from upstream node
            self.node_hyperparams = self.node_hyperparams_caller()
            if down_params:
                self.unobserved_factors_kernel = gamma_sample(self.unobserved_factors_kernel_concentration_caller(), np.exp(1*np.abs(parent.unobserved_factors)), size=self.n_genes)
                self.unobserved_factors = normal_sample(parent.unobserved_factors, self.unobserved_factors_kernel)

            # Observation mean
            self.set_mean()

            self.depth = parent.depth + 1

    def set_mean(self, node_mean=None, variational=False):
        if node_mean is not None:
            self.node_mean = node_mean
        else:
            if variational:
                self.node_mean = np.append(1, np.exp(self.log_baseline_caller())) * self.cnvs/2 * np.exp(self.variational_parameters['locals']['unobserved_factors_mean'])
            else:
                self.node_mean = self.baseline_caller() * self.cnvs/2 * np.exp(self.unobserved_factors)
            self.node_mean = self.node_mean / np.sum(self.node_mean)

    def get_mean(self, baseline=None, unobserved_factors=None, noise=None, cell_factors=None, global_factors=None, cnvs=None, norm=True):
        baseline = np.append(1, np.exp(self.log_baseline_caller())) if baseline is None else baseline
        unobserved_factors = self.variational_parameters['locals']['unobserved_factors_mean'] if unobserved_factors is None else unobserved_factors
        cnvs = self.cnvs if cnvs is None else cnvs
        node_mean = None
        if noise is not None:
            node_mean = baseline * cnvs/2 * np.exp(unobserved_factors + noise)
        else:
            node_mean = baseline * cnvs/2 * np.exp(unobserved_factors)
        if norm:
            if len(node_mean.shape) == 1:
                sum = np.sum(node_mean)
            else:
                sum = np.sum(node_mean, axis=1)
            if len(sum.shape) > 0:
                sum = sum.reshape(self.tssb.ntssb.num_data, 1)
            node_mean = node_mean / sum
        return node_mean

    # ========= Functions to take samples from node. =========
    def sample_observation(self, n):
        noise = self.cell_global_noise_factors_weights_caller()[n].dot(self.global_noise_factors_caller())
        node_mean = self.get_mean(unobserved_factors=self.unobserved_factors, baseline=self.baseline_caller(), noise=noise)
        s = multinomial_sample(self.lib_sizes_caller()[n], node_mean)
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
        return normal_lpdf(self.global_noise_factors, 0., 1./np.sqrt(self.global_noise_factors_precisions))

    def cell_global_noise_factors_logprior(self):
        return normal_lpdf(self.cell_global_noise_factors_weights, 0., self.cell_global_noise_factors_weights_scale)

    def unobserved_factors_logprior(self, x=None):
        if x is None:
            x = self.unobserved_factors

        if self.parent() is not None:
            if self.is_observed:
                llp = normal_lpdf(x, self.parent().unobserved_factors, self.unobserved_factors_root_kernel)
            else:
                llp = normal_lpdf(x, self.parent().unobserved_factors, self.unobserved_factors_kernel)
        else:
            llp = normal_lpdf(x, 0., self.unobserved_factors_root_kernel)

        return llp

    def logprior(self):
        # Prior of current node
        llp = self.unobserved_factors_logprior()
        if self.parent() is None:
            llp = llp + self.global_noise_factors_logprior() + self.cell_global_noise_factors_logprior()
            llp = llp + self.log_baseline_logprior()

        return llp

    def loglh(self, n, variational=False, axis=None):
        noise = self.cell_global_noise_factors_weights_caller()[n].dot(self.global_noise_factors_caller())
        node_mean = self.get_mean(baseline=np.append(1, np.exp(self.log_baseline_caller())), unobserved_factors=self.variational_parameters['locals']['unobserved_factors_mean'], noise=noise)
        llh = poisson_lpmf(self.tssb.ntssb.data[n],  self.lib_sizes_caller()[n] * node_mean, axis=axis)
        return llh

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

    # @partial(jit, static_argnums=(0,1,2,3,4,5))
    def compute_elbo(self, rng, cnvs, parent_vector, children_vector, ancestor_nodes_indices, tssb_indices, previous_branches_indices, tssb_weights, dp_alphas, dp_gammas, node_mask, data_mask_subset, indices, do_global, global_only, sticks_only,
                nu_sticks_log_alphas, nu_sticks_log_betas,
                psi_sticks_log_alphas, psi_sticks_log_betas, unobserved_means, unobserved_log_stds, log_unobserved_factors_kernel_means, log_unobserved_factors_kernel_log_stds,
                log_baseline_mean, log_baseline_log_std, cell_noise_mean, cell_noise_log_std, noise_factors_mean, noise_factors_log_std, factor_precision_log_means, factor_precision_log_stds):

        # single-sample Monte Carlo estimate of the variational lower bound
        mb_size = len(indices)

        def stop_global(globals):
            log_baseline_mean, log_baseline_log_std, noise_factors_mean, noise_factors_log_std, factor_precision_log_means, factor_precision_log_stds = globals[0], globals[1], globals[2], globals[3], globals[4], globals[5]
            log_baseline_mean = jax.lax.stop_gradient(log_baseline_mean)
            log_baseline_log_std = jax.lax.stop_gradient(log_baseline_log_std)
            noise_factors_mean = jax.lax.stop_gradient(noise_factors_mean)
            noise_factors_log_std = jax.lax.stop_gradient(noise_factors_log_std)
            factor_precision_log_means = jax.lax.stop_gradient(factor_precision_log_means)
            factor_precision_log_stds = jax.lax.stop_gradient(factor_precision_log_stds)
            return log_baseline_mean, log_baseline_log_std, noise_factors_mean, noise_factors_log_std, factor_precision_log_means, factor_precision_log_stds

        def alt_global(globals):
            log_baseline_mean, log_baseline_log_std, noise_factors_mean, noise_factors_log_std, factor_precision_log_means, factor_precision_log_stds = globals[0], globals[1], globals[2], globals[3], globals[4], globals[5]
            return log_baseline_mean, log_baseline_log_std, noise_factors_mean, noise_factors_log_std, factor_precision_log_means, factor_precision_log_stds

        log_baseline_mean, log_baseline_log_std, noise_factors_mean, noise_factors_log_std, factor_precision_log_means, factor_precision_log_stds = jax.lax.cond(do_global, alt_global, stop_global, (log_baseline_mean, log_baseline_log_std, noise_factors_mean, noise_factors_log_std, factor_precision_log_means, factor_precision_log_stds))

        def stop_node_grad(i):
            return (jax.lax.stop_gradient(nu_sticks_log_alphas[i]), jax.lax.stop_gradient(nu_sticks_log_betas[i]),
                    jax.lax.stop_gradient(psi_sticks_log_alphas[i]), jax.lax.stop_gradient(psi_sticks_log_betas[i]),
                    jax.lax.stop_gradient(log_unobserved_factors_kernel_log_stds[i]), jax.lax.stop_gradient(log_unobserved_factors_kernel_means[i]),
                    jax.lax.stop_gradient(unobserved_log_stds[i]), jax.lax.stop_gradient(unobserved_means[i]) )
        def stop_node_grads(i):
            return jax.lax.cond(node_mask[i] != 1, stop_node_grad, lambda i: (nu_sticks_log_alphas[i], nu_sticks_log_betas[i],
                                                                              psi_sticks_log_alphas[i], psi_sticks_log_betas[i],
                                                                              log_unobserved_factors_kernel_log_stds[i], log_unobserved_factors_kernel_means[i],
                                                                              unobserved_log_stds[i], unobserved_means[i]), i) # Sample all
        nu_sticks_log_alphas, nu_sticks_log_betas, psi_sticks_log_alphas, psi_sticks_log_betas, log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means = vmap(stop_node_grads)(jnp.arange(len(cnvs)))

        def stop_node_params_grads(locals):
            return  (jax.lax.stop_gradient(log_unobserved_factors_kernel_log_stds), jax.lax.stop_gradient(log_unobserved_factors_kernel_means),
                    jax.lax.stop_gradient(unobserved_log_stds), jax.lax.stop_gradient(unobserved_means) )
        def alt_node_params(locals):
            return (log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means)
        log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means = jax.lax.cond(sticks_only, stop_node_params_grads, alt_node_params,
            (log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means))

        def stop_non_global(locals):
            log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means, psi_sticks_log_betas, psi_sticks_log_alphas, nu_sticks_log_betas, nu_sticks_log_alphas = locals[0], locals[1], locals[2], locals[3], locals[4], locals[5], locals[6], locals[7]
            log_unobserved_factors_kernel_log_stds = jax.lax.stop_gradient(log_unobserved_factors_kernel_log_stds)
            log_unobserved_factors_kernel_means = jax.lax.stop_gradient(log_unobserved_factors_kernel_means)
            unobserved_log_stds = jax.lax.stop_gradient(unobserved_log_stds)
            unobserved_means = jax.lax.stop_gradient(unobserved_means)
            psi_sticks_log_betas = jax.lax.stop_gradient(psi_sticks_log_betas)
            psi_sticks_log_alphas = jax.lax.stop_gradient(psi_sticks_log_alphas)
            nu_sticks_log_betas = jax.lax.stop_gradient(nu_sticks_log_betas)
            nu_sticks_log_alphas = jax.lax.stop_gradient(nu_sticks_log_alphas)
            return log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means, psi_sticks_log_betas, psi_sticks_log_alphas, nu_sticks_log_betas, nu_sticks_log_alphas

        def alt_non_global(locals):
            log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means, psi_sticks_log_betas, psi_sticks_log_alphas, nu_sticks_log_betas, nu_sticks_log_alphas = locals[0], locals[1], locals[2], locals[3], locals[4], locals[5], locals[6], locals[7]
            return log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means, psi_sticks_log_betas, psi_sticks_log_alphas, nu_sticks_log_betas, nu_sticks_log_alphas

        log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means, psi_sticks_log_betas, psi_sticks_log_alphas, nu_sticks_log_betas, nu_sticks_log_alphas = jax.lax.cond(global_only, stop_non_global, alt_non_global,
            (log_unobserved_factors_kernel_log_stds, log_unobserved_factors_kernel_means, unobserved_log_stds, unobserved_means, psi_sticks_log_betas, psi_sticks_log_alphas, nu_sticks_log_betas, nu_sticks_log_alphas))

        def keep_cell_grad(i):
            return cell_noise_mean[indices][i], cell_noise_log_std[indices][i]
        def stop_cell_grad(i):
            return jax.lax.stop_gradient(cell_noise_mean[indices][i]), jax.lax.stop_gradient(cell_noise_log_std[indices][i])
        def stop_cell_grads(i):
            return jax.lax.cond(data_mask_subset[i] == 1, stop_cell_grad, keep_cell_grad, i)
        cell_noise_mean, cell_noise_log_std = vmap(stop_cell_grads)(jnp.arange(mb_size))

        log_baseline = diag_gaussian_sample(rng, log_baseline_mean, log_baseline_log_std)

        # noise
        log_factors_precisions = diag_gaussian_sample(rng, factor_precision_log_means, factor_precision_log_stds)
        noise_factors = diag_gaussian_sample(rng, noise_factors_mean, noise_factors_log_std)
        cell_noise = diag_gaussian_sample(rng, cell_noise_mean[indices], cell_noise_log_std[indices])
        noise = jnp.dot(cell_noise, noise_factors)

        def sample_unobs_kernel(i):
            return diag_gaussian_sample(rng, log_unobserved_factors_kernel_means[i], log_unobserved_factors_kernel_log_stds[i])
        def sample_all_unobs_kernel(i):
            return jax.lax.cond(node_mask[i] >= 0, sample_unobs_kernel, lambda i: jnp.zeros(cnvs[i].shape), i)
        nodes_log_unobserved_factors_kernels = vmap(sample_all_unobs_kernel)(jnp.arange(len(cnvs)))

        def sample_unobs(i):
            return diag_gaussian_sample(rng, unobserved_means[i], unobserved_log_stds[i])
        def sample_all_unobs(i):
            return jax.lax.cond(node_mask[i] >= 0, sample_unobs, lambda i: jnp.zeros(cnvs[i].shape), i)
        nodes_unobserved_factors = vmap(sample_all_unobs)(jnp.arange(len(cnvs)))

        nu_sticks_log_alphas = jnp.clip(nu_sticks_log_alphas, a_min=-5., a_max=5.)
        nu_sticks_log_betas = jnp.clip(nu_sticks_log_betas, a_min=-5, a_max=5.)
        def sample_nu(i):
            return jnp.clip(beta_sample(rng, nu_sticks_log_alphas[i], nu_sticks_log_betas[i]), 1e-6, 1-1e-6)
        def sample_all_nus(i):
            return jax.lax.cond(cnvs[i][0] >= 0, sample_nu, lambda i: jnp.array([1e-6]), i) # Sample all
        nu_sticks = jnp.clip(vmap(sample_all_nus)(jnp.arange(len(cnvs))), 1e-6, 1-1e-6)

        psi_sticks_log_alphas = jnp.clip(psi_sticks_log_alphas, a_min=-5., a_max=5.)
        psi_sticks_log_betas = jnp.clip(psi_sticks_log_betas, a_min=-5., a_max=5.)
        def sample_psi(i):
            return jnp.clip(beta_sample(rng, psi_sticks_log_alphas[i], psi_sticks_log_betas[i]), 1e-6, 1-1e-6)
        def sample_all_psis(i):
            return jax.lax.cond(cnvs[i][0] >= 0, sample_psi, lambda i: jnp.array([1e-6]), i) # Sample all
        psi_sticks = jnp.clip(vmap(sample_all_psis)(jnp.arange(len(cnvs))), 1e-6, 1-1e-6)

        def compute_node_ll(i):
            unobserved_factors = nodes_unobserved_factors[i]

            node_mean = jnp.exp(jnp.append(0, log_baseline)) * cnvs[i]/2 * jnp.exp(unobserved_factors + noise)
            sum = jnp.sum(node_mean, axis=1).reshape(mb_size, 1)
            node_mean = node_mean / sum
            node_mean = node_mean * (jnp.array(self.lib_sizes)[indices])
            pll = vmap(jax.scipy.stats.poisson.logpmf)(self.full_data[indices], node_mean)
            ll = jnp.sum(pll, axis=1) # N-vector

            # TSSB prior
            nu_stick = nu_sticks[i]
            psi_stick = psi_sticks[i]
            def prev_branches_psi(idx):
                return (idx != -1) * jnp.log(1. - psi_sticks[idx])
            def ancestors_nu(idx):
                _log_phi = jnp.log(psi_sticks[idx]) + jnp.sum(vmap(prev_branches_psi)(previous_branches_indices[idx]))
                _log_1_nu = jnp.log(1. - nu_sticks[idx])
                total = _log_phi + _log_1_nu
                return (idx != -1) * total
            log_phi = jnp.log(psi_stick) + jnp.sum(vmap(prev_branches_psi)(previous_branches_indices[i]))
            log_node_weight = jnp.log(nu_stick) + log_phi + jnp.sum(vmap(ancestors_nu)(ancestor_nodes_indices[i]))
            log_node_weight = log_node_weight + jnp.log(tssb_weights[i])
            ll = ll + log_node_weight # N-vector

            return ll

        def get_node_ll(i):
            return jax.lax.cond(node_mask[i] == 1, lambda _: compute_node_ll(i), lambda _: -jnp.inf*jnp.ones((mb_size)), operand=None)
        out = jnp.array(vmap(get_node_ll)(jnp.arange(len(parent_vector))))
        l = jnp.sum(jnn.logsumexp(out, axis=0) * data_mask_subset)

        def compute_node_kl(i):
            # unobserved_factors_kernel
            pl = diag_gamma_logpdf(jnp.exp(nodes_log_unobserved_factors_kernels[i]), jnp.log(self.unobserved_factors_kernel_concentration) * jnp.ones(cnvs[i].shape), (parent_vector[i] != -1)*jnp.abs(nodes_unobserved_factors[parent_vector[i]]))
            ent = - diag_gaussian_logpdf(nodes_log_unobserved_factors_kernels[i], log_unobserved_factors_kernel_means[i], log_unobserved_factors_kernel_log_stds[i])
            kl = (parent_vector[i] != -1) * (pl + ent)

            # unobserved_factors
            pl = diag_gaussian_logpdf(nodes_unobserved_factors[i],
                        (parent_vector[i] != -1) * nodes_unobserved_factors[parent_vector[i]],
                        nodes_log_unobserved_factors_kernels[i]*(parent_vector[i] != -1) + jnp.log(self.unobserved_factors_root_kernel*jnp.ones(cnvs[i].shape))*(parent_vector[i] == -1))
            ent = - diag_gaussian_logpdf(nodes_unobserved_factors[i], unobserved_means[i], unobserved_log_stds[i])
            kl = kl + pl + ent

            # sticks
            nu_pl = beta_logpdf(nu_sticks[i], jnp.log(jnp.array([1.])), jnp.log(jnp.array([dp_alphas[i]])))
            nu_ent = - beta_logpdf(nu_sticks[i], nu_sticks_log_alphas[i], nu_sticks_log_betas[i])
            kl = kl + nu_pl + nu_ent

            psi_pl = beta_logpdf(psi_sticks[i], jnp.log(jnp.array([1.])), jnp.log(jnp.array([dp_gammas[i]])))
            psi_ent = - beta_logpdf(psi_sticks[i], psi_sticks_log_alphas[i], psi_sticks_log_betas[i])
            kl = kl + psi_pl + psi_ent

            return kl

        def get_node_kl(i):
            return jax.lax.cond(node_mask[i] == 1, lambda _: compute_node_kl(i), lambda _: 0., operand=None)
        node_kl = jnp.sum(vmap(get_node_kl)(jnp.arange(len(parent_vector))))

        # Global vars KL
        baseline_kl = diag_gaussian_logpdf(log_baseline, jnp.zeros(log_baseline.shape), jnp.log(0.5)*jnp.ones(log_baseline.shape)) - diag_gaussian_logpdf(log_baseline, log_baseline_mean, log_baseline_log_std)
        factor_precision_kl = diag_gamma_logpdf(jnp.exp(log_factors_precisions), 2*jnp.ones(log_factors_precisions.shape), jnp.ones(log_factors_precisions.shape)) - diag_gaussian_logpdf(log_factors_precisions, factor_precision_log_means, factor_precision_log_stds)
        noise_factors_kl = diag_gaussian_logpdf(noise_factors, jnp.zeros(noise_factors.shape), jnp.sqrt(jnp.exp(-log_factors_precisions)).reshape(-1,1) * jnp.ones(noise_factors.shape)) - diag_gaussian_logpdf(noise_factors, noise_factors_mean, noise_factors_log_std)
        total_kl = node_kl + baseline_kl + factor_precision_kl + noise_factors_kl

        # Scale the KL by the data size
        total_kl = total_kl * jnp.sum(data_mask_subset != 0) / self.tssb.ntssb.num_data

        cell_noise_kl = diag_gaussian_logpdf(cell_noise, jnp.zeros(cell_noise.shape), jnp.zeros(cell_noise.shape), axis=1) - diag_gaussian_logpdf(cell_noise, cell_noise_mean, cell_noise_log_std, axis=1)
        cell_noise_kl = jnp.sum(cell_noise_kl * data_mask_subset)
        total_kl = total_kl + cell_noise_kl

        elbo_val = l + total_kl

        # Scale by minibatch
        elbo_val = elbo_val # * self.tssb.ntssb.num_data / jnp.sum(data_mask_subset != 0)
        return elbo_val

    # ========= Functions to acess root's parameters. =========
    def node_hyperparams_caller(self):
        if self.parent() is None:
            return self.node_hyperparams
        else:
            return self.parent().node_hyperparams_caller()

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

    def log_baseline_caller(self):
        if self.parent() is None:
            return self.variational_parameters['globals']['log_baseline_mean']
        else:
            return self.parent().log_baseline_caller()

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

    def cell_global_noise_factors_weights_caller(self):
        if self.parent() is None:
            return self.cell_global_noise_factors_weights
        else:
            return self.parent().cell_global_noise_factors_weights_caller()

    def global_noise_factors_caller(self):
        if self.parent() is None:
            return self.global_noise_factors
        else:
            return self.parent().global_noise_factors_caller()

    def set_event_string(self, var_names=None, estimated=True, unobs_threshold=1., kernel_threshold=1., max_len=5, event_fontsize=14):
        if var_names is None:
            var_names = np.arange(self.n_genes).astype(int).astype(str)

        unobserved_factors = self.unobserved_factors
        unobserved_factors_kernel = self.unobserved_factors_kernel
        if estimated:
            unobserved_factors = self.variational_parameters['locals']['unobserved_factors_mean']
            unobserved_factors_kernel = np.exp(self.variational_parameters['locals']['unobserved_factors_kernel_log_mean'])

        # Up-regulated
        up_color = 'red'
        up_list = np.where(np.logical_and(unobserved_factors > unobs_threshold, unobserved_factors_kernel > kernel_threshold))[0]
        sorted_idx = np.argsort(unobserved_factors[up_list])[:max_len]
        up_list = up_list[sorted_idx]
        up_str = ''
        if len(up_list) > 0:
            up_str = f'<font point-size="{event_fontsize}" color="{up_color}">+</font>' + ','.join(var_names[up_list])

        # Down-regulated
        down_color = 'blue'
        down_list = np.where(np.logical_and(unobserved_factors < -unobs_threshold, unobserved_factors_kernel > kernel_threshold))[0]
        sorted_idx = np.argsort(-unobserved_factors[down_list])[:max_len]
        down_list = down_list[sorted_idx]
        down_str = ''
        if len(down_list) > 0:
            down_str =  f'<font point-size="{event_fontsize}" color="{down_color}">-</font>' + ','.join(var_names[down_list])

        self.event_str = up_str
        sep_str = ''
        if len(up_list) > 0 and len(down_list) > 0:
            sep_str = '<br/><br/>'
        self.event_str = self.event_str + sep_str + down_str
