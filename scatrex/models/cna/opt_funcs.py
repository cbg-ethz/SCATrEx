import jax
from jax.api import jit, grad, vmap
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp
import jax.nn as jnn

from scatrex.util import *
from scatrex.callbacks import elbos_callback

from functools import partial


def compute_elbo(
    rng,
    data,
    cell_covariates,
    lib_sizes,
    unobserved_factors_kernel_rate,
    unobserved_factors_kernel_concentration,
    unobserved_factors_root_kernel,
    global_noise_factors_precisions_shape,
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
    do_sticks,
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
    batch_effects_mean,
    batch_effects_log_std,
):
    elbo, ll, kl, node_kl = _compute_elbo(
        rng,
        data,
        cell_covariates,
        lib_sizes,
        unobserved_factors_kernel_rate,
        unobserved_factors_kernel_concentration,
        unobserved_factors_root_kernel,
        global_noise_factors_precisions_shape,
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
        do_sticks,
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
        batch_effects_mean,
        batch_effects_log_std,
    )
    return elbo


def _compute_elbo(
    rng,
    data,
    cell_covariates,
    lib_sizes,
    unobserved_factors_kernel_rate,
    unobserved_factors_kernel_concentration,
    unobserved_factors_root_kernel,
    global_noise_factors_precisions_shape,
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
    do_sticks,
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
    batch_effects_mean,
    batch_effects_log_std,
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
            batch_effects_mean,
            batch_effects_log_std,
        ) = (
            globals[0],
            globals[1],
            globals[2],
            globals[3],
            globals[4],
            globals[5],
            globals[6],
            globals[7],
        )
        log_baseline_mean = jax.lax.stop_gradient(log_baseline_mean)
        log_baseline_log_std = jax.lax.stop_gradient(log_baseline_log_std)
        noise_factors_mean = jax.lax.stop_gradient(noise_factors_mean)
        noise_factors_log_std = jax.lax.stop_gradient(noise_factors_log_std)
        factor_precision_log_means = jax.lax.stop_gradient(factor_precision_log_means)
        factor_precision_log_stds = jax.lax.stop_gradient(factor_precision_log_stds)
        batch_effects_mean = jax.lax.stop_gradient(batch_effects_mean)
        batch_effects_log_std = jax.lax.stop_gradient(batch_effects_log_std)
        return (
            log_baseline_mean,
            log_baseline_log_std,
            noise_factors_mean,
            noise_factors_log_std,
            factor_precision_log_means,
            factor_precision_log_stds,
            batch_effects_mean,
            batch_effects_log_std,
        )

    def alt_global(globals):
        (
            log_baseline_mean,
            log_baseline_log_std,
            noise_factors_mean,
            noise_factors_log_std,
            factor_precision_log_means,
            factor_precision_log_stds,
            batch_effects_mean,
            batch_effects_log_std,
        ) = (
            globals[0],
            globals[1],
            globals[2],
            globals[3],
            globals[4],
            globals[5],
            globals[6],
            globals[7],
        )
        return (
            log_baseline_mean,
            log_baseline_log_std,
            noise_factors_mean,
            noise_factors_log_std,
            factor_precision_log_means,
            factor_precision_log_stds,
            batch_effects_mean,
            batch_effects_log_std,
        )

    (
        log_baseline_mean,
        log_baseline_log_std,
        noise_factors_mean,
        noise_factors_log_std,
        factor_precision_log_means,
        factor_precision_log_stds,
        batch_effects_mean,
        batch_effects_log_std,
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
            batch_effects_mean,
            batch_effects_log_std,
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

    def stop_sticks(locals):
        return (
            jax.lax.stop_gradient(nu_sticks_log_means),
            jax.lax.stop_gradient(nu_sticks_log_stds),
            jax.lax.stop_gradient(psi_sticks_log_means),
            jax.lax.stop_gradient(psi_sticks_log_stds),
        )

    def keep_sticks(locals):
        return (
            nu_sticks_log_means,
            nu_sticks_log_stds,
            psi_sticks_log_means,
            psi_sticks_log_stds,
        )

    (
        nu_sticks_log_means,
        nu_sticks_log_stds,
        psi_sticks_log_means,
        psi_sticks_log_stds,
    ) = jax.lax.cond(
        do_sticks,
        keep_sticks,
        stop_sticks,
        (
            nu_sticks_log_means,
            nu_sticks_log_stds,
            psi_sticks_log_means,
            psi_sticks_log_stds,
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
        return jax.lax.cond(data_mask_subset[i] != 1, stop_cell_grad, keep_cell_grad, i)

    cell_noise_mean, cell_noise_log_std = vmap(stop_cell_grads)(jnp.arange(mb_size))

    def has_children(i):
        return jnp.any(ancestor_nodes_indices.ravel() == i)

    def has_next_branch(i):
        return jnp.any(previous_branches_indices.ravel() == i)

    ones_vec = jnp.ones(cnvs[0].shape)
    zeros_vec = jnp.log(ones_vec)

    log_baseline = diag_gaussian_sample(rng, log_baseline_mean, log_baseline_log_std)

    # noise
    factor_precision_log_means = jnp.clip(
        factor_precision_log_means, a_min=jnp.log(1e-3), a_max=jnp.log(1e3)
    )
    factor_precision_log_stds = jnp.clip(
        factor_precision_log_stds, a_min=jnp.log(1e-3), a_max=jnp.log(1e2)
    )
    log_factors_precisions = diag_gaussian_sample(
        rng, factor_precision_log_means, factor_precision_log_stds
    )
    noise_factors_mean = jnp.clip(noise_factors_mean, a_min=-10.0, a_max=10.0)
    noise_factors_log_std = jnp.clip(
        noise_factors_log_std, a_min=jnp.log(1e-3), a_max=jnp.log(1e2)
    )
    noise_factors = diag_gaussian_sample(rng, noise_factors_mean, noise_factors_log_std)
    cell_noise_mean = jnp.clip(cell_noise_mean, a_min=-10.0, a_max=10.0)
    cell_noise_log_std = jnp.clip(
        cell_noise_log_std, a_min=jnp.log(1e-3), a_max=jnp.log(1e2)
    )
    cell_noise = diag_gaussian_sample(rng, cell_noise_mean, cell_noise_log_std)
    noise = jnp.dot(cell_noise, noise_factors)

    # batch effects
    batch_effects_mean = jnp.clip(batch_effects_mean, a_min=-10.0, a_max=10.0)
    batch_effects_log_std = jnp.clip(
        batch_effects_log_std, a_min=jnp.log(1e-3), a_max=jnp.log(1e2)
    )
    batch_effects_factors = diag_gaussian_sample(
        rng, batch_effects_mean, batch_effects_log_std
    )
    cell_covariates = jnp.array(cell_covariates)[indices]
    batch_effects = jnp.dot(cell_covariates, batch_effects_factors)

    # unobserved factors
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
        return diag_gaussian_sample(rng, unobserved_means[i], unobserved_log_stds[i])

    def sample_all_unobs(i):
        return jax.lax.cond(node_mask[i] >= 0, sample_unobs, lambda i: zeros_vec, i)

    nodes_unobserved_factors = vmap(sample_all_unobs)(jnp.arange(len(cnvs)))

    nu_sticks_log_alphas = jnp.clip(nu_sticks_log_means, -3.0, 3.0)
    nu_sticks_log_betas = jnp.clip(nu_sticks_log_stds, -3.0, 3.0)

    # def sample_nu(i):
    #     return jnp.clip(
    #         diag_gaussian_sample(
    #             rng, nu_sticks_log_means[i], nu_sticks_log_stds[i]
    #         ),
    #         -4.0,
    #         4.0,
    #     )
    #
    # def sample_valid_nu(i):
    #     return jax.lax.cond(
    #         has_children(i), sample_nu, lambda i: jnp.array([logit(1 - 1e-6)]), i
    #     )  # Sample all valid
    #
    # def sample_all_nus(i):
    #     return jax.lax.cond(
    #         cnvs[i][0] >= 0, sample_valid_nu, lambda i: jnp.array([logit(1e-6)]), i
    #     )  # Sample all
    #
    # log_nu_sticks = vmap(sample_all_nus)(jnp.arange(len(cnvs)))
    #
    # def sigmoid_nus(i):
    #     return jax.lax.cond(
    #         cnvs[i][0] >= 0,
    #         lambda i: jnn.sigmoid(log_nu_sticks[i]),
    #         lambda i: jnp.array([1e-6]),
    #         i,
    #     )  # Sample all
    #
    # nu_sticks = jnp.clip(vmap(sigmoid_nus)(jnp.arange(len(cnvs))), 1e-6, 1 - 1e-6)

    def sample_nu(i):
        return jnp.clip(
            beta_sample(rng, nu_sticks_log_alphas[i], nu_sticks_log_betas[i]),
            1e-4,
            1 - 1e-4,
        )

    def sample_valid_nu(i):
        return jax.lax.cond(
            has_children(i), sample_nu, lambda i: jnp.array([1 - 1e-4]), i
        )  # Sample all valid

    def sample_all_nus(i):
        return jax.lax.cond(
            cnvs[i][0] >= 0, sample_valid_nu, lambda i: jnp.array([1e-4]), i
        )  # Sample all

    nu_sticks = vmap(sample_all_nus)(jnp.arange(len(cnvs)))

    psi_sticks_log_alphas = jnp.clip(psi_sticks_log_means, -3.0, 3.0)
    psi_sticks_log_betas = jnp.clip(psi_sticks_log_stds, -3.0, 3.0)

    # def sample_psi(i):
    #     return jnp.clip(
    #         diag_gaussian_sample(
    #             rng, psi_sticks_log_means[i], psi_sticks_log_stds[i]
    #         ),
    #         -4.0,
    #         4.0,
    #     )
    #
    # def sample_valid_psis(i):
    #     return jax.lax.cond(
    #         has_next_branch(i),
    #         sample_psi,
    #         lambda i: jnp.array([logit(1 - 1e-6)]),
    #         i,
    #     )  # Sample all valid
    #
    # def sample_all_psis(i):
    #     return jax.lax.cond(
    #         cnvs[i][0] >= 0,
    #         sample_valid_psis,
    #         lambda i: jnp.array([logit(1e-6)]),
    #         i,
    #     )  # Sample all
    #
    # log_psi_sticks = vmap(sample_all_psis)(jnp.arange(len(cnvs)))
    #
    # def sigmoid_psis(i):
    #     return jax.lax.cond(
    #         cnvs[i][0] >= 0,
    #         lambda i: jnn.sigmoid(log_psi_sticks[i]),
    #         lambda i: jnp.array([1e-6]),
    #         i,
    #     )  # Sample all
    #
    # psi_sticks = jnp.clip(vmap(sigmoid_psis)(jnp.arange(len(cnvs))), 1e-6, 1 - 1e-6)

    def sample_psi(i):
        return jnp.clip(
            beta_sample(rng, psi_sticks_log_alphas[i], psi_sticks_log_betas[i]),
            1e-4,
            1 - 1e-4,
        )

    def sample_valid_psis(i):
        return jax.lax.cond(
            has_next_branch(i),
            sample_psi,
            lambda i: jnp.array([1 - 1e-4]),
            i,
        )  # Sample all valid

    def sample_all_psis(i):
        return jax.lax.cond(
            cnvs[i][0] >= 0,
            sample_valid_psis,
            lambda i: jnp.array([1e-4]),
            i,
        )  # Sample all

    psi_sticks = vmap(sample_all_psis)(jnp.arange(len(cnvs)))

    lib_sizes = jnp.array(lib_sizes)[indices]
    data = jnp.array(data)[indices]
    baseline = jnp.exp(jnp.append(0, log_baseline))

    def compute_node_ll(i):
        unobserved_factors = nodes_unobserved_factors[i] * (parent_vector[i] != -1)

        node_mean = (
            baseline * cnvs[i] / 2 * jnp.exp(unobserved_factors + noise + batch_effects)
        )
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

    small_ll = -1e30 * jnp.ones((mb_size))

    def get_node_ll(i):
        return jnp.where(
            node_mask[i] == 1,
            compute_node_ll(jnp.where(node_mask[i] == 1, i, 0)),
            small_ll,
        )

    out = jnp.array(vmap(get_node_ll)(jnp.arange(len(parent_vector))))
    l = jnp.sum(jnn.logsumexp(out, axis=0) * data_mask_subset)

    log_rate = jnp.log(unobserved_factors_kernel_rate)
    log_concentration = jnp.log(unobserved_factors_kernel_concentration)
    log_kernel = jnp.log(unobserved_factors_root_kernel)
    broadcasted_concentration = log_concentration * ones_vec
    broadcasted_rate = log_rate * ones_vec

    def compute_node_kl(i):
        kl = 0.0
        pl = diag_gamma_logpdf(
            jnp.clip(jnp.exp(nodes_log_unobserved_factors_kernels[i]), a_min=1e-6),
            broadcasted_concentration,
            (parent_vector[i] != -1)
            * (parent_vector[i] != 0)
            * unobserved_factors_kernel_rate
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
            * is_root_subtree  # promote overexpressing events near the root
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
        nu_ent = has_children(i) * -beta_logpdf(
            nu_sticks[i], nu_sticks_log_alphas[i], nu_sticks_log_betas[i]
        )
        kl += nu_pl + nu_ent

        psi_pl = has_next_branch(i) * beta_logpdf(
            psi_sticks[i],
            jnp.log(jnp.array([1.0])),
            jnp.log(jnp.array([dp_gammas[i]])),
        )
        psi_ent = has_next_branch(i) * -beta_logpdf(
            psi_sticks[i], psi_sticks_log_alphas[i], psi_sticks_log_betas[i]
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
        jnp.log(global_noise_factors_precisions_shape) * ones_mat,
        zeros_mat,
    ) - diag_loggaussian_logpdf(
        jnp.exp(log_factors_precisions),
        factor_precision_log_means,
        factor_precision_log_stds,
    )
    noise_factors_kl = diag_gaussian_logpdf(
        noise_factors,
        jnp.zeros(noise_factors.shape),
        jnp.log(jnp.sqrt(1.0 / jnp.exp(log_factors_precisions)).reshape(-1, 1))
        * jnp.ones(noise_factors.shape),
    ) - diag_gaussian_logpdf(noise_factors, noise_factors_mean, noise_factors_log_std)
    batch_effects_kl = diag_gaussian_logpdf(
        batch_effects_factors,
        jnp.zeros(batch_effects_factors.shape),
        jnp.zeros(batch_effects_factors.shape),
    ) - diag_gaussian_logpdf(
        batch_effects_factors, batch_effects_mean, batch_effects_log_std
    )
    total_kl = (
        node_kl
        + baseline_kl
        + factor_precision_kl
        + noise_factors_kl
        + batch_effects_kl
    )

    # Scale the KL by the data size
    total_kl = total_kl * jnp.sum(data_mask_subset != 0) / data.shape[0]

    zeros_mat = jnp.zeros(cell_noise.shape)
    cell_noise_kl = diag_gaussian_logpdf(
        cell_noise, zeros_mat, zeros_mat, axis=1
    ) - diag_gaussian_logpdf(cell_noise, cell_noise_mean, cell_noise_log_std, axis=1)
    cell_noise_kl = jnp.sum(cell_noise_kl * data_mask_subset)
    total_kl = total_kl + cell_noise_kl

    elbo_val = l + total_kl

    return elbo_val, l, total_kl, node_kls


def batch_elbo(
    rng,
    data,
    cell_covariates,
    lib_sizes,
    unobserved_factors_kernel_rate,
    unobserved_factors_kernel_concentration,
    unobserved_factors_root_kernel,
    global_noise_factors_precisions_shape,
    obs_params,
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
    do_sticks,
    sticks_only,
    params,
    num_samples,
):
    # Average over a batch of random samples from the var approx.
    rngs = random.split(rng, num_samples)
    init = [0]
    init.extend([None] * (23 + len(params)))
    vectorized_elbo = vmap(compute_elbo, in_axes=init)
    return jnp.mean(
        vectorized_elbo(
            rngs,
            data,
            cell_covariates,
            lib_sizes,
            unobserved_factors_kernel_rate,
            unobserved_factors_kernel_concentration,
            unobserved_factors_root_kernel,
            global_noise_factors_precisions_shape,
            obs_params,
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
            do_sticks,
            sticks_only,
            *params,
        )
    )


@partial(jit, static_argnums=(3, 4, 5, 6, 23))
def objective(
    data,
    cell_covariates,
    lib_sizes,
    unobserved_factors_kernel_rate,
    unobserved_factors_kernel_concentration,
    unobserved_factors_root_kernel,
    global_noise_factors_precisions_shape,
    obs_params,
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
    do_sticks,
    sticks_only,
    num_samples,
    params,
    t,
):
    rng = random.PRNGKey(t)
    return -batch_elbo(
        rng,
        data,
        cell_covariates,
        lib_sizes,
        unobserved_factors_kernel_rate,
        unobserved_factors_kernel_concentration,
        unobserved_factors_root_kernel,
        global_noise_factors_precisions_shape,
        obs_params,
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
        do_sticks,
        sticks_only,
        params,
        num_samples,
    )


@partial(jit, static_argnums=(3, 4, 5, 6, 21, 22))
def batch_objective(
    data,
    cell_covariates,
    lib_sizes,
    unobserved_factors_kernel_rate,
    unobserved_factors_kernel_concentration,
    unobserved_factors_root_kernel,
    global_noise_factors_precisions_shape,
    obs_params,
    parent_vector,
    children_vector,
    ancestor_nodes_indices,
    tssb_indices,
    previous_branches_indices,
    tssb_weights,
    dp_alphas,
    dp_gammas,
    node_mask,
    do_global,
    global_only,
    do_sticks,
    sticks_only,
    num_samples,
    num_data,
    params,
    t,
):
    rng = random.PRNGKey(t)
    # Average over a batch of random samples from the var approx.
    rngs = random.split(rng, num_samples)
    init = [0]
    init.extend([None] * (23 + len(params)))
    vectorized_elbo = vmap(_compute_elbo, in_axes=init)
    elbos, lls, kls, node_kls = vectorized_elbo(
        rngs,
        data,
        cell_covariates,
        lib_sizes,
        unobserved_factors_kernel_rate,
        unobserved_factors_kernel_concentration,
        unobserved_factors_root_kernel,
        global_noise_factors_precisions_shape,
        obs_params,
        parent_vector,
        children_vector,
        ancestor_nodes_indices,
        tssb_indices,
        previous_branches_indices,
        tssb_weights,
        dp_alphas,
        dp_gammas,
        node_mask,
        jnp.ones((num_data,)),
        jnp.arange(num_data),
        do_global,
        global_only,
        do_sticks,
        sticks_only,
        *params,
    )
    elbo = jnp.mean(elbos)
    ll = jnp.mean(lls)
    kl = jnp.mean(kls)
    node_kl = node_kls
    return elbo, ll, kl, node_kl


@partial(jit, static_argnums=(3, 4, 5, 6, 23))
def do_grad(
    data,
    cell_covariates,
    lib_sizes,
    unobserved_factors_kernel_rate,
    unobserved_factors_kernel_concentration,
    unobserved_factors_root_kernel,
    global_noise_factors_precisions_shape,
    obs_params,
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
    do_sticks,
    sticks_only,
    num_samples,
    params,
    i,
):
    return jax.value_and_grad(objective, argnums=24)(
        data,
        cell_covariates,
        lib_sizes,
        unobserved_factors_kernel_rate,
        unobserved_factors_kernel_concentration,
        unobserved_factors_root_kernel,
        global_noise_factors_precisions_shape,
        obs_params,
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
        do_sticks,
        sticks_only,
        num_samples,
        params,
        i,
    )


def update(
    data,
    cell_covariates,
    lib_sizes,
    unobserved_factors_kernel_rate,
    unobserved_factors_kernel_concentration,
    unobserved_factors_root_kernel,
    global_noise_factors_precisions_shape,
    obs_params,
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
    do_sticks,
    sticks_only,
    num_samples,
    i,
    opt_state,
    opt_update,
    get_params,
):
    # print("Recompiling update!")
    params = get_params(opt_state)
    value, gradient = do_grad(
        data,
        cell_covariates,
        lib_sizes,
        unobserved_factors_kernel_rate,
        unobserved_factors_kernel_concentration,
        unobserved_factors_root_kernel,
        global_noise_factors_precisions_shape,
        obs_params,
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
        do_sticks,
        sticks_only,
        num_samples,
        params,
        i,
    )
    opt_state = opt_update(i, gradient, opt_state)
    return opt_state, gradient, params, value
