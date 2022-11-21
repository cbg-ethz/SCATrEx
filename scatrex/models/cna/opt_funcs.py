import jax
from jax import jit, grad, vmap
from jax import random
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.nn as jnn

from scatrex.util import *
from scatrex.callbacks import elbos_callback

from jax.scipy.special import digamma, betaln, gammaln

from functools import partial

def loggaussian_ent(mean, std):
    return jnp.log(std) + mean + 0.5 + 0.5*jnp.log(2*jnp.pi)

def diag_loggaussian_ent(mean, log_std, axis=None):
    return jnp.sum(vmap(loggaussian_ent)(mean, jnp.exp(log_std)), axis=axis)


def complete_elbo(self, rng, mc_samples=3):
    rngs = random.split(rng, mc_samples)

    # Get data
    data = self.data
    lib_sizes = self.root["node"].root["node"].lib_sizes
    n_cells, n_genes = self.data.shape

    # Get global variational parameters
    log_baseline_mean = self.root["node"].root["node"].variational_parameters["globals"]["log_baseline_mean"]
    log_baseline_log_std = self.root["node"].root["node"].variational_parameters["globals"]["log_baseline_log_std"]
    cell_noise_mean = self.root["node"].root["node"].variational_parameters["globals"]["cell_noise_mean"]
    cell_noise_log_std = self.root["node"].root["node"].variational_parameters["globals"]["cell_noise_log_std"]
    noise_factors_mean = self.root["node"].root["node"].variational_parameters["globals"]["noise_factors_mean"]
    noise_factors_log_std = self.root["node"].root["node"].variational_parameters["globals"]["noise_factors_log_std"]

    # Sample global
    baseline_samples = sample_baseline(rngs, log_baseline_mean, log_baseline_log_std)
    cell_noise_samples = sample_cell_noise_factors(rngs, cell_noise_mean, cell_noise_log_std)
    noise_factor_samples = sample_noise_factors(rngs, noise_factors_mean, noise_factors_log_std)

    def sub_descend(root, depth=0):
        elbo = 0
        subtree_weight = 0

        if root["node"].parent() is None:
            parent_unobserved_samples = jnp.zeros((mc_samples, n_genes))
            unobserved_samples = jnp.zeros((mc_samples, n_genes))
            unobserved_kernel_samples = jnp.zeros((mc_samples, n_genes))
        else:
            # Sample parent
            parent_unobserved_means = root["node"].parent().variational_parameters["locals"]["unobserved_factors_mean"]
            parent_unobserved_log_stds = root["node"].parent().variational_parameters["locals"]["unobserved_factors_log_std"]
            parent_unobserved_samples = sample_unobserved(rngs, parent_unobserved_means, parent_unobserved_log_stds)

            # Sample node
            unobserved_means = root["node"].variational_parameters["locals"]["unobserved_factors_mean"]
            unobserved_log_stds = root["node"].variational_parameters["locals"]["unobserved_factors_log_std"]
            unobserved_samples = sample_unobserved(rngs, unobserved_means, unobserved_log_stds)
            unobserved_factors_kernel_log_mean = root["node"].variational_parameters["locals"]["unobserved_factors_kernel_log_mean"]
            unobserved_factors_kernel_log_std = root["node"].variational_parameters["locals"]["unobserved_factors_kernel_log_std"]
            unobserved_kernel_samples = sample_unobserved_kernel(rngs, unobserved_factors_kernel_log_mean, unobserved_factors_kernel_log_std)

        psi_not_prev_sum = 0
        for i, child in enumerate(root["children"]):
            nu_alpha = child["node"].variational_parameters["locals"]["nu_log_mean"]
            nu_beta = child["node"].variational_parameters["locals"]["nu_log_std"]
            psi_alpha = child["node"].variational_parameters["locals"]["psi_log_mean"]
            psi_beta = child["node"].variational_parameters["locals"]["psi_log_std"]

            # Compute local exact KL divergence term
            child["node"].stick_kl = beta_kl(nu_alpha, nu_beta, 1, (root["node"].tssb.alpha_decay**depth)*root["node"].tssb.dp_alpha) + beta_kl(psi_alpha, psi_beta, 1, root["node"].tssb.dp_gamma)

            # Compute expected local NTSSB weight term
            E_log_phi = E_q_log_beta(psi_alpha, psi_beta) + psi_not_prev_sum
            E_log_nu = E_q_log_beta(nu_alpha, nu_beta)
            E_log_1_nu = E_q_log_1_beta(nu_alpha, nu_beta)

            child["node"].weight_until_here = child["node"].data_weights + root["node"].weight_until_here
            child["node"].expected_weights = child["node"].data_weights*E_log_nu + root["node"].weight_until_here*E_log_1_nu + child["node"].weight_until_here*E_log_phi

            if i > 0:
                psi_not_prev_sum += E_q_log_1_beta(psi_alpha, psi_beta)

            # Go down in the tree
            elbo, subtree_weight = sub_descend(child, depth=depth+1)


        # Compute local approximate expected log likelihood term
        root["node"].ell = ell(root["node"].data_weights, baseline_samples, cell_noise_samples, noise_factor_samples, unobserved_samples, root["node"].cnvs, lib_sizes, data)

        if root["node"].parent() is None:
            root["node"].param_kl = 0.
        else:
            # Compute local approximate KL divergence term
            root["node"].param_kl = local_paramkl(parent_unobserved_samples, unobserved_samples, unobserved_kernel_samples,
                                                unobserved_means,
                                                unobserved_log_stds,
                                                unobserved_factors_kernel_log_mean,
                                                unobserved_factors_kernel_log_std,
                                                jnp.array([0.1, 1.]), jnp.array(0.))

        # If this node is the root of its subtree, compute expected weights here
        if depth == 0:
            nu_alpha = child["node"].variational_parameters["locals"]["nu_log_mean"]
            nu_beta = child["node"].variational_parameters["locals"]["nu_log_std"]
            psi_alpha = child["node"].variational_parameters["locals"]["psi_log_mean"]
            psi_beta = child["node"].variational_parameters["locals"]["psi_log_std"]

            E_log_nu = E_q_log_beta(nu_alpha, nu_beta)
            root["node"].expected_weights = root["node"].data_weights*E_log_nu

            # Compute local exact KL divergence term
            root["node"].stick_kl = beta_kl(nu_alpha, nu_beta, 1, (root["node"].tssb.alpha_decay**depth)*root["node"].tssb.dp_alpha) + beta_kl(psi_alpha, psi_beta, 1, root["node"].tssb.dp_gamma)


        # Compute local ELBO quantities
        root["node"].local_elbo = root["node"].ell + root["node"].expected_weights - root["node"].data_weights*np.log(root["node"].data_weights)
        root["node"].local_elbo = root["node"].local_elbo - root["node"].param_kl - root["node"].stick_kl

        # Remove root contribution
        elbo = elbo + root["node"].local_elbo

        # Track total weight in the subtree
        subtree_weight += root["node"].data_weights

        return elbo, subtree_weight

    def descend(super_root, elbo=0):
        subtree_elbo, subtree_weights = sub_descend(super_root["node"].root)
        elbo += np.sum(subtree_elbo + subtree_weights * super_root["node"].weight)
        for super_child in super_root["children"]:
            elbo += descend(super_child)
        return elbo

    elbo = descend(self.root)

    # Compute global KL
    global_kl = jnp.sum(baseline_kl(log_baseline_mean, log_baseline_log_std))
    global_kl += jnp.sum(noise_factors_kl(noise_factors_mean, noise_factors_log_std))
    global_kl += jnp.sum(cell_noise_kl(cell_noise_mean, cell_noise_log_std))

    # Add to ELBO
    elbo = elbo - global_kl - self.root["node"].root["node"].local_elbo

    return elbo

@jax.jit
def beta_kl(a1, b1, a2, b2):
    def logbeta_func(a,b):
        return gammaln(a) + gammaln(b) - gammaln(a+b)

    kl = logbeta_func(a1,b1) - logbeta_func(a2,b2)
    kl += (a1-a2)*digamma(a1)
    kl += (b1-b2)*digamma(b1)
    kl += (a2-a1 + b2-b1)*digamma(a1+b1)
    return kl

@jax.jit
def E_q_log_beta(
    alpha,
    beta,
):
    return digamma(alpha) - digamma(alpha + beta)

@jax.jit
def E_q_log_1_beta(
    alpha,
    beta,
):
    return digamma(beta) - digamma(alpha + beta)

# This computes E_q[p(z_n=\epsilon | \nu, \psi)]
@jax.jit
def compute_expected_weight(
    nu_alpha,
    nu_beta,
    psi_alpha,
    psi_beta,
    prev_nu_sticks_sum,
    prev_psi_sticks_sum,
):
    nu_digamma_sum = digamma(nu_alpha + nu_beta)
    E_log_nu = digamma(nu_alpha) - nu_digamma_sum
    nu_sticks_sum = digamma(nu_beta) - nu_digamma_sum + prev_nu_sticks_sum
    psi_sticks_sum = local_psi_sticks_sum(psi_alpha, psi_beta) + prev_psi_sticks_sum
    weight = E_log_nu + nu_sticks_sum + psi_sticks_sum
    return weight, nu_sticks_sum, psi_sticks_sum

@jax.jit
def sample_baseline(
    rngs,
    log_baseline_mean,
    log_baseline_log_std,
):
    def _sample(rng, log_baseline_mean, log_baseline_log_std):
        return jnp.append(1, jnp.exp(diag_gaussian_sample(rng, log_baseline_mean, log_baseline_log_std)))
    vectorized_sample = vmap(_sample, in_axes=[0, None, None])
    return vectorized_sample(rngs, log_baseline_mean, log_baseline_log_std)

@jax.jit
def sample_cell_noise_factors(
    rngs,
    cell_noise_mean,
    cell_noise_log_std,
):
    def _sample(rng, cell_noise_mean, cell_noise_log_std):
        return diag_gaussian_sample(rng, cell_noise_mean, cell_noise_log_std)
    vectorized_sample = vmap(_sample, in_axes=[0, None, None])
    return vectorized_sample(rngs, cell_noise_mean, cell_noise_log_std)

@jax.jit
def sample_noise_factors(
    rngs,
    noise_factors_mean,
    noise_factors_log_std,
):
    def _sample(rng, noise_factors_mean, noise_factors_log_std):
        return diag_gaussian_sample(rng, noise_factors_mean, noise_factors_log_std)
    vectorized_sample = vmap(_sample, in_axes=[0, None, None])
    return vectorized_sample(rngs, noise_factors_mean, noise_factors_log_std)

@jax.jit
def sample_unobserved(
    rngs,
    unobserved_means,
    unobserved_log_stds,
):
    def _sample(rng, unobserved_means, unobserved_log_stds):
        return diag_gaussian_sample(rng, unobserved_means, unobserved_log_stds)
    vectorized_sample = vmap(_sample, in_axes=[0, None, None])
    return vectorized_sample(rngs, unobserved_means, unobserved_log_stds)

@jax.jit
def sample_unobserved_kernel(
    rngs,
    log_unobserved_factors_kernel_means,
    log_unobserved_factors_kernel_log_stds,
):
    def _sample(rng, log_unobserved_factors_kernel_means, log_unobserved_factors_kernel_log_stds):
        return jnp.exp(diag_gaussian_sample(rng, log_unobserved_factors_kernel_means, log_unobserved_factors_kernel_log_stds))
    vectorized_sample = vmap(_sample, in_axes=[0, None, None])
    return vectorized_sample(rngs, log_unobserved_factors_kernel_means, log_unobserved_factors_kernel_log_stds)

@jax.jit
def baseline_kl(
    log_baseline_mean,
    log_baseline_log_std,
):
    std = jnp.exp(log_baseline_log_std)
    return -log_baseline_log_std + .5 * (std**2 + log_baseline_mean**2 - 1.)


@jax.jit
def cell_noise_kl(
    cell_noise_mean,
    cell_noise_log_std,
):
    std = jnp.exp(cell_noise_log_std)
    return -cell_noise_log_std + .5 * (std**2 + cell_noise_mean**2 - 1.)


@jax.jit
def noise_factors_kl(
    noise_factors_mean,
    noise_factors_log_std,
):
    std = jnp.exp(noise_factors_log_std)
    return -noise_factors_log_std + .5 * (std**2 + noise_factors_mean**2 - 1.)

@jax.jit
def ell(
    data_node_weights, # lambda_{nk} with only the node attachments
    baseline_samples,
    cell_noise_samples,
    noise_factor_samples,
    unobserved_samples, # N vector corresponding to node attachments
    cnv, # N vector corresponding to node attachments
    lib_sizes,
    data,
    mask,
):
    def _ell(data_node_weights, baseline_sample, cell_noise_sample, noise_factor_sample, unobserved_sample):
        noise_sample = cell_noise_sample.dot(noise_factor_sample)
        node_mean = (
            baseline_sample * cnv/2 * jnp.exp(unobserved_sample + noise_sample)
        )
        sum = jnp.sum(node_mean, axis=1).reshape(-1, 1)
        node_mean = node_mean / sum
        node_mean = node_mean * lib_sizes
        pll = vmap(jax.scipy.stats.poisson.logpmf)(data, node_mean)
        ell = jnp.sum(pll, axis=1)  # N-vector
        ell *= data_node_weights
        ell *= mask
        return ell
    vectorized = vmap(_ell, in_axes=[None, 0,0,0,0])
    return jnp.mean(vectorized(data_node_weights, baseline_samples, cell_noise_samples, noise_factor_samples, unobserved_samples), axis=0)


@jax.jit
def ll(
    baseline_samples,
    cell_noise_samples,
    noise_factor_samples,
    unobserved_samples,
    cnv,
    lib_sizes,
    data,
):
    def _ll(baseline_sample, cell_noise_sample, noise_factor_sample, unobserved_sample):
        noise_sample = cell_noise_sample.dot(noise_factor_sample)
        node_mean = (
            baseline_sample * cnv/2 * jnp.exp(unobserved_sample + noise_sample)
        )
        sum = jnp.sum(node_mean, axis=1).reshape(-1, 1)
        node_mean = node_mean / sum
        node_mean = node_mean * lib_sizes
        pll = vmap(jax.scipy.stats.poisson.logpmf)(data, node_mean)
        ll = jnp.sum(pll, axis=1)  # N-vector
        return ll
    vectorized = vmap(_ll, in_axes=[0,0,0,0])
    return jnp.mean(vectorized(baseline_samples, cell_noise_samples, noise_factor_samples, unobserved_samples), axis=0)



@jax.jit
def local_paramkl(
    parent_unobserved_samples,
    child_unobserved_samples,
    child_unobserved_kernel_samples,
    child_unobserved_means,
    child_unobserved_log_stds,
    child_log_unobserved_factors_kernel_means,
    child_log_unobserved_factors_kernel_log_stds,
    hyperparams, # [concentration, rate]
    child_in_root_subtree, # to make the prior prefer amplifications
):
    def _local_paramkl(
        parent_unobserved_sample,
        child_unobserved_sample,
        child_unobserved_kernel_sample,
        child_unobserved_means,
        child_unobserved_log_stds,
        child_log_unobserved_factors_kernel_means,
        child_log_unobserved_factors_kernel_log_stds,
        hyperparams,
        ):
        kl = 0.0
        # Kernel
        pl = diag_gamma_logpdf(
            child_unobserved_kernel_sample,
            jnp.log(hyperparams[0] * jnp.ones((child_unobserved_kernel_sample.shape[0],))),
            (hyperparams[1]*jnp.abs(parent_unobserved_sample)),
        )
        ent = -diag_loggaussian_logpdf(
            child_unobserved_kernel_sample,
            child_log_unobserved_factors_kernel_means,
            child_log_unobserved_factors_kernel_log_stds,
        )
#         ent = diag_loggaussian_ent(child_log_unobserved_factors_kernel_means, child_log_unobserved_factors_kernel_log_stds)
        kl += pl + ent

        # Cell state
        pl = diag_gaussian_logpdf(
            child_unobserved_sample,
            parent_unobserved_sample,
            jnp.log(0.001 + child_unobserved_kernel_sample),
        )
        ent = -diag_gaussian_logpdf(
            child_unobserved_sample, child_unobserved_means, child_unobserved_log_stds
        )
        kl += pl + ent

        return kl
    vectorized = vmap(_local_paramkl, in_axes=[0,0,0,None,None,None,None,None])
    return jnp.mean(vectorized(parent_unobserved_samples, child_unobserved_samples,
            child_unobserved_kernel_samples, child_unobserved_means, child_unobserved_log_stds,
            child_log_unobserved_factors_kernel_means, child_log_unobserved_factors_kernel_log_stds, hyperparams))

@jax.jit
def update_local_parameters(
    rngs,
    child_unobserved_means,
    child_unobserved_log_stds,
    child_log_unobserved_factors_kernel_means,
    child_log_unobserved_factors_kernel_log_stds,
    data_node_weights,
    parent_unobserved_samples,
    baseline_samples,
    cell_noise_samples,
    noise_factor_samples,
    hyperparams, # [concentration, rate]
    child_in_root_subtree, # to make the prior prefer amplifications
    cnv,
    lib_sizes,
    data,
    mask,
    states,
    i,
    mb_scaling=1,
    lr=0.01
    ):

    def local_loss(
        params,
        parent_unobserved_samples,
        baseline_samples,
        cell_noise_samples,
        noise_factor_samples,
        hyperparams,
        child_in_root_subtree
        ):
        child_unobserved_means = params[0]
        child_unobserved_log_stds = params[1]
        child_log_unobserved_factors_kernel_means = params[2]
        child_log_unobserved_factors_kernel_log_stds = params[3]

        child_unobserved_kernel_samples = sample_unobserved_kernel(rngs, child_log_unobserved_factors_kernel_means, child_log_unobserved_factors_kernel_log_stds)
        child_unobserved_samples = sample_unobserved(rngs, child_unobserved_means, child_unobserved_log_stds)

        child_unobserved_kernel_samples = jnp.clip(child_unobserved_kernel_samples, a_min=1e-8, a_max=5)

        loss = 0.

        loss = jnp.sum(ell(data_node_weights,
            baseline_samples,
            cell_noise_samples,
            noise_factor_samples,
            child_unobserved_samples,
            cnv,
            lib_sizes,
            data,
            mask,))
        kl = local_paramkl(parent_unobserved_samples,
            child_unobserved_samples,
            child_unobserved_kernel_samples,
            child_unobserved_means,
            child_unobserved_log_stds,
            child_log_unobserved_factors_kernel_means,
            child_log_unobserved_factors_kernel_log_stds,
            hyperparams, # [concentration, rate]
            child_in_root_subtree,)
        # Scale by minibatch
        loss = loss + kl*mb_scaling
        return loss

    child_unobserved_log_stds = jnp.clip(child_unobserved_log_stds, a_min=jnp.log(1e-8))
    child_log_unobserved_factors_kernel_means = jnp.clip(child_log_unobserved_factors_kernel_means, a_min=jnp.log(1e-8), a_max=0.)
    child_log_unobserved_factors_kernel_log_stds = jnp.clip(child_log_unobserved_factors_kernel_log_stds, a_min=jnp.log(1e-8), a_max=0.)

    params = jnp.array([child_unobserved_means, child_unobserved_log_stds,
                    child_log_unobserved_factors_kernel_means, child_log_unobserved_factors_kernel_log_stds])
    loss, grads = jax.value_and_grad(local_loss)(params, parent_unobserved_samples,
                                                baseline_samples,
                                                cell_noise_samples,
                                                noise_factor_samples,
                                                hyperparams,
                                                child_in_root_subtree)

    state1, state2, state3, state4 = states



    m3, v3 = state3
    b1=0.9
    b2=0.999
    eps=1e-8
    m3 = (1 - b1) * grads[2] + b1 * m3  # First  moment estimate.
    v3 = (1 - b2) * jnp.square(grads[2]) + b2 * v3  # Second moment estimate.
    state3 = (m3, v3)
    mhat = m3 / (1 - jnp.asarray(b1, m3.dtype) ** (i + 1))  # Bias correction.
    vhat = v3 / (1 - jnp.asarray(b2, m3.dtype) ** (i + 1))
    child_log_unobserved_factors_kernel_means = child_log_unobserved_factors_kernel_means + lr * mhat / (jnp.sqrt(vhat) + eps)


    m4, v4 = state4
    b1=0.9
    b2=0.999
    eps=1e-8
    m4 = (1 - b1) * grads[3] + b1 * m4  # First  moment estimate.
    v4 = (1 - b2) * jnp.square(grads[3]) + b2 * v4  # Second moment estimate.
    state4 = (m4, 4)
    mhat = m4 / (1 - jnp.asarray(b1, m4.dtype) ** (i + 1))  # Bias correction.
    vhat = v4 / (1 - jnp.asarray(b2, m4.dtype) ** (i + 1))
    child_log_unobserved_factors_kernel_log_stds = child_log_unobserved_factors_kernel_log_stds + lr * mhat / (jnp.sqrt(vhat) + eps)

    m1, v1 = state1
    b1=0.9
    b2=0.999
    eps=1e-8
    m1 = (1 - b1) * grads[0] + b1 * m1  # First  moment estimate.
    v1 = (1 - b2) * jnp.square(grads[0]) + b2 * v1  # Second moment estimate.
    state1 = (m1, v1)
    mhat = m1 / (1 - jnp.asarray(b1, m1.dtype) ** (i + 1))  # Bias correction.
    vhat = v1 / (1 - jnp.asarray(b2, m1.dtype) ** (i + 1))
    child_unobserved_means = child_unobserved_means + lr * mhat / (jnp.sqrt(vhat) + eps)

    m2, v2 = state2
    m2 = (1 - b1) * grads[1] + b1 * m2  # First  moment estimate.
    v2 = (1 - b2) * jnp.square(grads[1]) + b2 * v2  # Second moment estimate.
    state2 = (m2, v2)
    mhat = m2 / (1 - jnp.asarray(b1, m2.dtype) ** (i + 1))  # Bias correction.
    vhat = v2 / (1 - jnp.asarray(b2, m2.dtype) ** (i + 1))
    child_unobserved_log_stds = child_unobserved_log_stds + lr * mhat / (jnp.sqrt(vhat) + eps)



    states = (state1, state2, state3, state4)


    return loss, states, child_unobserved_means, child_unobserved_log_stds, child_log_unobserved_factors_kernel_means, child_log_unobserved_factors_kernel_log_stds

@jax.jit
def baseline_node_grad(
        rngs,
        log_baseline_mean,
        log_baseline_log_std,
        data_node_weights,
        child_unobserved_samples,
        cell_noise_samples,
        noise_factor_samples,
        cnv,
        lib_sizes,
        data,
        mask,):
    def local_loss(
        params,
        child_unobserved_samples,
        cell_noise_samples,
        noise_factor_samples,
        ):
        log_baseline_mean, log_baseline_log_std = params[0], params[1]
        baseline_samples = sample_baseline(rngs, log_baseline_mean, log_baseline_log_std)

        loss = jnp.sum(ell(data_node_weights,
            baseline_samples,
            cell_noise_samples,
            noise_factor_samples,
            child_unobserved_samples,
            cnv,
            lib_sizes,
            data,
            mask,))
        return loss

    params = jnp.array([log_baseline_mean, log_baseline_log_std])
    grads = jax.grad(local_loss)(params, child_unobserved_samples, cell_noise_samples, noise_factor_samples,)
    return grads

@jax.jit
def baseline_kl_grad(mean, log_std):
    def _kl(mean, log_std):
        return jnp.sum(baseline_kl(mean, log_std))
    return jnp.array(jax.grad(_kl, argnums=(0,1))(mean, log_std))

@jax.jit
def baseline_step(log_baseline_mean, log_baseline_log_std, grads, states, i, lr=0.01):
    state1, state2 = states

    m1, v1 = state1
    b1=0.9
    b2=0.999
    eps=1e-8
    m1 = (1 - b1) * grads[0] + b1 * m1  # First  moment estimate.
    v1 = (1 - b2) * jnp.square(grads[0]) + b2 * v1  # Second moment estimate.
    state1 = (m1, v1)
    mhat = m1 / (1 - jnp.asarray(b1, m1.dtype) ** (i + 1))  # Bias correction.
    vhat = v1 / (1 - jnp.asarray(b2, m1.dtype) ** (i + 1))
    log_baseline_mean = log_baseline_mean + lr * mhat / (jnp.sqrt(vhat) + eps)

    m2, v2 = state2
    m2 = (1 - b1) * grads[1] + b1 * m2  # First  moment estimate.
    v2 = (1 - b2) * jnp.square(grads[1]) + b2 * v2  # Second moment estimate.
    state2 = (m2, v2)
    mhat = m2 / (1 - jnp.asarray(b1, m2.dtype) ** (i + 1))  # Bias correction.
    vhat = v2 / (1 - jnp.asarray(b2, m2.dtype) ** (i + 1))
    log_baseline_log_std = log_baseline_log_std + lr * mhat / (jnp.sqrt(vhat) + eps)

    state = (state1, state2)

    return state, log_baseline_mean, log_baseline_log_std

@jax.jit
def noise_node_grad(
        rngs,
        noise_factors_mean,
        noise_factors_log_std,
        data_node_weights,
        child_unobserved_samples,
        cell_noise_samples,
        baseline_samples,
        cnv,
        lib_sizes,
        data,
        mask,):
    def local_loss(
        params,
        child_unobserved_samples,
        cell_noise_samples,
        baseline_samples,
        ):
        noise_factors_mean, noise_factors_log_std = params[0], params[1]
        noise_factor_samples = sample_noise_factors(rngs, noise_factors_mean, noise_factors_log_std)

        loss = jnp.sum(ell(data_node_weights,
            baseline_samples,
            cell_noise_samples,
            noise_factor_samples,
            child_unobserved_samples,
            cnv,
            lib_sizes,
            data,
            mask,))
        return loss

    params = jnp.array([noise_factors_mean, noise_factors_log_std])
    grads = jnp.array(jax.grad(local_loss)(params, child_unobserved_samples, cell_noise_samples, baseline_samples,))
    return grads

@jax.jit
def noise_kl_grad(mean, log_std):
    def _kl(mean, log_std):
        return jnp.sum(noise_factors_kl(mean, log_std))
    return jnp.array(jax.grad(_kl, argnums=(0,1))(mean, log_std))

@jax.jit
def noise_step(noise_factors_mean, noise_factors_log_std, grads, states, i, lr=0.01):
    state1, state2 = states

    m1, v1 = state1
    b1=0.9
    b2=0.999
    eps=1e-8
    m1 = (1 - b1) * grads[0] + b1 * m1  # First  moment estimate.
    v1 = (1 - b2) * jnp.square(grads[0]) + b2 * v1  # Second moment estimate.
    state1 = (m1, v1)
    mhat = m1 / (1 - jnp.asarray(b1, m1.dtype) ** (i + 1))  # Bias correction.
    vhat = v1 / (1 - jnp.asarray(b2, m1.dtype) ** (i + 1))
    noise_factors_mean = noise_factors_mean + lr * mhat / (jnp.sqrt(vhat) + eps)

    m2, v2 = state2
    m2 = (1 - b1) * grads[1] + b1 * m2  # First  moment estimate.
    v2 = (1 - b2) * jnp.square(grads[1]) + b2 * v2  # Second moment estimate.
    state2 = (m2, v2)
    mhat = m2 / (1 - jnp.asarray(b1, m2.dtype) ** (i + 1))  # Bias correction.
    vhat = v2 / (1 - jnp.asarray(b2, m2.dtype) ** (i + 1))
    noise_factors_log_std = noise_factors_log_std + lr * mhat / (jnp.sqrt(vhat) + eps)

    state = (state1, state2)

    return state, noise_factors_mean, noise_factors_log_std


@jax.jit
def cellnoise_node_grad(
        rngs,
        cell_noise_mean,
        cell_noise_log_std,
        data_node_weights,
        child_unobserved_samples,
        noise_factor_samples,
        baseline_samples,
        cnv,
        lib_sizes,
        data,
        mask,):
    def local_loss(
        params,
        child_unobserved_samples,
        noise_factor_samples,
        baseline_samples,
        ):
        cell_noise_mean, cell_noise_log_std = params[0], params[1]
        cell_noise_samples = sample_noise_factors(rngs, cell_noise_mean, cell_noise_log_std)

        loss = jnp.sum(ell(data_node_weights,
            baseline_samples,
            cell_noise_samples,
            noise_factor_samples,
            child_unobserved_samples,
            cnv,
            lib_sizes,
            data,
            mask,))
        return loss

    params = jnp.array([cell_noise_mean, cell_noise_log_std])
    grads = jnp.array(jax.grad(local_loss)(params, child_unobserved_samples, noise_factor_samples, baseline_samples,))
    return grads

@jax.jit
def cellnoise_kl_grad(mean, log_std):
    def _kl(mean, log_std):
        return jnp.sum(cell_noise_kl(mean, log_std))
    return jnp.array(jax.grad(_kl, argnums=(0,1))(mean, log_std))

@jax.jit
def cellnoise_step(cell_noise_mean, cell_noise_log_std, grads, states, i, lr=0.01):
    state1, state2 = states

    m1, v1 = state1
    b1=0.9
    b2=0.999
    eps=1e-8
    m1 = (1 - b1) * grads[0] + b1 * m1  # First  moment estimate.
    v1 = (1 - b2) * jnp.square(grads[0]) + b2 * v1  # Second moment estimate.
    state1 = (m1, v1)
    mhat = m1 / (1 - jnp.asarray(b1, m1.dtype) ** (i + 1))  # Bias correction.
    vhat = v1 / (1 - jnp.asarray(b2, m1.dtype) ** (i + 1))
    cell_noise_mean = cell_noise_mean + lr * mhat / (jnp.sqrt(vhat) + eps)

    m2, v2 = state2
    m2 = (1 - b1) * grads[1] + b1 * m2  # First  moment estimate.
    v2 = (1 - b2) * jnp.square(grads[1]) + b2 * v2  # Second moment estimate.
    state2 = (m2, v2)
    mhat = m2 / (1 - jnp.asarray(b1, m2.dtype) ** (i + 1))  # Bias correction.
    vhat = v2 / (1 - jnp.asarray(b2, m2.dtype) ** (i + 1))
    cell_noise_log_std = cell_noise_log_std + lr * mhat / (jnp.sqrt(vhat) + eps)

    state = (state1, state2)

    return state, cell_noise_mean, cell_noise_log_std


#
# @jax.jit
# def update_global_parameters(
#     hyperparams, # [concentration, rate]
#     child_in_root_subtree, # to make the prior prefer amplifications
#     cnv,
#     lib_sizes,
#     data,
#     parent_unobserved_sample,
#     child_unobserved_sample,
#     child_unobserved_kernel_sample,
#     ):
#     def local_loss(local_params):
#         return ell(local_params) + baseline_kl(baseline)
#         return loss
#
#
#     return child_unobserved_means, child_unobserved_log_stds, child_log_unobserved_factors_kernel_means, child_log_unobserved_factors_kernel_log_stds

#
# def tree_traversal_compute_elbo(rng, mc_samples=3):
#     """
#     This function traverses the tree starting at the root and computes the
#     complete ELBO
#     """
#     rngs = random.split(rng, mc_samples)
#
#     # Get data
#     data = self.data
#     lib_sizes = self.root["node"].root["node"].lib_sizes
#     n_cells, n_genes = self.data.shape
#
#     # Get global variational parameters
#     log_baseline_mean = self.root["node"].root["node"].variational_parameters["globals"]["log_baseline_mean"]
#     log_baseline_log_std = self.root["node"].root["node"].variational_parameters["globals"]["log_baseline_log_std"]
#     cell_noise_mean = self.root["node"].root["node"].variational_parameters["globals"]["cell_noise_mean"]
#     cell_noise_log_std = self.root["node"].root["node"].variational_parameters["globals"]["cell_noise_log_std"]
#     noise_factors_mean = self.root["node"].root["node"].variational_parameters["globals"]["noise_factors_mean"]
#     noise_factors_log_std = self.root["node"].root["node"].variational_parameters["globals"]["noise_factors_log_std"]
#
#     # Sample global
#     baseline_samples = sample_baseline(rngs, log_baseline_mean, log_baseline_log_std)
#     cell_noise_factors_samples = sample_cell_noise_factors(rngs, cell_noise_mean, cell_noise_log_std)
#     noise_factors_samples = sample_noise_factors(rng, noise_factors_mean, noise_factors_log_std)
#     noise_samples = cell_noise_factors_samples.dot(noise_factors_samples)
#
#     # Traverse tree
#     def descend(root):
#         total_subtree_weight = 0
#         indices = list(range(len(root["children"])))
#         # indices = indices[::-1]
#
#         parent_unobserved_means = root["node"].variational_parameters["locals"]["unobserved_factors_mean"]
#         parent_unobserved_log_stds = root["node"].variational_parameters["locals"]["unobserved_factors_log_std"]
#         # Sample parent
#         parent_unobserved_samples = sample_unobserved(rngs, parent_unobserved_means, parent_unobserved_log_stds)
#         psi_not_prev_sum = 0
#         for i in indices:
#             child = root["children"][i]
#             cnv = child.cnvs * jnp.ones((n_cells,n_genes))
#             data_node_weights = child.data_weights
#
#             # Sample node
#             unobserved_samples = sample_unobserved(rngs, unobserved_means, unobserved_log_stds)
#             unobserved_kernel_samples = sample_unobserved_kernel(rngs, unobserved_factors_kernel_log_means, unobserved_factors_kernel_log_stds)
#
#             # Compute node ll
#             unobserved_samples = unobserved_samples * jnp.ones((mc_samples, n_cells, n_genes)) # broadcast
#             unobserved_kernel_samples = unobserved_kernel_samples * jnp.ones((mc_samples, n_cells, n_genes)) # broadcast
#             ll = ell(data_node_weights, baseline_samples, noise_samples, unobserved_samples, cnv, lib_sizes, data)
#             child.ell = ll
#             child.param_kl = local_paramkl(unobserved_samples, unobserved_kernel_samples, parent_unobserved_samples)
#
#             E_log_phi = E_q_log_beta(psi_alpha, psi_beta) + psi_not_prev_sum
#             E_log_nu = E_q_log_beta(nu_alpha, nu_beta)
#             E_log_1_nu = E_q_log_1_beta(nu_alpha, nu_beta)
#
#             child.weight_until_here = child.data_weights + root.weight_until_here
#             child.expected_weights = child.data_weights*E_log_nu + root.weight_until_here*E_log_1_nu + child.weight_until_here*E_log_phi
#             total_weight += child.data_weights
#
#             if i > 0:
#                 psi_not_prev_sum += E_q_log_1_beta(psi_alpha, psi_beta)
#
#         node.stick_kl = beta_kl(root["node"].variational_parameters["locals"]["nu_log_mean"], 1, root["node"].variational_parameters["locals"]["nu_log_mean"], root.tssb.dp_alpha)
#         node.stick_kl += beta_kl(root["node"].variational_parameters["locals"]["psi_log_mean"], 1, root["node"].variational_parameters["locals"]["psi_log_mean"], root.tssb.dp_gamma)
#
#         expected_weight, nu_sticks_sum, psi_sticks_sum = compute_expected_weight(nu_alpha, nu_beta, psi_alpha, psi_beta, prev_nu_sticks_sum, prev_psi_sticks_sum)
#         node.ew = expected_weight
#
#         return total_weight, lls, expected_weight, nodes
#
#     _, lls, expected_w, nodes = descend(self.root)
#
#     # Normalize data_node_weights and compute local ELBO contributions
#     data_weights = []
#     for node in nodes:
#         data_weights.append(node.data_weights)
#     data_weights = np.array(data_weights).reshape(-1,mb_size)/np.sum(data_weights)
#     elbo = 0
#     for i, node in nodes:
#         node.data_weights = data_weights[data_indices,i]
#         # Compute ELBO using normalized data_weights
#         node_data_elbo_contributions = node.data_weights*node.ell + node.ew - node.data_weights*np.log(node.data_weights))
#         node_kl = node.stick_kl + node.param_kl
#         elbo += node_data_elbo_contributions - node_kl
#
#     return elbo
#
#
# def tree_traversal_update(root, rng, update_global=True, n_inner_steps=10, mc_samples=3):
#     """
#     This function traverses the tree starting at `root` and updates the
#     variational parameters and ELBO contributions while doing it
#     """
#     rngs = random.split(rng, mc_samples)
#
#     # Get data
#     data = self.data
#     lib_sizes = self.root["node"].root["node"].lib_sizes
#     n_cells, n_genes = self.data.shape
#
#     # Get global variational parameters
#     log_baseline_mean = self.root["node"].root["node"].variational_parameters["globals"]["log_baseline_mean"]
#     log_baseline_log_std = self.root["node"].root["node"].variational_parameters["globals"]["log_baseline_log_std"]
#     cell_noise_mean = self.root["node"].root["node"].variational_parameters["globals"]["cell_noise_mean"]
#     cell_noise_log_std = self.root["node"].root["node"].variational_parameters["globals"]["cell_noise_log_std"]
#     noise_factors_mean = self.root["node"].root["node"].variational_parameters["globals"]["noise_factors_mean"]
#     noise_factors_log_std = self.root["node"].root["node"].variational_parameters["globals"]["noise_factors_log_std"]
#
#     # Sample global
#     baseline_samples = sample_baseline(rngs, log_baseline_mean, log_baseline_log_std)
#     cell_noise_factors_samples = sample_cell_noise_factors(rngs, cell_noise_mean, cell_noise_log_std)
#     noise_factors_samples = sample_noise_factors(rng, noise_factors_mean, noise_factors_log_std)
#     noise_samples = cell_noise_factors_samples.dot(noise_factors_samples)
#
#     # Traverse tree and update variational parameters
#     def descend(root, depth=0):
#         weight_down = 0
#         indices = list(range(len(root["children"])))
#         indices = indices[::-1]
#
#         parent_unobserved_means = root["node"].variational_parameters["locals"]["unobserved_factors_mean"]
#         parent_unobserved_log_stds = root["node"].variational_parameters["locals"]["unobserved_factors_log_std"]
#         for i in indices:
#             child = root["children"][i]
#             cnv = child.cnvs * jnp.ones((n_cells,n_genes))
#             data_node_weights = child.data_weights
#
#             # Sample parent
#             parent_unobserved_samples = sample_unobserved(rngs, parent_unobserved_means, parent_unobserved_log_stds)
#
#             # Update local parameters
#             unobserved_means = root["node"].variational_parameters["locals"]["unobserved_factors_mean"]
#             unobserved_log_stds = root["node"].variational_parameters["locals"]["unobserved_factors_log_std"]
#             unobserved_factors_kernel_log_mean = root["node"].variational_parameters["locals"]["unobserved_factors_kernel_log_mean"]
#             unobserved_factors_kernel_log_std = root["node"].variational_parameters["locals"]["unobserved_factors_kernel_log_std"]
#             update_local_parameters(baseline_samples, noise_samples, parent_unobserved_samples, steps=n_inner_steps)
#
#             # Sample updated node
#             unobserved_samples = sample_unobserved(rngs, unobserved_means, unobserved_log_stds)
#             unobserved_kernel_samples = sample_unobserved_kernel(rngs, unobserved_factors_kernel_log_means, unobserved_factors_kernel_log_stds)
#
#             # Compute node ll
#             unobserved_samples = unobserved_samples * jnp.ones((mc_samples, n_cells, n_genes)) # broadcast
#             unobserved_kernel_samples = unobserved_kernel_samples * jnp.ones((mc_samples, n_cells, n_genes)) # broadcast
#             ll = ell(data_node_weights, baseline_samples, noise_samples, unobserved_samples, cnv, lib_sizes, data)
#             child.ell[data_indices] = ll
#             child.param_kl = local_paramkl(unobserved_samples, unobserved_kernel_samples, parent_unobserved_samples)
#
#             child_weight, _, _, _ = descend(child, depth + 1)
#             post_alpha = 1.0 + child_weight
#             post_beta = self.dp_gamma + weight_down
#             child["node"].variational_parameters["locals"]["psi_log_mean"] = np.log(post_alpha)
#             child["node"].variational_parameters["locals"]["psi_log_std"] = np.log(post_beta)
#             weight_down += child_weight
#
#             prev_psi_sticks_sum += local_psi_sticks_sum(psi_alpha, psi_beta)
#
#         weight_here = np.sum(root["node"].data_weights)
#         total_weight = weight_here + weight_down
#         post_alpha = 1.0 + weight_here
#         post_beta = (self.alpha_decay**depth) * self.dp_alpha + weight_down
#         root["node"].variational_parameters["locals"]["nu_log_mean"] = np.log(post_alpha)
#         root["node"].variational_parameters["locals"]["nu_log_std"] = np.log(post_beta)
#
#         node.stick_kl = beta_kl(root["node"].variational_parameters["locals"]["nu_log_mean"], root["node"].variational_parameters["locals"]["nu_log_mean"])
#         node.stick_kl += beta_kl(root["node"].variational_parameters["locals"]["psi_log_mean"], root["node"].variational_parameters["locals"]["psi_log_mean"])
#
#         expected_weight, nu_sticks_sum, psi_sticks_sum = compute_expected_weight(nu_alpha, nu_beta, psi_alpha, psi_beta, prev_nu_sticks_sum, prev_psi_sticks_sum)
#         node.ew = expected_weight
#         node.data_weights[data_indices] = np.exp(node.ell + node.ew)
#
#         return total_weight, lls, expected_weight, nodes
#
#     _, lls, expected_w, nodes = descend(root)
#
#     # Update global parameters
#     if root.parent() is None and update_global:
#         # Use samples from tree traversal to update global parameters
#         update_global_parameters(steps=n_inner_steps)
#
#     # Compute global KL
#     global_kl = baseline_kl(log_baseline_mean, log_baseline_log_std)
#
#     # Use lls from tree traversal to update data_node_weights
#     data_weights = []
#     for node in nodes:
#         data_weights.append(node.data_weights)
#     data_weights = np.array(data_weights).reshape(-1,mb_size)/np.sum(data_weights)
#     for i, node in nodes:
#         node.data_weights[data_indices] = data_weights[data_indices,i]
#         # Compute ELBO using normalized data_weights
#         node_data_elbo_contributions = node.data_weights*(node.ell + node.ew - np.log(node.data_weights))
#         node_elbo_contributions = node.stick_kl + node.param_kl
#
#     # Compute total elbo: use decomposibility!
#     elbo =
#
#     return elbo


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

        # log_phi = jnp.log(psi_stick) + jnp.sum(
        #     vmap(prev_branches_psi)(previous_branches_indices[i])
        # )
        # log_node_weight = (
        #     jnp.log(nu_stick)
        #     + log_phi
        #     + jnp.sum(vmap(ancestors_nu)(ancestor_nodes_indices[i]))
        # )
        # log_node_weight = log_node_weight + jnp.log(tssb_weights[i])
        # ll = ll + log_node_weight  # N-vector

        return ll

    small_ll = -1e30 * jnp.ones((mb_size))

    def get_node_ll(i):
        return jnp.where(
            node_mask[i] >= 0,
            compute_node_ll(jnp.where(node_mask[i] >= 0, i, 0)),
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
