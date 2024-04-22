import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

@jax.jit
def sample_direction(key, log_alpha, log_beta): # univariate: one sample
    print("haahahdirection")
    return jnp.maximum(jnp.exp(tfd.ExpGamma(jnp.exp(log_alpha), log_rate=log_beta).sample(seed=key)), 1e-6)
sample_direction_val_and_grad = jax.jit(jax.vmap(jax.value_and_grad(sample_direction, argnums=(1,2)), in_axes=(0, 0, 0))) # per-dimension val and grad
@jax.jit
def _mc_sample_direction_val_and_grad(key, mus, logs):
    keys = jax.random.split(key, mus.size)
    return sample_direction_val_and_grad(jnp.array(keys), mus, logs)
mc_sample_direction_val_and_grad = jax.jit(jax.vmap(_mc_sample_direction_val_and_grad, in_axes=(0,None,None)))

@jax.jit
def sample_state(key, mu, log_std): # univariate: one sample
    print("haahahstate")
    return tfd.Normal(mu, jnp.exp(log_std)).sample(seed=key)
sample_state_val_and_grad = jax.jit(jax.vmap(jax.value_and_grad(sample_state, argnums=(1,2)), in_axes=(0, 0, 0))) # per-dimension val and grad
@jax.jit
def _mc_sample_state_val_and_grad(key, mus, logs):
    keys = jax.random.split(key, mus.size)
    return sample_state_val_and_grad(jnp.array(keys), mus, logs)
mc_sample_state_val_and_grad = jax.jit(jax.vmap(_mc_sample_state_val_and_grad, in_axes=(0,None,None)))

@jax.jit
def direction_logp(this_direction, parent_state, direction_shape, inheritance_strength): # single sample
    return tfd.Gamma(direction_shape, log_rate=inheritance_strength*jnp.abs(parent_state)).log_prob(this_direction)
univ_direction_logp_val_and_grad = jax.jit(jax.value_and_grad(direction_logp, argnums=0)) # Take grad wrt to this
direction_logp_val_and_grad = jax.jit(jax.vmap(univ_direction_logp_val_and_grad, in_axes=(0,0,None,None))) # Take grad wrt to this
mc_direction_logp_val_and_grad = jax.jit(jax.vmap(direction_logp_val_and_grad, in_axes=(0,0,None,None))) # Multiple sample value_and_grad

univ_direction_logp_val_and_grad_wrt_parent = jax.jit(jax.value_and_grad(direction_logp, argnums=1)) # Take grad wrt to parent
direction_logp_val_and_grad_wrt_parent = jax.jit(jax.vmap(univ_direction_logp_val_and_grad_wrt_parent, in_axes=(0,0,None,None))) # Multiple sample value_and_grad
mc_direction_logp_val_and_grad_wrt_parent = jax.jit(jax.vmap(direction_logp_val_and_grad_wrt_parent, in_axes=(0,0,None,None))) # Multiple sample value_and_grad

@jax.jit
def direction_logq(log_alpha, log_beta):
    return tfd.Gamma(jnp.exp(log_alpha), log_rate=log_beta).entropy()
direction_logq_val_and_grad = jax.jit(jax.vmap(jax.value_and_grad(direction_logq, argnums=(0,1)), in_axes=(0,0))) # Take grad wrt to parameters

@jax.jit
def state_logp(this_state, parent_state, this_direction): # single sample
    return tfd.Normal(parent_state, this_direction).log_prob(this_state) # sum across dimensions
state_logp_val = jax.jit(state_logp) 
mc_loc_logp_val = jax.jit(jax.vmap(state_logp_val, in_axes=(0,0,0))) # Multiple sample 

univ_state_logp_val_and_grad = jax.jit(jax.value_and_grad(state_logp, argnums=0)) # Take grad wrt to this
state_logp_val_and_grad = jax.jit(jax.vmap(univ_state_logp_val_and_grad, in_axes=(0,0,0))) # Take grad wrt to this
mc_state_logp_val_and_grad = jax.jit(jax.vmap(state_logp_val_and_grad, in_axes=(0,0,0))) # Multiple sample value_and_grad

univ_state_logp_val_and_grad_wrt_parent = jax.jit(jax.value_and_grad(state_logp, argnums=1)) # Take grad wrt to parent
state_logp_val_and_grad_wrt_parent = jax.jit(jax.vmap(univ_state_logp_val_and_grad_wrt_parent, in_axes=(0,0,0))) # Take grad wrt to parent
mc_state_logp_val_and_grad_wrt_parent = jax.jit(jax.vmap(state_logp_val_and_grad_wrt_parent, in_axes=(0,0,0))) # Multiple sample value_and_grad

univ_state_logp_val_and_grad_wrt_direction = jax.jit(jax.value_and_grad(state_logp, argnums=2)) # Take grad wrt to angle
state_logp_val_and_grad_wrt_direction = jax.jit(jax.vmap(univ_state_logp_val_and_grad_wrt_direction, in_axes=(0,0,0))) # Take grad wrt to angle
mc_state_logp_val_and_grad_wrt_direction = jax.jit(jax.vmap(state_logp_val_and_grad_wrt_direction, in_axes=(0,0,0))) # Multiple sample value_and_grad

@jax.jit
def state_logq(mu, log_std):
    return tfd.Normal(mu, jnp.exp(log_std)).entropy()
state_logq_val_and_grad = jax.jit(jax.vmap(jax.value_and_grad(state_logq, argnums=(0,1)), in_axes=(0,0))) # Take grad wrt to parameters


# Noise

@jax.jit
def sample_obs_weights(key, mean, log_std): # NxK
    return tfd.Normal(mean, jnp.exp(log_std)).sample(seed=key)
sample_obs_weights_val_and_grad = jax.vmap(jax.vmap(jax.value_and_grad(sample_obs_weights, argnums=(1,2)), in_axes=(None,0,0)), in_axes=(None,0,0)) # per-dimension val and grad
mc_sample_obs_weights_val_and_grad = jax.jit(jax.vmap(sample_obs_weights_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad

@jax.jit
def sample_factor_weights(key, mu, log_std): # KxG
    return tfd.Normal(mu, jnp.exp(log_std)).sample(seed=key)
sample_factor_weights_val_and_grad = jax.vmap(jax.vmap(jax.value_and_grad(sample_factor_weights, argnums=(1,2)), in_axes=(None,0,0)), in_axes=(None,0,0)) # per-dimension val and grad
mc_sample_factor_weights_val_and_grad = jax.jit(jax.vmap(sample_factor_weights_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad


@jax.jit
def obs_weights_logp(sample, mean, log_std): # single sample, NxK
    return tfd.Normal(mean, jnp.exp(log_std)).log_prob(sample) # sum across obs and dimensions
univ_obs_weights_logp_val_and_grad = jax.jit(jax.value_and_grad(obs_weights_logp, argnums=0)) # Take grad wrt to sample (Nx1)
obs_weights_logp_val_and_grad = jax.jit(jax.vmap(jax.vmap(univ_obs_weights_logp_val_and_grad, in_axes=(0, None, None)), in_axes=(0,None,None))) # Take grad wrt to sample (NxK)
mc_obs_weights_logp_val_and_grad = jax.jit(jax.vmap(obs_weights_logp_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad: SxNxK

@jax.jit
def obs_weights_logq(mu, log_std):
    return tfd.Normal(mu, jnp.exp(log_std)).entropy()
obs_weights_logq_val_and_grad = jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(obs_weights_logq, argnums=(0,1)), in_axes=(0,0)), in_axes=(0,0))) # Take grad wrt to parameters


@jax.jit
def factor_weights_logp(sample, mean, precision): # single sample, KxG
    return jnp.sum(tfd.Normal(mean, 1./jnp.sqrt(precision)).log_prob(sample)) # sum over 1
univ_factor_weights_logp_val_and_grad = jax.jit(jax.value_and_grad(factor_weights_logp, argnums=0))
factor_weights_logp_val_and_grad = jax.jit(jax.vmap(jax.vmap(univ_factor_weights_logp_val_and_grad, in_axes=(0,None,None)), in_axes=(0,None,0))) # Take grad wrt to sample (KxG)
mc_factor_weights_logp_val_and_grad = jax.jit(jax.vmap(factor_weights_logp_val_and_grad, in_axes=(0,None,0))) # Multiple sample value_and_grad: SxKxG

@jax.jit
def factor_weights_logp_summed(sample, mean, precision): # single sample, KxG
    return jnp.sum(tfd.Normal(mean, 1./jnp.sqrt(precision)).log_prob(sample)) # sum over genes
factor_weights_logp_val_and_grad_wrt_precisions = jax.jit(jax.value_and_grad(factor_weights_logp_summed, argnums=2))
mc_factor_weights_logp_val_and_grad_wrt_precisions = jax.jit(jax.vmap(factor_weights_logp_val_and_grad_wrt_precisions, in_axes=(0,None,0))) # Multiple sample value_and_grad: SxKxG

@jax.jit
def factor_weights_logq(mu, log_std):
    return tfd.Normal(mu, jnp.exp(log_std)).entropy()
factor_weights_logq_val_and_grad = jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(factor_weights_logq, argnums=(0,1)), in_axes=(0,0)), in_axes=(0,0))) # Take grad wrt to parameters


# Cell scales
@jax.jit
def sample_cell_scales(key, log_alpha, log_beta): # Nx1
    return jnp.exp(tfd.ExpGamma(jnp.exp(log_alpha), jnp.exp(log_beta)).sample(seed=key))
sample_cell_scales_val_and_grad = jax.vmap(jax.vmap(jax.value_and_grad(sample_cell_scales, argnums=(1,2)), in_axes=(None,0,0)), in_axes=(None,0,0)) # per-dimension val and grad
mc_sample_cell_scales_val_and_grad = jax.jit(jax.vmap(sample_cell_scales_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad

@jax.jit
def cell_scales_logp(sample, log_alpha, log_beta): # single sample, Nx1
    return jnp.sum(tfd.Gamma(jnp.exp(log_alpha), jnp.exp(log_beta)).log_prob(sample)) # sum across obs and dimensions
univ_cell_scales_logp_val_and_grad = jax.jit(jax.value_and_grad(cell_scales_logp, argnums=0)) # Take grad wrt to sample (Nx1)
cell_scales_logp_val_and_grad = jax.jit(jax.vmap(univ_cell_scales_logp_val_and_grad, in_axes=(0,None,None))) # Take grad wrt to sample (Nx1)
mc_cell_scales_logp_val_and_grad = jax.jit(jax.vmap(cell_scales_logp_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad: SxNx1

@jax.jit
def cell_scales_logq(log_alpha, log_beta):
    return tfd.Gamma(jnp.exp(log_alpha), jnp.exp(log_beta)).entropy()
cell_scales_logq_val_and_grad = jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(cell_scales_logq, argnums=(0,1)), in_axes=(0,0)), in_axes=(0,0))) # Take grad wrt to parameters

# Gene scales
@jax.jit
def sample_gene_scales(key, log_alpha, log_beta): # G
    return jnp.exp(tfd.ExpGamma(jnp.exp(log_alpha), jnp.exp(log_beta)).sample(seed=key))
sample_gene_scales_val_and_grad = jax.vmap(jax.value_and_grad(sample_gene_scales, argnums=(1,2)), in_axes=(None,0,0)) # per-dimension val and grad
mc_sample_gene_scales_val_and_grad = jax.jit(jax.vmap(sample_gene_scales_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad

@jax.jit
def gene_scales_logp(sample, log_alpha, log_beta): # single sample
    return tfd.Gamma(jnp.exp(log_alpha), jnp.exp(log_beta)).log_prob(sample) # sum across obs and dimensions
univ_gene_scales_logp_val_and_grad = jax.jit(jax.value_and_grad(gene_scales_logp, argnums=0)) # Take grad wrt to sample (G,)
gene_scales_logp_val_and_grad = jax.jit(jax.vmap(univ_gene_scales_logp_val_and_grad, in_axes=(0,None,None))) # Take grad wrt to sample (G,)
mc_gene_scales_logp_val_and_grad = jax.jit(jax.vmap(gene_scales_logp_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad: SxG

@jax.jit
def gene_scales_logq(log_alpha, log_beta):
    return tfd.Gamma(jnp.exp(log_alpha), jnp.exp(log_beta)).entropy()
gene_scales_logq_val_and_grad = jax.jit(jax.vmap(jax.value_and_grad(gene_scales_logq, argnums=(0,1)), in_axes=(0,0))) # Take grad wrt to parameters


# Factor variances
@jax.jit
def sample_factor_precisions(key, log_alpha, log_beta): # Kx1
    return jnp.exp(tfd.ExpGamma(jnp.exp(log_alpha), jnp.exp(log_beta)).sample(seed=key))
sample_factor_precisions_val_and_grad = jax.vmap(jax.vmap(jax.value_and_grad(sample_factor_precisions, argnums=(1,2)), in_axes=(None,0,0)), in_axes=(None,0,0)) # per-dimension val and grad
mc_sample_factor_precisions_val_and_grad = jax.jit(jax.vmap(sample_factor_precisions_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad

@jax.jit
def factor_precisions_logp(sample, log_alpha, log_beta): # single sample
    return jnp.sum(tfd.Gamma(jnp.exp(log_alpha), jnp.exp(log_beta)).log_prob(sample)) # sum across obs and dimensions
univ_factor_precisions_logp_val_and_grad = jax.jit(jax.value_and_grad(factor_precisions_logp, argnums=0)) # Take grad wrt to sample (G,)
factor_precisions_logp_val_and_grad = jax.jit(jax.vmap(univ_factor_precisions_logp_val_and_grad, in_axes=(0,None,None))) # Take grad wrt to sample (G,)
mc_factor_precisions_logp_val_and_grad = jax.jit(jax.vmap(factor_precisions_logp_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad: SxG

@jax.jit
def factor_precisions_logq(log_alpha, log_beta):
    return tfd.Gamma(jnp.exp(log_alpha), jnp.exp(log_beta)).entropy()
factor_precisions_logq_val_and_grad = jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(factor_precisions_logq, argnums=(0,1)), in_axes=(0,0)), in_axes=(0,0))) # Take grad wrt to parameters


@jax.jit
def _mc_obs_ll(obs, state, cnv, obs_weights, factor_weights, cell_scales, gene_scales): # For each MC sample: NxG
    m = cell_scales * gene_scales * cnv/2 * jnp.exp(state + obs_weights.dot(factor_weights))
    return jnp.sum(jax.vmap(jax.scipy.stats.poisson.logpmf, in_axes=[0, 0])(obs, m), axis=1) # sum over dimensions

@jax.jit
def ll(x, weights, state, cnv, obs_weights, factor_weights, cell_scales, gene_scales): # single sample
    loc = cell_scales * gene_scales * cnv/2 * jnp.exp(state + obs_weights.dot(factor_weights))
    return jnp.sum(jnp.sum(tfd.Poisson(loc).log_prob(x),axis=1) * weights)
ll_val_and_grad_state = jax.jit(jax.value_and_grad(ll, argnums=2)) # Take grad wrt to psi
mc_ll_val_and_grad_state = jax.jit(jax.vmap(ll_val_and_grad_state, 
                                          in_axes=(None,None,0,None,0,0,0,0)))

ll_val_and_grad_factor_weights = jax.jit(jax.value_and_grad(ll, argnums=5)) # Take grad wrt to factor_weights
mc_ll_val_and_grad_factor_weights = jax.jit(jax.vmap(ll_val_and_grad_factor_weights, 
                                                     in_axes=(None,None,0,None,0,0,0,0)))

ll_val_and_grad_cell_scales = jax.jit(jax.value_and_grad(ll, argnums=6)) # Take grad wrt to cell_scales
mc_ll_val_and_grad_cell_scales = jax.jit(jax.vmap(ll_val_and_grad_cell_scales, 
                                                     in_axes=(None,None,0,None,0,0,0,0)))

ll_val_and_grad_gene_scales = jax.jit(jax.value_and_grad(ll, argnums=7)) # Take grad wrt to gene_scales
mc_ll_val_and_grad_gene_scales = jax.jit(jax.vmap(ll_val_and_grad_gene_scales, 
                                                     in_axes=(None,None,0,None,0,0,0,0)))


@jax.jit
def ll_obs(x, weight, state, cnv, obs_weights, factor_weights, cell_scales, gene_scales): # single obs
    loc = cell_scales * gene_scales * cnv/2 * jnp.exp(state + obs_weights.dot(factor_weights))
    return jnp.sum(tfd.Poisson(loc).log_prob(x)) * weight

univ_ll_val_and_grad_obs_weights = jax.jit(jax.value_and_grad(ll_obs, argnums=4)) # Take grad wrt to obs_weights
ll_val_and_grad_obs_weights = jax.jit(jax.vmap(univ_ll_val_and_grad_obs_weights, in_axes=(0,0, None, None, 0, None, 0, None)))
mc_ll_val_and_grad_obs_weights = jax.jit(jax.vmap(ll_val_and_grad_obs_weights, 
                                                  in_axes=(None,None,0,None,0,0,0,0)))

univ_ll_val_and_grad_cell_scales = jax.jit(jax.value_and_grad(ll_obs, argnums=6)) # Take grad wrt to cell_scales
ll_val_and_grad_cell_scales = jax.jit(jax.vmap(univ_ll_val_and_grad_cell_scales, in_axes=(0,0, None, None, 0, None, 0, None)))
mc_ll_val_and_grad_cell_scales = jax.jit(jax.vmap(ll_val_and_grad_cell_scales, 
                                                  in_axes=(None,None,0,None,0,0,0,0)))

@jax.jit
def ll_suffstats(state, cnv, gene_scales, A, B_g, C, D_g, E): # for a single state sample
    """
    A: \sum_n q(z_n = this node) * \sum_g x_ng * E[log\gamma_n]
    B_g: \sum_n q(z_n = this node) * x_ng
    C: \sum_n q(z_n = this node) * \sum_g x_ng * E[(s_nW_g)]
    D_g: \sum_n q(z_n = this node) * E[\gamma_n] * E[exp(s_nW_g)]
    E: \sum_n q(z_n = this node) * lgamma(x_ng+1)
    """
    ll = A + jnp.sum(B_g * (jnp.log(gene_scales) + jnp.log(cnv/2) + state)) + \
          C - jnp.sum(gene_scales * cnv/2 * jnp.exp(state) * D_g) - E
    return ll

@jax.jit
def ll_state_suff(state, cnv, gene_scales, B_g, D_g): # for a single state sample
    """
    B_g: \sum_n q(z_n = this node) * x_ng
    D_g: \sum_n q(z_n = this node) * E[\gamma_n] * E[s_nW_g]
    """
    ll = jnp.sum(B_g * state) - jnp.sum(gene_scales * cnv/2 * jnp.exp(state) * D_g)
    return ll

ll_state_suff_val_and_grad = jax.jit(jax.value_and_grad(ll_state_suff, argnums=0)) # Take grad wrt to psi
mc_ll_state_suff_val_and_grad = jax.jit(jax.vmap(ll_state_suff_val_and_grad, 
                                          in_axes=(0,None,0,None, None)))

# To get noise sample
sample_prod = jax.jit(lambda mat1, mat2: mat1.dot(mat2))