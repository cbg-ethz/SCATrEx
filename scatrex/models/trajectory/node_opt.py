import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

@jax.jit
def sample_angle(key, mu, log_kappa): # univariate: one sample
    return tfd.VonMises(mu, jnp.exp(log_kappa)).sample(seed=key)
sample_angle_val_and_grad = jax.vmap(jax.value_and_grad(sample_angle, argnums=(1,2)), in_axes=(None, 0, 0)) # per-dimension val and grad
mc_sample_angle_val_and_grad = jax.jit(jax.vmap(sample_angle_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad

@jax.jit
def sample_loc(key, mu, log_std): # univariate: one sample
    return tfd.Normal(mu, jnp.exp(log_std)).sample(seed=key)
sample_loc_val_and_grad = jax.vmap(jax.value_and_grad(sample_loc, argnums=(1,2)), in_axes=(None, 0, 0)) # per-dimension val and grad
mc_sample_loc_val_and_grad = jax.jit(jax.vmap(sample_loc_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad

@jax.jit
def angle_logp(this_angle, parent_angle, concentration): # single sample
    return jnp.sum(tfd.VonMises(parent_angle, concentration).log_prob(this_angle))
angle_logp_val_and_grad = jax.jit(jax.value_and_grad(angle_logp, argnums=0)) # Take grad wrt to this
mc_angle_logp_val_and_grad = jax.jit(jax.vmap(angle_logp_val_and_grad, in_axes=(0,0, None))) # Multiple sample value_and_grad

angle_logp_val_and_grad_wrt_parent = jax.jit(jax.value_and_grad(angle_logp, argnums=1)) # Take grad wrt to parent
mc_angle_logp_val_and_grad_wrt_parent = jax.jit(jax.vmap(angle_logp_val_and_grad, in_axes=(0,0, None))) # Multiple sample value_and_grad

@jax.jit
def angle_logq(mu, log_kappa):
    return jnp.sum(tfd.VonMises(mu, jnp.exp(log_kappa)).entropy())
angle_logq_val_and_grad = jax.jit(jax.value_and_grad(angle_logq, argnums=(0,1))) # Take grad wrt to parameters

@jax.jit
def loc_logp(this_loc, parent_loc, this_angle, log_std, radius): # single sample
    mean = parent_loc + jnp.hstack([jnp.cos(this_angle)*radius, jnp.sin(this_angle)*radius]) # Use samples from parent
    return jnp.sum(tfd.Normal(mean, jnp.exp(log_std)).log_prob(this_loc)) # sum across dimensions
loc_logp_val = jax.jit(loc_logp) 
mc_loc_logp_val = jax.jit(jax.vmap(loc_logp_val, in_axes=(0,0,0, None, None))) # Multiple sample 

loc_logp_val_and_grad = jax.jit(jax.value_and_grad(loc_logp, argnums=0)) # Take grad wrt to this
mc_loc_logp_val_and_grad = jax.jit(jax.vmap(loc_logp_val_and_grad, in_axes=(0,0,0, None, None))) # Multiple sample value_and_grad

loc_logp_val_and_grad_wrt_parent = jax.jit(jax.value_and_grad(loc_logp, argnums=1)) # Take grad wrt to parent
mc_loc_logp_val_and_grad_wrt_parent = jax.jit(jax.vmap(loc_logp_val_and_grad_wrt_parent, in_axes=(0,0,0, None, None))) # Multiple sample value_and_grad

loc_logp_val_and_grad_wrt_angle = jax.jit(jax.value_and_grad(loc_logp, argnums=2)) # Take grad wrt to angle
mc_loc_logp_val_and_grad_wrt_angle = jax.jit(jax.vmap(loc_logp_val_and_grad_wrt_angle, in_axes=(0,0,0, None, None))) # Multiple sample value_and_grad

@jax.jit
def loc_logq(mu, log_std):
    return tfd.Normal(mu, jnp.exp(log_std)).entropy()
loc_logq_val_and_grad = jax.jit(jax.vmap(jax.value_and_grad(loc_logq, argnums=(0,1)), in_axes=(0,0))) # Take grad wrt to parameters




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
    return jnp.sum(tfd.Normal(mean, jnp.exp(log_std)).log_prob(sample)) # sum across obs and dimensions
obs_weights_logp_val_and_grad = jax.jit(jax.value_and_grad(obs_weights_logp, argnums=0)) # Take grad wrt to sample (NxK)
mc_obs_weights_logp_val_and_grad = jax.jit(jax.vmap(obs_weights_logp_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad: SxNxK

@jax.jit
def obs_weights_logq(mu, log_std):
    return tfd.Normal(mu, jnp.exp(log_std)).entropy()
obs_weights_logq_val_and_grad = jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(obs_weights_logq, argnums=(0,1)), in_axes=(0,0)), in_axes=(0,0))) # Take grad wrt to parameters


@jax.jit
def factor_weights_logp(sample, mean, log_std): # single sample, KxG
    return jnp.sum(tfd.Normal(mean, jnp.exp(log_std)).log_prob(sample)) # sum across factors and genes
factor_weights_logp_val_and_grad = jax.jit(jax.value_and_grad(factor_weights_logp, argnums=0)) # Take grad wrt to sample (KxG)
mc_factor_weights_logp_val_and_grad = jax.jit(jax.vmap(factor_weights_logp_val_and_grad, in_axes=(0,None,None))) # Multiple sample value_and_grad: SxKxG

@jax.jit
def factor_weights_logq(mu, log_std):
    return tfd.Normal(mu, jnp.exp(log_std)).entropy()
factor_weights_logq_val_and_grad = jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(factor_weights_logq, argnums=(0,1)), in_axes=(0,0)), in_axes=(0,0))) # Take grad wrt to parameters

@jax.jit
def _mc_obs_ll(obs, node_mean, obs_weights, factor_weights, std): # For each MC sample: Nx2
    m = node_mean + obs_weights.dot(factor_weights)
    return jnp.sum(jax.vmap(jax.scipy.stats.norm.logpdf, in_axes=[0, 0, None])(obs, m, std), axis=1) # sum over dimensions

@jax.jit
def ll(x, weights, psi, obs_weights, factor_weights, log_std): # single sample
    loc = psi + obs_weights.dot(factor_weights)
    return jnp.sum(jnp.sum(tfd.Normal(loc, jnp.exp(log_std)).log_prob(x),axis=1) * weights)
ll_val_and_grad_psi = jax.jit(jax.value_and_grad(ll, argnums=2)) # Take grad wrt to psi
mc_ll_val_and_grad_psi = jax.jit(jax.vmap(ll_val_and_grad_psi, 
                                          in_axes=(None,None,0,0,0,None)))

ll_val_and_grad_obs_weights = jax.jit(jax.value_and_grad(ll, argnums=3)) # Take grad wrt to obs_weights
mc_ll_val_and_grad_obs_weights = jax.jit(jax.vmap(ll_val_and_grad_obs_weights, 
                                                  in_axes=(None,None,0,0,0,None)))

ll_val_and_grad_factor_weights = jax.jit(jax.value_and_grad(ll, argnums=4)) # Take grad wrt to factor_weights
mc_ll_val_and_grad_factor_weights = jax.jit(jax.vmap(ll_val_and_grad_factor_weights, 
                                                     in_axes=(None,None,0,0,0,None)))

@jax.jit
def ll_suffstats(node_mean, mass, A, B_g, C, D_g, E, std): # for a single node_mean sample
    """
    mass: \sum_n q(z_n = this node)
    A: \sum_n q(z_n = this node) * \sum_g x_ng ** 2
    B_g: \sum_n q(z_n = this node) * x_ng
    C: \sum_n q(z_n = this node) * \sum_g x_ng * E[s_nW_g]
    D_g: \sum_n q(z_n = this node) * E[s_nW_g]
    E: \sum_n q(z_n = this node) * \sum_g E[(s_nW_g)**2]
    """
    v = std**2
    ll = -jnp.log(2*jnp.pi*v) * mass - A/(2*v) + jnp.sum(B_g*node_mean)/v + C/v - \
        (mass*jnp.sum(node_mean**2))/(2*v) - jnp.sum(D_g*node_mean)/v - E/(2*v)
    return ll

@jax.jit
def ll_node_mean_suff(node_mean, mass, B_g, D_g, log_std): # for a single node_mean sample
    """
    mass: \sum_n q(z_n = this node)
    B_g: \sum_n q(z_n = this node) * x_ng
    D_g: \sum_n q(z_n = this node) * E[s_nW_g]
    """
    v = jnp.exp(log_std)**2
    ll = jnp.sum(B_g*node_mean)/v - (mass*jnp.sum(node_mean**2))/(2*v) - jnp.sum(D_g*node_mean)/v
    return ll
ll_node_mean_suff_val_and_grad = jax.jit(jax.value_and_grad(ll_node_mean_suff, argnums=0)) # Take grad wrt to psi
mc_ll_node_mean_suff_val_and_grad = jax.jit(jax.vmap(ll_node_mean_suff_val_and_grad, 
                                          in_axes=(0,None, None, None, None)))

# To get noise sample
sample_prod = jax.jit(lambda mat1, mat2: mat1.dot(mat2))