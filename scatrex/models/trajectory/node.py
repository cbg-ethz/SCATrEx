from numpy import *
import numpy as np
from numpy.random import *

from functools import partial
import jax.numpy as jnp
import jax
import tensorflow_probability.substrates.jax.distributions as tfd

from .node_opt import * # node optimization functions
from .node_opt import _mc_obs_ll
from ...utils.math_utils import *
from ...ntssb.node import *

def update_params(params, params_gradient, step_size):
    new_params = []
    for i, param in enumerate(params):
        new_params.append(param + step_size * params_gradient[i])
    return new_params

class TrajectoryNode(AbstractNode):
    def __init__(
        self,
        observed_parameters, # subtree root location and angle
        root_loc_mean=2.,
        loc_mean=.5,
        angle_concentration=10.,
        loc_variance=.1,
        obs_variance=.1,
        n_factors=2,
        obs_weight_variance=1.,
        factor_variance=1.,
        **kwargs,
    ):
        """
        This model generates nodes in a 2D space by sampling an angle, which roughly follows
        the angle at which the parent was generated, and a location around some radius.
        TODO: Make the location prior a Gamma instead of a Normal, still centered around some radius
        but with a peak at zero and a long tail, to allow for more or less close by nodes
        """
        super(TrajectoryNode, self).__init__(observed_parameters, **kwargs)

        self.n_genes = self.observed_parameters[0].size

        # Node hyperparameters
        if self.parent() is None:
            self.node_hyperparams = dict(
                root_loc_mean=root_loc_mean,
                angle_concentration=angle_concentration,
                loc_mean=loc_mean,
                loc_variance=loc_variance,
                obs_variance=obs_variance,
                n_factors=n_factors,
                obs_weight_variance=obs_weight_variance,
                factor_variance=factor_variance,
            )
        else:
            self.node_hyperparams = self.node_hyperparams_caller()

        self.reset_parameters(**self.node_hyperparams)

        if self.tssb is not None:
            self.reset_variational_parameters()
            self.sample_variational_distributions()
            self.reset_sufficient_statistics(self.tssb.ntssb.num_batches)

    def combine_params(self):
        return self.params[0] # get loc

    def get_mean(self):
        return self.combine_params()
    
    def set_mean(self, node_mean=None):
        if node_mean is not None:   
            self.node_mean = node_mean
        else:
            self.node_mean = self.get_mean()

    def get_observed_parameters(self):
        return self.observed_parameters[0] # get root loc
    
    def get_params(self):
        return self.get_mean()
    
    def get_param(self, param='mean'):
        if param == 'observed':
            return self.get_observed_parameters()
        elif param == 'mean':
            return self.get_mean()
        else:
            raise ValueError(f"No param available for `{param}`")
        
    def remove_noise(self, data):
        """
        Noise is additive in this model
        """
        return data - self.noise_factors_caller()

    # ========= Functions to initialize node. =========
    def set_node_hyperparams(self, **kwargs):
        self.node_hyperparams.update(**kwargs)

    def reset_parameters(
        self,
        root_loc_mean=2.,
        loc_mean=.5,
        angle_concentration=10.,
        loc_variance=.1,
        obs_variance=.1,
        n_factors=2,
        obs_weight_variance=1.,
        factor_variance=1.,
    ):
        self.node_hyperparams = dict(
            root_loc_mean=root_loc_mean,
            angle_concentration=angle_concentration,
            loc_mean=loc_mean,
            loc_variance=loc_variance,
            obs_variance=obs_variance,
            n_factors=n_factors,
            obs_weight_variance=obs_weight_variance,
            factor_variance=factor_variance,
        )

        parent = self.parent()
        
        if parent is None:
            self.depth = 0.0
            self.params = self.observed_parameters # loc and angle

            n_factors = self.node_hyperparams['n_factors']
            factor_variance = self.node_hyperparams['factor_variance']
            rng = np.random.default_rng(seed=self.seed)
            self.factor_weights = rng.normal(0., np.sqrt(factor_variance), size=(n_factors, 2)) * 1./3.
            
            if n_factors > 0:
                n_genes_per_factor = int(2/n_factors)
                offset = 6.
                perm = np.random.permutation(2)
                for factor in range(n_factors):
                    gene_idx = perm[factor*n_genes_per_factor:(factor+1)*n_genes_per_factor]
                    self.factor_weights[factor,gene_idx] *= offset
            

            # Set data-dependent parameters
            if self.tssb is not None:
                num_data = self.tssb.ntssb.num_data
                if num_data is not None:
                    self.reset_data_parameters()

        elif parent.tssb != self.tssb:
            self.depth = 0.0
            self.params = self.observed_parameters # loc and angle
        else:  # Non-root node: inherits everything from upstream node
            self.depth = parent.depth + 1
            loc_mean = self.node_hyperparams['loc_mean']
            angle_concentration = self.node_hyperparams['angle_concentration'] * self.depth
            rng = np.random.default_rng(seed=self.seed)
            sampled_angle = rng.vonmises(parent.params[1], angle_concentration)
            sampled_loc = rng.normal(
                loc_mean,
                self.node_hyperparams['loc_variance']
            )
            sampled_loc = parent.params[0] + np.array([np.cos(sampled_angle)*np.abs(sampled_loc), np.sin(sampled_angle)*np.abs(sampled_loc)])
            self.params = [sampled_loc, sampled_angle]
        
        self.set_mean()
            
    # Generate structure on the factors
    def reset_data_parameters(self):    
        num_data = self.tssb.ntssb.num_data
        n_factors = self.node_hyperparams['n_factors']
        rng = np.random.default_rng(seed=self.seed)
        self.obs_weights = rng.normal(0., 1., size=(num_data, n_factors)) * 1./3.
        if n_factors > 0:
            n_obs_per_factor = int(num_data/n_factors)
            offset = 6.
            perm = np.random.permutation(num_data)
            for factor in range(n_factors):
                obs_idx = perm[factor*n_obs_per_factor:(factor+1)*n_obs_per_factor]
                self.obs_weights[obs_idx,factor] *= offset

        self.noise_factors = self.obs_weights.dot(self.factor_weights)

    def reset_variational_parameters(self):    
        # Assignments
        num_data = self.tssb.ntssb.num_data
        if num_data is not None:
            self.variational_parameters['q_z'] = jnp.ones(num_data,)

        self.variational_parameters['sum_E_log_1_nu'] = 0.
        self.variational_parameters['E_log_phi'] = 0.

        # Sticks
        self.variational_parameters["delta_1"] = 1.
        self.variational_parameters["delta_2"] = (self.tssb.alpha_decay**self.depth) * self.tssb.dp_alpha 
        self.variational_parameters["sigma_1"] = 1.
        self.variational_parameters["sigma_2"] = self.tssb.dp_gamma

        # Pivots
        self.variational_parameters["q_rho"] = np.ones(len(self.tssb.children_root_nodes),)

        parent = self.parent()
        if parent is None and self.tssb.parent() is None:
            rng = np.random.default_rng(self.seed)
            # root stores global parameters
            n_factors = self.node_hyperparams['n_factors']
            self.variational_parameters["global"] = {
                'factor_weights': {'mean': jnp.array(self.node_hyperparams['factor_variance']/10.*rng.normal(size=(n_factors, 2))),
                                   'log_std': -2. + jnp.zeros((n_factors, 2))}
            }
            if num_data is not None:
                rng = np.random.default_rng(self.seed+1)
                self.variational_parameters["local"] = {
                    'obs_weights': {'mean': jnp.array(self.node_hyperparams['obs_weight_variance']/10.*rng.normal(size=(num_data, n_factors))),
                                    'log_std': -2. + jnp.zeros((num_data, n_factors))}
                }
                self.obs_weights = self.variational_parameters["local"]["obs_weights"]["mean"]
                self.factor_weights = self.variational_parameters["global"]["factor_weights"]["mean"]
                self.noise_factors = self.obs_weights.dot(self.factor_weights)
        elif parent is None:
            return # no variational parameters for root nodes of TSSBs in this model
        else: # only the non-root nodes have variational parameters
            # Kernel
            radius = self.node_hyperparams['loc_mean']

            if "direction" not in parent.variational_parameters["kernel"]:
                mean_angle = jnp.array([parent.observed_parameters[1]])
                parent_loc = jnp.array(parent.observed_parameters[0])
            else:
                mean_angle = parent.variational_parameters["kernel"]["direction"]["mean"]
                parent_loc = parent.variational_parameters["kernel"]["state"]["mean"]

            rng = np.random.default_rng(self.seed+2)
            mean_angle = rng.vonmises(mean_angle, self.node_hyperparams['angle_concentration'] * self.depth)
            mean_loc = parent_loc + jnp.array([np.cos(mean_angle[0])*radius, jnp.sin(mean_angle[0])*radius])
            rng = np.random.default_rng(self.seed+3)
            mean_loc = rng.normal(mean_loc, self.node_hyperparams['loc_variance'])
            self.variational_parameters["kernel"] = {
                'direction': {'mean': jnp.array(mean_angle), 'log_kappa': jnp.array([-1.])},
                'state': {'mean': jnp.array(mean_loc), 'log_std': jnp.array([-1., -1.])}
            }
            self.params = [self.variational_parameters["kernel"]["state"]["mean"], 
                        self.variational_parameters["kernel"]["direction"]["mean"]]

    def set_learned_parameters(self):
        if self.parent() is None and self.tssb.parent() is None:
            self.obs_weights = self.variational_parameters["local"]["obs_weights"]["mean"]
            self.factor_weights = self.variational_parameters["global"]["factor_weights"]["mean"]
            self.noise_factors = self.obs_weights.dot(self.factor_weights)
        elif self.parent() is None:
            self.params = self.observed_parameters
        else:
            self.params = [self.variational_parameters["kernel"]["state"]["mean"], 
                        self.variational_parameters["kernel"]["direction"]["mean"]]

    def reset_sufficient_statistics(self, num_batches=1):
        self.suff_stats = {
            'ent': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(c_n = this tree) q(z_n = this node) * log q(z_n = this node)
            'mass': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node)
            'A': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node) * \sum_g x_ng ** 2
            'B_g': {'total': 0, 'batch': np.zeros((num_batches,2))}, # \sum_n q(z_n = this node) * x_ng
            'C': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node) * \sum_g x_ng * E[s_nW_g]
            'D_g': {'total': 0, 'batch': np.zeros((num_batches,2))}, # \sum_n q(z_n = this node) * E[s_nW_g]
            'E': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node) * \sum_g E[(s_nW_g)**2]
        }
        if self.parent() is None and self.tssb.parent() is None:
            self.local_suff_stats = {
                'locals_kl': {'total': 0., 'batch': np.zeros((num_batches,))},
            }

    def merge_suff_stats(self, suff_stats):
        for stat in self.suff_stats:
            self.suff_stats[stat]['total'] += suff_stats[stat]['total']
            self.suff_stats[stat]['batch'] += suff_stats[stat]['batch']

    def update_sufficient_statistics(self, batch_idx=None):
        if batch_idx is not None:
            idx = self.tssb.ntssb.batch_indices[batch_idx]
        else:
            idx = jnp.arange(self.tssb.ntssb.num_data)
        
        if self.parent() is None and self.tssb.parent() is None:
            locals_kl = self.compute_local_priors(idx) + self.compute_local_entropies(idx)
            if batch_idx is not None:
                self.local_suff_stats['locals_kl']['total'] -= self.local_suff_stats['locals_kl']['batch'][batch_idx]
                self.local_suff_stats['locals_kl']['batch'][batch_idx] = locals_kl
                self.local_suff_stats['locals_kl']['total'] += self.local_suff_stats['locals_kl']['batch'][batch_idx]
            else:
                self.local_suff_stats['locals_kl']['total'] = locals_kl

        ent = assignment_entropies(self.variational_parameters['q_z'][idx]) 
        ent *= self.tssb.variational_parameters['q_c'][idx]
        E_ass = self.variational_parameters['q_z'][idx] * self.tssb.variational_parameters['q_c'][idx]
        E_sw = jnp.mean(self.get_noise_sample(idx),axis=0)
        E_sw2 = jnp.mean(self.get_noise_sample(idx)**2,axis=0)
        x = self.tssb.ntssb.data[idx]
        
        new_ent = jnp.sum(ent)
        new_mass = jnp.sum(E_ass)
        new_A = jnp.sum(E_ass * jnp.sum(x**2, axis=1))
        new_B = jnp.sum(E_ass[:,None] * x, axis=0)
        new_C = jnp.sum(E_ass * jnp.sum(x * E_sw, axis=1))
        new_D = jnp.sum(E_ass[:,None] * E_sw, axis=0)
        new_E = jnp.sum(E_ass * jnp.sum(E_sw2, axis=1))

        if batch_idx is not None:
            self.suff_stats['ent']['total'] -= self.suff_stats['ent']['batch'][batch_idx]
            self.suff_stats['ent']['batch'][batch_idx] = new_ent
            self.suff_stats['ent']['total'] += self.suff_stats['ent']['batch'][batch_idx]

            self.suff_stats['mass']['total'] -= self.suff_stats['mass']['batch'][batch_idx]
            self.suff_stats['mass']['batch'][batch_idx] = new_mass
            self.suff_stats['mass']['total'] += self.suff_stats['mass']['batch'][batch_idx]

            self.suff_stats['A']['total'] -= self.suff_stats['A']['batch'][batch_idx]
            self.suff_stats['A']['batch'][batch_idx] = new_A
            self.suff_stats['A']['total'] += self.suff_stats['A']['batch'][batch_idx]

            self.suff_stats['B_g']['total'] -= self.suff_stats['B_g']['batch'][batch_idx]
            self.suff_stats['B_g']['batch'][batch_idx] = new_B
            self.suff_stats['B_g']['total'] += self.suff_stats['B_g']['batch'][batch_idx]

            self.suff_stats['C']['total'] -= self.suff_stats['C']['batch'][batch_idx]
            self.suff_stats['C']['batch'][batch_idx] = new_C
            self.suff_stats['C']['total'] += self.suff_stats['C']['batch'][batch_idx]

            self.suff_stats['D_g']['total'] -= self.suff_stats['D_g']['batch'][batch_idx]
            self.suff_stats['D_g']['batch'][batch_idx] = new_D
            self.suff_stats['D_g']['total'] += self.suff_stats['D_g']['batch'][batch_idx]

            self.suff_stats['E']['total'] -= self.suff_stats['E']['batch'][batch_idx]
            self.suff_stats['E']['batch'][batch_idx] = new_E
            self.suff_stats['E']['total'] += self.suff_stats['E']['batch'][batch_idx]
        else:
            self.suff_stats['ent']['total'] = new_ent
            self.suff_stats['mass']['total'] = new_mass
            self.suff_stats['A']['total'] = new_A
            self.suff_stats['B_g']['total'] = new_B
            self.suff_stats['C']['total'] = new_C
            self.suff_stats['D_g']['total'] = new_D
            self.suff_stats['E']['total'] = new_E

    # ========= Functions to take samples from node. =========
    def sample_observation(self, n):
        node_mean = self.get_mean()
        noise_factors = self.noise_factors_caller()[n]
        rng = np.random.default_rng(seed=self.seed+n)
        s = rng.normal(node_mean + noise_factors, self.node_hyperparams['obs_variance'])
        return s

    def sample_observations(self):
        n_obs = len(self.data)
        node_mean = self.get_mean()
        noise_factors = self.noise_factors_caller()[np.array(list(self.data))]
        rng = np.random.default_rng(seed=self.seed)
        s = rng.normal(node_mean + noise_factors, self.node_hyperparams['obs_variance'], size=[n_obs, self.n_genes])
        return s

    # ========= Functions to access root's parameters. =========
    def node_hyperparams_caller(self):
        if self.parent() is None:
            return self.node_hyperparams
        else:
            return self.parent().node_hyperparams_caller()

    def noise_factors_caller(self):
        return self.tssb.ntssb.root['node'].root['node'].noise_factors

    def get_obs_weights_sample(self):
        return self.tssb.ntssb.root['node'].root['node'].obs_weights_sample

    def set_local_sample(self, sample, idx=None):
        if idx is None:
            idx = jnp.arange(self.tssb.ntssb.num_data)
        self.tssb.ntssb.root['node'].root['node'].obs_weights_sample = self.tssb.ntssb.root['node'].root['node'].obs_weights_sample.at[:,idx].set(sample)

    def get_factor_weights_sample(self):
        return self.tssb.ntssb.root['node'].root['node'].factor_weights_sample

    def set_global_sample(self, sample):
        self.tssb.ntssb.root['node'].root['node'].factor_weights_sample = jnp.array(sample)

    def get_noise_sample(self, idx):
        obs_weights = self.get_obs_weights_sample()[:,idx]
        factor_weights = self.get_factor_weights_sample()
        return jax.vmap(sample_prod, in_axes=(0,0))(obs_weights,factor_weights)

    def get_direction_sample(self):
        return self.samples[1]
    
    def get_state_sample(self):
        return self.samples[0]

    def get_prior_angle_concentration(self, depth=None):
        if depth is None:
            depth = self.depth
        return self.node_hyperparams['angle_concentration'] * jnp.maximum(depth, 1) # Prior hyperparameter
    
    # ======== Functions using the variational parameters. =========
    def compute_loglikelihood(self, idx):
        # Use stored samples for loc
        node_mean_samples = self.samples[0]
        obs_weights_samples = self.get_obs_weights_sample()[:,idx]
        factor_weights_samples = self.get_factor_weights_sample()
        std = jnp.sqrt(self.node_hyperparams['obs_variance'])
        # Average over samples for each observation
        ll = jnp.mean(jax.vmap(_mc_obs_ll, in_axes=[None,0,0,0,None])(self.tssb.ntssb.data[idx], 
                                                                      node_mean_samples, 
                                                                      obs_weights_samples, 
                                                                      factor_weights_samples, 
                                                                      std), axis=0) # mean over MC samples 
        return ll
    
    def compute_loglikelihood_suff(self):
        node_mean_samples = self.samples[0]
        std = jnp.sqrt(self.node_hyperparams['obs_variance'])
        ll = jnp.mean(jax.vmap(ll_suffstats, in_axes=[0, None, None, None, None, None, None, None])
                      (node_mean_samples,self.suff_stats['mass']['total'],self.suff_stats['A']['total'],
                       self.suff_stats['B_g']['total'], self.suff_stats['C']['total'],
                       self.suff_stats['D_g']['total'],self.suff_stats['E']['total'],std)
                      )
        return ll

    def sample_variational_distributions(self, n_samples=10):
        if self.parent() is not None:
            if self.parent().samples is not None:
                n_samples = self.parent().samples[0].shape[0]
        if self.parent() is None and self.tssb.parent() is None:
            self.sample_locals(n_samples=n_samples, store=True)
            self.sample_globals(n_samples=n_samples, store=True)
        self.sample_kernel(n_samples=n_samples, store=True)

    def sample_locals(self, n_samples, store=True):
        key = jax.random.PRNGKey(self.seed)
        key, sample_grad = self.local_sample_and_grad(jnp.arange(self.tssb.ntssb.num_data), key, n_samples=n_samples)
        sampled_obs_weights, _ = sample_grad
        if store:
            self.obs_weights_sample = sampled_obs_weights
        else:
            return sampled_obs_weights
        
    def sample_globals(self, n_samples, store=True):
        key = jax.random.PRNGKey(self.seed)
        key, sample_grad = self.global_sample_and_grad(key, n_samples=n_samples)
        sampled_factor_weights, _ = sample_grad
        if store:
            self.factor_weights_sample = sampled_factor_weights
        else:
            return sampled_factor_weights

    def sample_kernel(self, n_samples=10, store=True):
        parent = self.parent()
        if parent is None:
            return self._sample_root_kernel(n_samples=n_samples, store=store)
        
        key = jax.random.PRNGKey(self.seed)
        key, sample_grad = self.direction_sample_and_grad(key, n_samples=n_samples)
        sampled_angle, _ = sample_grad

        key, sample_grad = self.state_sample_and_grad(key, n_samples=n_samples)
        sampled_loc, _ = sample_grad
        
        samples = [sampled_loc, sampled_angle]
        if store:
            self.samples = samples
        else:
            return samples

    def _sample_root_kernel(self, n_samples=10, store=True):
        # In this model the root is just the known parameters, so just store n_samples copies of them to mimic a sample
        observed_angle = jnp.array([self.observed_parameters[1]]) # Observed angle
        observed_loc = jnp.array([self.observed_parameters[0]]) # Observed location
        sampled_angle = jnp.vstack(jnp.repeat(observed_angle, n_samples, axis=0))
        sampled_loc = jnp.vstack(jnp.repeat(observed_loc, n_samples, axis=0))
        samples = [sampled_loc, sampled_angle]
        if store:
            self.samples = samples
        else:
            return samples

    def compute_global_priors(self):
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['factor_variance']))
        return jnp.mean(mc_factor_weights_logp_val_and_grad(self.factor_weights_sample, 0., log_std)[0])
    
    def compute_local_priors(self, batch_indices):
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_weight_variance']))
        return jnp.mean(mc_obs_weights_logp_val_and_grad(self.obs_weights_sample[:,batch_indices], 0., log_std)[0])

    def compute_global_entropies(self):
        mean = self.variational_parameters['global']['factor_weights']['mean']
        log_std = self.variational_parameters['global']['factor_weights']['log_std']        
        return jnp.sum(factor_weights_logq_val_and_grad(mean, log_std)[0])

    def compute_local_entropies(self, batch_indices):
        mean = self.variational_parameters['local']['obs_weights']['mean'][batch_indices]
        log_std = self.variational_parameters['local']['obs_weights']['log_std'][batch_indices]
        return jnp.sum(obs_weights_logq_val_and_grad(mean, log_std)[0])

    def compute_kernel_prior(self):
        parent = self.parent()
        if parent is None:
            return self.compute_root_prior()
        
        prior_mean_angle = self.parent().get_direction_sample()
        prior_angle_concentration = self.get_prior_angle_concentration()
        angle_samples = self.get_direction_sample()
        angle_logpdf = mc_angle_logp_val_and_grad(angle_samples, prior_mean_angle, prior_angle_concentration)[0]

        radius = self.node_hyperparams['loc_mean']
        parent_loc = self.parent().get_state_sample() 
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['loc_variance']))
        loc_samples = self.get_state_sample()
        loc_logpdf = mc_loc_logp_val_and_grad(loc_samples, parent_loc, angle_samples, log_std, radius)[0]

        return jnp.mean(angle_logpdf + loc_logpdf)
    
    def compute_root_direction_prior(self, parent_alpha):
        concentration = self.get_prior_angle_concentration()
        alpha = self.get_direction_sample()
        return jnp.mean(mc_angle_logp_val_and_grad(alpha, parent_alpha, concentration)[0])
    
    def compute_root_state_prior(self, parent_psi):
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['loc_variance']))
        radius = self.node_hyperparams['root_loc_mean']
        psi = self.get_state_sample()
        alpha = self.get_direction_sample()
        return jnp.mean(mc_loc_logp_val_and_grad(psi, parent_psi, alpha, log_std, radius)[0])

    def compute_root_kernel_prior(self, samples):
        parent_alpha = samples[0]
        logp = self.compute_root_direction_prior(parent_alpha)
        parent_psi = samples[1]
        logp += self.compute_root_state_prior(parent_psi)
        return logp

    def compute_root_prior(self):
        return 0.

    def compute_kernel_entropy(self):
        parent = self.parent()
        if parent is None:
            return self.compute_root_entropy()
        
        # Angle
        angle_logpdf = tfd.VonMises(np.exp(self.variational_parameters['kernel']['direction']['mean']),
                                    jnp.exp(self.variational_parameters['kernel']['direction']['log_kappa'])
                                    ).entropy()        
        angle_logpdf = jnp.sum(angle_logpdf) 

        # Location
        loc_logpdf = tfd.Normal(self.variational_parameters['kernel']['state']['mean'], 
                                jnp.exp(self.variational_parameters['kernel']['state']['log_std'])
                                ).entropy()
        loc_logpdf = jnp.sum(loc_logpdf) # Sum across features

        return angle_logpdf + loc_logpdf
    
    def compute_root_entropy(self):
        # In this model the root nodes have no unknown parameters
        return 0.
    
    # ======== Functions for updating the variational parameters. =========
    def local_sample_and_grad(self, idx, key, n_samples):
        """Sample and take gradient of local parameters. Must be root"""
        mean = self.variational_parameters['local']['obs_weights']['mean'][idx]
        log_std = self.variational_parameters['local']['obs_weights']['log_std'][idx]
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_obs_weights_val_and_grad(jnp.array(sub_keys), mean, log_std)

    def global_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of global parameters. Must be root"""
        mean = self.variational_parameters['global']['factor_weights']['mean']
        log_std = self.variational_parameters['global']['factor_weights']['log_std']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_factor_weights_val_and_grad(jnp.array(sub_keys), mean, log_std)

    def compute_locals_prior_grad(self, sample):
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_weight_variance']))
        return mc_obs_weights_logp_val_and_grad(sample, 0., log_std)[1]

    def compute_globals_prior_grad(self, sample):
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['factor_variance']))
        return mc_factor_weights_logp_val_and_grad(sample, 0., log_std)[1]

    def compute_locals_entropy_grad(self, idx):
        mean = self.variational_parameters['local']['obs_weights']['mean'][idx]
        log_std = self.variational_parameters['local']['obs_weights']['log_std'][idx]
        return obs_weights_logq_val_and_grad(mean, log_std)[1]
    
    def compute_globals_entropy_grad(self):
        mean = self.variational_parameters['global']['factor_weights']['mean']
        log_std = self.variational_parameters['global']['factor_weights']['log_std']        
        return factor_weights_logq_val_and_grad(mean, log_std)[1]
    
    def state_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of state"""
        mu = self.variational_parameters['kernel']['state']['mean']
        log_std = self.variational_parameters['kernel']['state']['log_std']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_loc_val_and_grad(jnp.array(sub_keys), mu, log_std)

    def direction_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of direction"""
        mu = self.variational_parameters['kernel']['direction']['mean']
        log_kappa = self.variational_parameters['kernel']['direction']['log_kappa']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_angle_val_and_grad(jnp.array(sub_keys), mu, log_kappa)

    def state_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of state"""
        mu = self.variational_parameters['kernel']['state']['mean']
        log_std = self.variational_parameters['kernel']['state']['log_std']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_loc_val_and_grad(jnp.array(sub_keys), mu, log_std)

    def compute_direction_prior_grad(self, alpha, parent_alpha, parent_loc):
        """Gradient of logp(alpha|parent_alpha,parent_loc)"""
        return self.compute_direction_prior_grad_wrt_direction(alpha, parent_alpha, parent_loc)

    def compute_direction_prior_grad_wrt_direction(self, alpha, parent_alpha, parent_loc):
        """Gradient of logp(alpha|parent_alpha) wrt this alpha"""
        concentration = self.get_prior_angle_concentration()
        return mc_angle_logp_val_and_grad(alpha, parent_alpha, concentration)[1]

    def compute_direction_prior_grad_wrt_state(self, alpha, parent_alpha, parent_loc):
        """Gradient of logp(alpha|parent_alpha) wrt this alpha"""
        return 0.

    def compute_direction_prior_child_grad_wrt_state(self, child_direction, direction, state):
        """Gradient of logp(child_alpha|alpha) wrt this direction"""
        return 0.

    def compute_direction_prior_child_grad_wrt_direction(self, child_direction, direction, state):
        """Gradient of logp(child_alpha|alpha) wrt this direction"""
        return self.compute_direction_prior_child_grad(child_direction, direction)

    def compute_direction_prior_child_grad(self, child_alpha, alpha):
        """Gradient of logp(child_alpha|alpha) wrt this alpha"""
        concentration = self.get_prior_angle_concentration(depth=self.depth+1)
        return mc_angle_logp_val_and_grad_wrt_parent(child_alpha, alpha, concentration)[1]

    def compute_state_prior_grad(self, psi, parent_psi, alpha):
        """Gradient of logp(psi|parent_psi,new_alpha) wrt this psi"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['loc_variance']))
        radius = self.node_hyperparams['loc_mean']
        return mc_loc_logp_val_and_grad(psi, parent_psi, alpha, log_std, radius)[1]

    def compute_state_prior_child_grad(self, child_psi, psi, child_alpha):
        """Gradient of logp(child_psi|psi,child_alpha) wrt this psi"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['loc_variance']))
        radius = self.node_hyperparams['loc_mean']
        return mc_loc_logp_val_and_grad_wrt_parent(child_psi, psi, child_alpha, log_std, radius)[1]

    def compute_root_state_prior_child_grad(self, child_psi, psi, child_alpha):
        """Gradient of logp(child_psi|psi,child_alpha) wrt this psi"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['loc_variance']))
        radius = self.node_hyperparams['root_loc_mean']
        return mc_loc_logp_val_and_grad_wrt_parent(child_psi, psi, child_alpha, log_std, radius)[1]

    def compute_state_prior_grad_wrt_direction(self, psi, parent_psi, alpha):
        """Gradient of logp(psi|parent_psi,alpha) wrt this alpha"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['loc_variance']))
        radius = self.node_hyperparams['loc_mean']
        return mc_loc_logp_val_and_grad_wrt_angle(psi, parent_psi, alpha, log_std, radius)[1]

    def compute_direction_entropy_grad(self):
        """Gradient of logq(alpha) wrt this alpha"""
        mu = self.variational_parameters['kernel']['direction']['mean']
        log_kappa = self.variational_parameters['kernel']['direction']['log_kappa']        
        return angle_logq_val_and_grad(mu, log_kappa)[1]

    def compute_state_entropy_grad(self):
        """Gradient of logq(psi) wrt this psi"""
        mu = self.variational_parameters['kernel']['state']['mean']
        log_std = self.variational_parameters['kernel']['state']['log_std']        
        return loc_logq_val_and_grad(mu, log_std)[1]

    def compute_ll_state_grad(self, x, weights, psi):
        """Gradient of logp(x|psi,noise) wrt this psi"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_variance']))
        locals = self.get_obs_weights_sample()
        globals = self.get_factor_weights_sample()
        return mc_ll_val_and_grad_psi(x, weights, psi, locals, globals, log_std)[1]

    def compute_ll_state_grad_suff(self, psi):
        """Gradient of logp(x|psi,noise) wrt this psi using suff stats"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_variance']))
        return mc_ll_node_mean_suff_val_and_grad(psi, self.suff_stats['mass']['total'], 
                                           self.suff_stats['B_g']['total'], 
                                           self.suff_stats['D_g']['total'], log_std)[1]

    def compute_ll_locals_grad(self, x, idx, weights):
        """Gradient of logp(x|psi,locals,globals) wrt locals"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_variance']))
        psi = self.get_state_sample()
        locals = self.get_obs_weights_sample()[:,idx]
        globals = self.get_factor_weights_sample()
        return mc_ll_val_and_grad_obs_weights(x, weights, psi, locals, globals, log_std)[1]

    def compute_ll_globals_grad(self, x, idx, weights):
        """Gradient of logp(x|psi,locals,globals) wrt globals"""
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_variance']))
        psi = self.get_state_sample()
        locals = self.get_obs_weights_sample()[:,idx]
        globals = self.get_factor_weights_sample()
        return mc_ll_val_and_grad_factor_weights(x, weights, psi, locals, globals, log_std)[1]
    
    def update_direction_params(self, direction_params_grad, direction_sample_grad, direction_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(direction_params_grad[0] * direction_sample_grad, axis=0)
        angle_mean_grad = mc_grad + direction_params_entropy_grad[0]
        self.variational_parameters['kernel']['direction']['mean'] += angle_mean_grad * step_size

        mc_grad = jnp.mean(direction_params_grad[1] * direction_sample_grad, axis=0)
        angle_log_kappa_grad = mc_grad + direction_params_entropy_grad[1]
        self.variational_parameters['kernel']['direction']['log_kappa'] += angle_log_kappa_grad * step_size

    def update_state_params(self, state_params_grad, state_sample_grad, state_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(state_params_grad[0] * state_sample_grad, axis=0)
        loc_mean_grad = mc_grad + state_params_entropy_grad[0]
        self.variational_parameters['kernel']['state']['mean'] += loc_mean_grad * step_size

        mc_grad = jnp.mean(state_params_grad[1] * state_sample_grad, axis=0)
        loc_log_std_grad = mc_grad + state_params_entropy_grad[1]
        self.variational_parameters['kernel']['state']['log_std'] += loc_log_std_grad * step_size

    def update_local_params(self, idx, local_params_grad, local_sample_grad, local_params_entropy_grad, ent_anneal=1., step_size=0.001):
        mc_grad = jnp.mean(local_params_grad[0] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[0]
        new_param = self.variational_parameters['local']['obs_weights']['mean'][idx] + param_grad * step_size
        self.variational_parameters['local']['obs_weights']['mean'] = self.variational_parameters['local']['obs_weights']['mean'].at[idx].set(new_param)

        mc_grad = jnp.mean(local_params_grad[1] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[1]
        new_param = self.variational_parameters['local']['obs_weights']['log_std'][idx] + param_grad * step_size
        self.variational_parameters['local']['obs_weights']['log_std'] = self.variational_parameters['local']['obs_weights']['log_std'].at[idx].set(new_param)

    def update_global_params(self, global_params_grad, global_sample_grad, global_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(global_params_grad[0] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[0]
        self.variational_parameters['global']['factor_weights']['mean'] += param_grad * step_size

        mc_grad = jnp.mean(global_params_grad[1] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[1]
        self.variational_parameters['global']['factor_weights']['log_std'] += param_grad * step_size

    def initialize_global_opt_states(self):
        n_factors = self.node_hyperparams['n_factors']
        m = jnp.zeros((n_factors,self.n_genes))
        v = jnp.zeros((n_factors,self.n_genes))
        state1 = (m,v)
        m = jnp.zeros((n_factors,self.n_genes))
        v = jnp.zeros((n_factors,self.n_genes))
        state2 = (m,v)
        states = (state1, state2)
        return states

    def update_global_params_adaptive(self, global_params_grad, global_sample_grad, global_params_entropy_grad, i, states, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001):
        mc_grad = jnp.mean(global_params_grad[0] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[0]
        
        m, v = states[0]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state1 = (m, v)
        self.variational_parameters['global']['factor_weights']['mean'] += step_size * mhat / (jnp.sqrt(vhat) + eps)


        mc_grad = jnp.mean(global_params_grad[1] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[1]
        
        m, v = states[1]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        self.variational_parameters['global']['factor_weights']['log_std'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        states = (state1, state2)
        return states
    
    def initialize_state_states(self):
        m = jnp.zeros((self.n_genes,))
        v = jnp.zeros((self.n_genes,))
        state1 = (m,v)
        m = jnp.zeros((self.n_genes,))
        v = jnp.zeros((self.n_genes,))
        state2 = (m,v)
        states = (state1, state2)
        return states    

    def update_state_adaptive(self, state_params_grad, state_sample_grad, state_params_entropy_grad, i, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001):
        states = self.state_states

        mc_grad = jnp.mean(state_params_grad[0] * state_sample_grad, axis=0)
        param_grad = mc_grad + state_params_entropy_grad[0]
        
        m, v = states[0]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state1 = (m, v)
        self.variational_parameters['kernel']['state']['mean'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        mc_grad = jnp.mean(state_params_grad[1] * state_sample_grad, axis=0)
        param_grad = mc_grad + state_params_entropy_grad[1]
        
        m, v = states[1]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        self.variational_parameters['kernel']['state']['log_std'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        states = (state1, state2)
        self.state_states = states
    
    def initialize_direction_states(self):
        m = jnp.zeros((1,))
        v = jnp.zeros((1,))
        state1 = (m,v)
        m = jnp.zeros((1,))
        v = jnp.zeros((1,))
        state2 = (m,v)
        states = (state1, state2)
        return states   

    def update_direction_adaptive(self, direction_params_grad, direction_sample_grad, direction_params_entropy_grad, i, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001):
        states = self.direction_states
        mc_grad = jnp.mean(direction_params_grad[0] * direction_sample_grad, axis=0)
        param_grad = mc_grad + direction_params_entropy_grad[0]
        
        m, v = states[0]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state1 = (m, v)
        self.variational_parameters['kernel']['direction']['mean'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        mc_grad = jnp.mean(direction_params_grad[1] * direction_sample_grad, axis=0)
        param_grad = mc_grad + direction_params_entropy_grad[1]
        
        m, v = states[1]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        self.variational_parameters['kernel']['direction']['log_kappa'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        states = (state1, state2)
        self.direction_states = states    