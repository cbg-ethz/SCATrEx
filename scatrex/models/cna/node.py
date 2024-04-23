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

MIN_ALPHA = jnp.log(1e-2)
MAX_BETA = jnp.log(1e6)

def update_params(params, params_gradient, step_size):
    new_params = []
    for i, param in enumerate(params):
        new_params.append(param + step_size * params_gradient[i])
    return new_params

class CNANode(AbstractNode):
    def __init__(
        self,
        observed_parameters, # copy number state
        cell_scale_mean=1e2,
        cell_scale_shape=1.,
        gene_scale_mean=1e2,
        gene_scale_shape=10.,
        direction_shape=.1,
        inheritance_strength=1.,
        n_factors=2,
        obs_weight_variance=1.,
        factor_precision_shape=2.,
        min_cnv = 1e-6,
        max_cnv=6.,
        **kwargs,
    ):
        """
        This model generates nodes in gene expression space combined with observed copy number states
        """
        super(CNANode, self).__init__(observed_parameters, **kwargs)

        # The observed parameters are the CNVs of all genes
        self.cnvs = np.array(self.observed_parameters)
        self.cnvs[np.where(self.cnvs == 0)[0]] = min_cnv # To avoid zero in Poisson likelihood
        self.cnvs[np.where(self.cnvs > max_cnv)[0]] = max_cnv # Dosage
        self.observed_parameters = np.array(self.cnvs)
        self.cnvs = jnp.array(self.cnvs)

        self.n_genes = self.cnvs.size

        # Node hyperparameters
        if self.parent() is None:
            self.node_hyperparams = dict(
                cell_scale_mean=cell_scale_mean,
                cell_scale_shape=cell_scale_shape,
                gene_scale_mean=gene_scale_mean,
                gene_scale_shape=gene_scale_shape,
                direction_shape=direction_shape,
                inheritance_strength=inheritance_strength,
                n_factors=n_factors,
                obs_weight_variance=obs_weight_variance,
                factor_precision_shape=factor_precision_shape,
            )
        else:
            self.node_hyperparams = self.node_hyperparams_caller()

        self.reset_parameters(**self.node_hyperparams)

        if self.tssb is not None:
            self.reset_variational_parameters()
            self.sample_variational_distributions()
            self.reset_sufficient_statistics(self.tssb.ntssb.num_batches)
            # For adaptive optimization
            self.reset_opt()

    def reset_opt(self):
        # For adaptive optimization
        self.direction_states = self.initialize_direction_states()
        self.state_states = self.initialize_state_states()

    def apply_clip(self, param, minval=-jnp.inf, maxval=jnp.inf):
        param = jnp.maximum(param, minval)
        param = jnp.minimum(param, maxval)
        return param

    def combine_params(self):
        return np.exp(self.params[0]) * 0.5 * self.cnvs # params is a list of two: 0 is \qsi and 1 is \chi

    def get_mean(self):
        return self.combine_params()
    
    def set_mean(self, node_mean=None):
        if node_mean is not None:   
            self.node_mean = node_mean
        else:
            self.node_mean = self.get_mean()

    def get_observed_parameters(self):
        return self.cnvs
    
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
        Noise is multiplicative in this model
        """
        return data * 1./self.cell_scales_caller() * 1./self.gene_scales_caller() * 1./jnp.exp(self.noise_factors_caller())

    # ========= Functions to initialize node. =========
    def set_node_hyperparams(self, **kwargs):
        self.node_hyperparams.update(**kwargs)

    def reset_parameters(
        self,
        cell_scale_mean=1e2,
        cell_scale_shape=1.,
        gene_scale_mean=1e2,
        gene_scale_shape=10.,
        direction_shape=.1,
        inheritance_strength=1.,
        n_factors=2,
        obs_weight_variance=1.,
        factor_precision_shape=2.,
    ):
        self.node_hyperparams = dict(
            cell_scale_mean=cell_scale_mean,
            cell_scale_shape=cell_scale_shape,
            gene_scale_mean=gene_scale_mean,
            gene_scale_shape=gene_scale_shape,
            direction_shape=direction_shape,
            inheritance_strength=inheritance_strength,
            n_factors=n_factors,
            obs_weight_variance=obs_weight_variance,
            factor_precision_shape=factor_precision_shape,
        )

        parent = self.parent()
        
        if parent is None:
            self.depth = 0.0
            self.params = [np.zeros((self.n_genes,)), np.ones((self.n_genes,))] # state and direction

            # Gene scales
            rng = np.random.default_rng(seed=self.seed)
            self.gene_scales = rng.gamma(self.node_hyperparams['gene_scale_shape'], self.node_hyperparams['gene_scale_mean']/self.node_hyperparams['gene_scale_shape'], size=(1, self.n_genes))

            # Structured noise
            n_factors = self.node_hyperparams['n_factors']
            factor_precision_shape = self.node_hyperparams['factor_precision_shape']
            
            self.factor_precisions = rng.gamma(factor_precision_shape, 1., size=(n_factors,1))
            factor_scales = np.sqrt(1./self.factor_precisions)
            self.factor_weights = rng.normal(0., factor_scales, size=(n_factors, self.n_genes)) * 1./np.sqrt(factor_precision_shape)
            
            if n_factors > 0:
                n_genes_per_factor = int(2/n_factors)
                offset = np.sqrt(factor_precision_shape)
                perm = np.random.permutation(self.n_genes)
                for factor in range(n_factors):
                    gene_idx = perm[factor*n_genes_per_factor:(factor+1)*n_genes_per_factor]
                    self.factor_weights[factor,gene_idx] *= offset

            # Set data-dependent parameters
            if self.tssb is not None:
                num_data = self.tssb.ntssb.num_data
                if num_data is not None:
                    self.reset_data_parameters()
        else:  # Non-root node: inherits everything from upstream node
            self.depth = parent.depth + 1
            rng = np.random.default_rng(seed=self.seed)
            sampled_direction = rng.gamma(self.node_hyperparams['direction_shape'], 
                1./self.node_hyperparams['inheritance_strength'] * 10.**(-jnp.abs(parent.params[0])))
            sampled_state = jnp.maximum(rng.normal(parent.params[0], sampled_direction), -2)
            sampled_state = jnp.minimum(sampled_state, 2)
            self.params = [sampled_state, sampled_direction]
        
        self.set_mean()
            
    # Generate cell sizes and structure on the factors
    def reset_data_parameters(self):    
        num_data = self.tssb.ntssb.num_data
        rng = np.random.default_rng(seed=self.seed)
        self.cell_scales = rng.gamma(self.node_hyperparams['cell_scale_shape'], self.node_hyperparams['cell_scale_mean']/self.node_hyperparams['cell_scale_shape'], size=(num_data, 1))
        
        n_factors = self.node_hyperparams['n_factors']
        self.obs_weights = rng.normal(0., 1., size=(num_data, n_factors)) * 1./3.
        if n_factors > 0:
            n_obs_per_factor = int(num_data/n_factors)
            offset = 6.
            perm = np.random.permutation(num_data)
            for factor in range(n_factors):
                obs_idx = perm[factor*n_obs_per_factor:(factor+1)*n_obs_per_factor]
                self.obs_weights[obs_idx,factor] *= offset

        self.noise_factors = self.obs_weights.dot(self.factor_weights)

    def reset_data_variational_parameters(self):
        if self.parent() is None and self.tssb.parent() is None:
            num_data = self.tssb.ntssb.num_data

            # Set priors
            lib_sizes = np.sum(self.tssb.ntssb.data, axis=1)
            self.lib_ratio = np.ones((self.tssb.ntssb.data.shape[0], 1))
            self.lib_ratio *= np.mean(lib_sizes) / np.var(lib_sizes)
            gene_sizes = np.sum(self.tssb.ntssb.data, axis=0)
            self.gene_means = np.mean(self.tssb.ntssb.data, axis=0)
            self.gene_ratio = np.mean(gene_sizes) / np.var(gene_sizes)

            cell_scales_alpha_init = self.node_hyperparams['cell_scale_shape'] * jnp.ones((num_data,1))
            cell_scales_beta_init = self.node_hyperparams['cell_scale_shape'] * jnp.ones((num_data,1))
            gene_scales_alpha_init = self.node_hyperparams['gene_scale_shape'] * jnp.ones((self.n_genes,))
            gene_scales_beta_init = self.node_hyperparams['gene_scale_shape'] * jnp.ones((self.n_genes,)) * 1./self.gene_means

            rng = np.random.default_rng(self.seed)
            # root stores global parameters
            n_factors = self.node_hyperparams['n_factors']
            factor_precision_shape = self.node_hyperparams['factor_precision_shape']
            self.variational_parameters["global"] = {
                'gene_scales': {'log_alpha': jnp.log(gene_scales_alpha_init),
                                'log_beta': jnp.log(gene_scales_beta_init)},
                 'factor_precisions': {'log_alpha': jnp.log(10. * jnp.ones((n_factors,1))), 
                                         'log_beta' : jnp.log(10./factor_precision_shape * jnp.ones((n_factors,1)))},                                
                'factor_weights': {'mean': jnp.array(0.01*rng.normal(size=(n_factors, self.n_genes))),
                                   'log_std': -2. + jnp.zeros((n_factors, self.n_genes))}
            }
            rng = np.random.default_rng(self.seed+1)
            self.variational_parameters["local"] = {
                'cell_scales': {'log_alpha': jnp.log(cell_scales_alpha_init),
                            'log_beta': jnp.log(cell_scales_beta_init)},
                'obs_weights': {'mean': jnp.array(self.node_hyperparams['obs_weight_variance']/10.*rng.normal(size=(num_data, n_factors))),
                                'log_std': -2. + jnp.zeros((num_data, n_factors))}
            }
            self.cell_scales = jnp.exp(self.variational_parameters["local"]["cell_scales"]["log_alpha"]-self.variational_parameters["local"]["cell_scales"]["log_beta"])
            self.gene_scales = jnp.exp(self.variational_parameters["global"]["gene_scales"]["log_alpha"]-self.variational_parameters["global"]["gene_scales"]["log_beta"])
            self.obs_weights = self.variational_parameters["local"]["obs_weights"]["mean"]
            self.factor_precisions = self.variational_parameters["global"]["factor_weights"]["mean"]
            self.factor_weights = self.variational_parameters["global"]["factor_weights"]["mean"]
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
        self.variational_parameters["delta_2"] = 1.
        self.variational_parameters["sigma_1"] = 1.
        self.variational_parameters["sigma_2"] = 1.

        # Pivots
        self.variational_parameters["q_rho"] = np.ones(len(self.tssb.children_root_nodes),)

        parent = self.parent()
        if parent is None and self.tssb.parent() is None:
            self.params = [jnp.zeros((self.n_genes,)), jnp.ones((self.n_genes,))]
            
            if num_data is not None:
                self.reset_data_variational_parameters()
            else:
                rng = np.random.default_rng(self.seed)
                # root stores global parameters
                n_factors = self.node_hyperparams['n_factors']
                factor_precision_shape = self.node_hyperparams['factor_precision_shape']
                self.variational_parameters["global"] = {
                    'gene_scales': {'log_alpha': jnp.ones((self.n_genes,)),
                                    'log_beta': jnp.ones((self.n_genes,))},
                    'factor_precisions': {'log_alpha': jnp.log(10. * jnp.ones((n_factors,1))), 
                                         'log_beta' : jnp.log(10./factor_precision_shape * jnp.ones((n_factors,1)))},
                    'factor_weights': {'mean': jnp.array(0.01*rng.normal(size=(n_factors, self.n_genes))),
                                    'log_std': -2. + jnp.zeros((n_factors, self.n_genes))}
                }
        else: # only the non-root nodes have kernel variational parameters
            # Kernel
            parent_param = jnp.zeros((self.n_genes,))
            if parent is not None:
                parent_param = parent.params[0]
                
            rng = np.random.default_rng(self.seed+2)
            sampled_direction = rng.gamma(self.node_hyperparams['direction_shape'],
                                          jnp.exp(-self.node_hyperparams['inheritance_strength'] * jnp.abs(parent_param)))
            rng = np.random.default_rng(self.seed+3)
            if np.all(parent_param == 0):
                sampled_state = rng.normal(parent_param*0.1, 0.01) # is root node, so avoid messing with main node attachments
            else:
                sampled_state = jnp.clip(rng.normal(parent_param*0.1, sampled_direction), a_min=-1, a_max=1) # to explore (without numerical explosions)
            
            init_concentration = 10.
            self.variational_parameters["kernel"] = {
                'direction': {'log_alpha': jnp.log(init_concentration*jnp.ones((self.n_genes,))), 'log_beta': jnp.log(init_concentration/self.node_hyperparams['direction_shape'] * jnp.ones((self.n_genes,)))},
                'state': {'mean': jnp.array(sampled_state), 'log_std': jnp.array(rng.normal(-2., 0.01, size=self.n_genes))}
            }
            self.params = [self.variational_parameters["kernel"]["state"]["mean"], 
                        jnp.exp(self.variational_parameters["kernel"]["direction"]["log_alpha"]-self.variational_parameters["kernel"]["direction"]["log_beta"])]

    def reset_variational_kernel(self, log_std=-4, init_concentration=10):
        parent = self.parent()
        if parent is None and self.tssb.parent() is None:
            pass
        else:
            parent_param = jnp.zeros((self.n_genes,))
            if parent is not None:
                parent_param = parent.params[0]
                
            rng = np.random.default_rng(self.seed+2)
            sampled_direction = rng.gamma(self.node_hyperparams['direction_shape'],
                                            jnp.exp(-self.node_hyperparams['inheritance_strength'] * jnp.abs(parent_param)))
            rng = np.random.default_rng(self.seed+3)
            if np.all(parent_param == 0):
                sampled_state = rng.normal(parent_param*0.1, np.exp(log_std)) # is root node, so avoid messing with main node attachments
            else:
                sampled_state = jnp.clip(rng.normal(parent_param*0.1, sampled_direction), a_min=-1, a_max=1) # to explore (without numerical explosions)
            
            self.variational_parameters["kernel"] = {
                'direction': {'log_alpha': jnp.log(init_concentration*jnp.ones((self.n_genes,))), 'log_beta': jnp.log(init_concentration/self.node_hyperparams['direction_shape'] * jnp.ones((self.n_genes,)))},
                'state': {'mean': jnp.array(sampled_state), 'log_std': jnp.array(rng.normal(log_std, 0.01, size=self.n_genes))}
            }
            self.params = [self.variational_parameters["kernel"]["state"]["mean"], 
                        jnp.exp(self.variational_parameters["kernel"]["direction"]["log_alpha"]-self.variational_parameters["kernel"]["direction"]["log_beta"])]


    def reset_variational_noise_factors(self):    
        rng = np.random.default_rng(self.seed)
        n_factors = self.node_hyperparams['n_factors']
        factor_precision_shape = self.node_hyperparams['factor_precision_shape']
        self.variational_parameters["global"]["factor_precisions"] = {
                'log_alpha': jnp.log(10. * jnp.ones((n_factors,1))), 
                'log_beta' : jnp.log(10./factor_precision_shape * jnp.ones((n_factors,1)))
        }
        self.variational_parameters["global"]["factor_weights"] = {
                'mean': jnp.array(0.01*rng.normal(size=(n_factors, self.n_genes))),
                'log_std': -2. + jnp.zeros((n_factors, self.n_genes))
        }
        num_data = self.tssb.ntssb.num_data
        self.variational_parameters["local"]["obs_weights"] = {
                'mean': jnp.array(self.node_hyperparams['obs_weight_variance']/10.*rng.normal(size=(num_data, n_factors))),
                'log_std': -2. + jnp.zeros((num_data, n_factors))
        }

    def set_learned_parameters(self):
        if self.parent() is None and self.tssb.parent() is None:
            self.obs_weights = self.variational_parameters["local"]["obs_weights"]["mean"]
            self.factor_precisions = jnp.exp(self.variational_parameters["global"]["factor_precisions"]["log_alpha"]
                            -self.variational_parameters["global"]["factor_precisions"]["log_beta"])            
            self.factor_weights = self.variational_parameters["global"]["factor_weights"]["mean"]
            self.noise_factors = self.obs_weights.dot(self.factor_weights)
            self.cell_scales = jnp.exp(self.variational_parameters["local"]["cell_scales"]["log_alpha"]
                                       -self.variational_parameters["local"]["cell_scales"]["log_beta"])
            self.gene_scales = jnp.exp(self.variational_parameters["global"]["gene_scales"]["log_alpha"]
                            -self.variational_parameters["global"]["gene_scales"]["log_beta"])
        else:
            self.params = [self.variational_parameters["kernel"]["state"]["mean"], 
                        jnp.exp(self.variational_parameters["kernel"]["direction"]["log_alpha"]-self.variational_parameters["kernel"]["direction"]["log_beta"])]

    def reset_sufficient_statistics(self, num_batches=1):
        self.suff_stats = {
            'ent': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(c_n = this tree) q(z_n = this node) * log q(z_n = this node)
            'mass': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node)
            'A': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node) * \sum_g x_ng * E[\gamma_n]
            'B_g': {'total': 0, 'batch': np.zeros((num_batches,self.n_genes))}, # \sum_n q(z_n = this node) * x_ng
            'C': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node) * \sum_g x_ng * E[s_nW_g]
            'D_g': {'total': 0, 'batch': np.zeros((num_batches,self.n_genes))}, # \sum_n q(z_n = this node) * E[\gamma_n] * E[s_nW_g]
            'E': {'total': 0, 'batch': np.zeros((num_batches,))}, # \sum_n q(z_n = this node) * lgamma(x_ng+1)
        }
        if self.parent() is None and self.tssb.parent() is None:
            self.local_suff_stats = {
                'locals_kl': {'total': 0., 'batch': np.zeros((num_batches,))},
            }

    def init_new_node_kernel(self, **kwargs):
        # Get data for which prob of assigning to parent is > 1/n_total_nodes
        weights = self.parent().variational_parameters['q_z'] * self.tssb.variational_parameters['q_c']
        idx = np.where(weights > 1./np.sqrt(self.tssb.ntssb.n_total_nodes))[0]
        # Initialize prioritizing cells with lowest ll in parent
        if len(idx) > 0 and 'll' in self.parent().variational_parameters: # only proceed in this manner if parent has already been evaluated
            thres = np.quantile(self.parent().variational_parameters['ll'][idx], q=.1)
            idx = idx[np.where(self.parent().variational_parameters['ll'][idx] < thres)[0]]
            self.reset_variational_state(idx=idx, **kwargs)
            # Sample
            if self.parent().samples is not None:
                n_samples = self.parent().samples[0].shape[0]
                self.sample_kernel(n_samples=n_samples)

    def init_kernel(self, **kwargs):
        # Get data for which prob of assigning here is > 1/n_total_nodes
        weights = self.variational_parameters['q_z'] * self.tssb.variational_parameters['q_c']
        idx = np.where(weights > 1./np.sqrt(self.tssb.ntssb.n_total_nodes))[0]
        # Initialize prioritizing cells with highest ll here
        thres = np.quantile(self.variational_parameters['ll'][idx], q=.9)
        idx = idx[np.where(self.variational_parameters['ll'][idx] >= thres)[0]]
        self.reset_variational_state(idx=idx, **kwargs)

    def reset_variational_state(self, log_std=0., idx=None, weights=None):
        if self.parent() is None and self.tssb.parent() is None:
            return
        else:
            if idx is None:
                idx = jnp.arange(self.tssb.ntssb.num_data)
            if weights is None:
                weights = self.variational_parameters['q_z'][idx] * self.tssb.variational_parameters['q_c'][idx]                
            cell_scales_mean = jnp.mean(self.get_cell_scales_sample()[:,idx],axis=0)
            gene_scales_mean = jnp.mean(self.get_gene_scales_sample(),axis=0)
            noise_factors = jnp.mean(self.get_noise_sample(idx),axis=0)
            cnvs_contrib = self.cnvs/2
            init_state = jnp.log(1+jnp.sum(self.tssb.ntssb.data[idx]/(cell_scales_mean*gene_scales_mean*cnvs_contrib*jnp.exp(noise_factors)) * weights[:,None],axis=0)/jnp.sum(weights[:,None]))
            self.variational_parameters['kernel']['state']['mean'] = jnp.clip(init_state, -1, 1) # avoid explosion
            self.variational_parameters['kernel']['state']['log_std'] = log_std * jnp.ones((self.n_genes,))
        
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
        E_loggamma = jnp.mean(jnp.log(self.get_cell_scales_sample()[:,idx]),axis=0)
        E_gamma = jnp.mean(self.get_cell_scales_sample()[:,idx],axis=0)
        E_sw = jnp.mean(self.get_noise_sample(idx),axis=0)
        E_expsw = jnp.mean(jnp.exp(self.get_noise_sample(idx)),axis=0)
        x = self.tssb.ntssb.data[idx]
        
        new_ent = jnp.sum(ent)
        new_mass = jnp.sum(E_ass)
        new_A = jnp.sum(E_ass * E_loggamma.ravel() * jnp.sum(x, axis=1))
        new_B = jnp.sum(E_ass[:,None] * x, axis=0)
        new_C = jnp.sum(E_ass * jnp.sum(x * E_sw, axis=1))
        new_D = jnp.sum(E_ass[:,None] * E_gamma * E_expsw, axis=0)
        new_E = jnp.sum(E_ass * jnp.sum(gammaln(x+1), axis=1))

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
        state = self.params[0]
        cnvs = self.cnvs
        noise_factors = self.noise_factors_caller()[n]
        cell_scales = self.cell_scales_caller()[n]
        gene_scales = self.gene_scales_caller()
        rng = np.random.default_rng(seed=self.seed+n)
        s = rng.poisson(cell_scales*gene_scales*cnvs/2*2**(state)*jnp.exp(noise_factors))
        return s

    def sample_observations(self):
        n_obs = len(self.data)
        state = self.params[0]
        cnvs = self.cnvs
        noise_factors = self.noise_factors_caller()[np.array(list(self.data))]
        cell_scales = self.cell_scales_caller()[np.array(list(self.data))]
        gene_scales = self.gene_scales_caller()
        rng = np.random.default_rng(seed=self.seed)
        s = rng.poisson(cell_scales*gene_scales*cnvs/2*2**(state)*jnp.exp(noise_factors), size=[n_obs, self.n_genes])
        return s

    # ========= Functions to access root's parameters. =========
    def node_hyperparams_caller(self):
        if self.parent() is None:
            return self.node_hyperparams
        else:
            return self.parent().node_hyperparams_caller()

    def noise_factors_caller(self):
        return self.tssb.ntssb.root['node'].root['node'].noise_factors

    def cell_scales_caller(self):
        return self.tssb.ntssb.root['node'].root['node'].cell_scales

    def get_cell_scales_sample(self):
        return self.tssb.ntssb.root['node'].root['node'].cell_scales_sample

    def gene_scales_caller(self):
        return self.tssb.ntssb.root['node'].root['node'].gene_scales

    def get_gene_scales_sample(self):
        return self.tssb.ntssb.root['node'].root['node'].gene_scales_sample

    def get_obs_weights_sample(self):
        return self.tssb.ntssb.root['node'].root['node'].obs_weights_sample

    def set_local_sample(self, sample, idx=None):
        """
        obs_weights, cell_scales
        """
        if idx is None:
            idx = jnp.arange(self.tssb.ntssb.num_data)
        self.tssb.ntssb.root['node'].root['node'].obs_weights_sample = self.tssb.ntssb.root['node'].root['node'].obs_weights_sample.at[:,idx].set(sample[0])
        self.tssb.ntssb.root['node'].root['node'].cell_scales_sample = self.tssb.ntssb.root['node'].root['node'].cell_scales_sample.at[:,idx].set(sample[1])        

    def get_factor_weights_sample(self):
        return self.tssb.ntssb.root['node'].root['node'].factor_weights_sample

    def set_global_sample(self, sample):
        """
        factor_weights, gene_scales
        """
        self.tssb.ntssb.root['node'].root['node'].factor_weights_sample = jnp.array(sample[0])
        self.tssb.ntssb.root['node'].root['node'].gene_scales_sample = jnp.array(sample[1])

    def get_noise_sample(self, idx):
        obs_weights = self.get_obs_weights_sample()[:,idx]
        factor_weights = self.get_factor_weights_sample()
        return jax.vmap(sample_prod, in_axes=(0,0))(obs_weights,factor_weights)

    def get_direction_sample(self):
        return self.samples[1]
    
    def get_state_sample(self):
        return self.samples[0]
    
    # ======== Functions using the variational parameters. =========
    def compute_loglikelihood(self, idx):
        # Use stored samples for loc
        node_mean_samples = self.samples[0]
        cnvs = self.cnvs
        obs_weights_samples = self.get_obs_weights_sample()[:,idx]
        factor_weights_samples = self.get_factor_weights_sample()
        cell_scales_samples = self.get_cell_scales_sample()[:,idx]
        gene_scales_samples = self.get_gene_scales_sample()
        # Average over samples for each observation
        ll = jnp.mean(jax.vmap(_mc_obs_ll, in_axes=[None,0,None,0,0,0,0])(self.tssb.ntssb.data[idx], 
                                                                      node_mean_samples, 
                                                                      cnvs,
                                                                      obs_weights_samples, 
                                                                      factor_weights_samples,
                                                                      cell_scales_samples,
                                                                      gene_scales_samples,                                                                      
                                                                      ), axis=0) # mean over MC samples 
        return ll

    def compute_loglikelihood_suff(self):
        state_samples = self.samples[0]
        gene_scales_samples = self.get_gene_scales_sample()
        cnv = self.cnvs
        ll = jnp.mean(jax.vmap(ll_suffstats, in_axes=[0, None, 0, None, None, None, None, None])
                      (state_samples, cnv, gene_scales_samples, self.suff_stats['A']['total'], self.suff_stats['B_g']['total'],
                       self.suff_stats['C']['total'], self.suff_stats['D_g']['total'], self.suff_stats['E']['total'])
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
        sample, _ = sample_grad
        obs_weights_sample, cell_scales_sample = sample
        if store:
            self.obs_weights_sample = obs_weights_sample
            self.cell_scales_sample = cell_scales_sample
        else:
            return obs_weights_sample, cell_scales_sample
        
    def sample_globals(self, n_samples, store=True):
        key = jax.random.PRNGKey(self.seed)
        key, sample_grad = self.global_sample_and_grad(key, n_samples=n_samples)
        sample, _ = sample_grad
        factor_weights_sample, gene_scales_sample, factor_precisions_sample = sample
        if store:
            self.factor_weights_sample = factor_weights_sample
            self.gene_scales_sample = gene_scales_sample
            self.factor_precisions_sample = factor_precisions_sample
        else:
            return factor_weights_sample, gene_scales_sample, factor_precisions_sample

    def sample_kernel(self, n_samples=10, store=True):
        if self.parent() is None and self.tssb.parent() is None:
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
        # In this model the complete tree's root parameters are fixed and not learned, so just store n_samples copies of them to mimic a sample
        sampled_direction = jnp.vstack(jnp.repeat(jnp.array([self.params[1]]), n_samples, axis=0))
        sampled_state = jnp.vstack(jnp.repeat(jnp.array([self.params[0]]), n_samples, axis=0))
        samples = [sampled_state, sampled_direction]
        if store:
            self.samples = samples
        else:
            return samples

    def compute_global_priors(self):
        factor_weights_contrib = jnp.sum(jnp.mean(mc_factor_weights_logp_val_and_grad(self.factor_weights_sample, 0., self.factor_precisions_sample)[0], axis=0))
        log_alpha = jnp.log(self.node_hyperparams['gene_scale_shape'])
        log_beta = jnp.log(self.node_hyperparams['gene_scale_shape'] * 1./self.gene_means)
        gene_scales_contrib = jnp.sum(jnp.mean(mc_gene_scales_logp_val_and_grad(self.gene_scales_sample, log_alpha, log_beta)[0], axis=0))
        log_alpha = jnp.log(self.node_hyperparams['factor_precision_shape'])
        log_beta = jnp.log(1.)
        factor_precisions_contrib = jnp.sum(jnp.mean(mc_factor_precisions_logp_val_and_grad(self.factor_precisions_sample, log_alpha, log_beta)[0], axis=0))
        return factor_weights_contrib + gene_scales_contrib + factor_precisions_contrib
    
    def compute_local_priors(self, batch_indices):
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_weight_variance']))
        obs_weights_contrib = jnp.sum(jnp.mean(mc_obs_weights_logp_val_and_grad(self.obs_weights_sample[:,batch_indices], 0., log_std)[0], axis=0))
        log_alpha = jnp.log(self.node_hyperparams['cell_scale_shape'])
        log_beta = jnp.log(self.node_hyperparams['cell_scale_shape'] * self.lib_ratio)
        cell_scales_contrib = jnp.sum(jnp.mean(mc_cell_scales_logp_val_and_grad(self.cell_scales_sample[batch_indices], log_alpha, log_beta)[0], axis=0))
        return obs_weights_contrib + cell_scales_contrib

    def compute_global_entropies(self):
        mean = self.variational_parameters['global']['factor_weights']['mean']
        log_std = self.variational_parameters['global']['factor_weights']['log_std']        
        factor_weights_contrib = jnp.sum(factor_weights_logq_val_and_grad(mean, log_std)[0])
        
        log_alpha = self.variational_parameters['global']['gene_scales']['log_alpha']
        log_beta = self.variational_parameters['global']['gene_scales']['log_beta']                
        gene_scales_contrib = jnp.sum(gene_scales_logq_val_and_grad(log_alpha, log_beta)[0])

        log_alpha = self.variational_parameters['global']['factor_precisions']['log_alpha']
        log_beta = self.variational_parameters['global']['factor_precisions']['log_beta']                
        factor_precisions_contrib = jnp.sum(factor_precisions_logq_val_and_grad(log_alpha, log_beta)[0])
        
        return factor_weights_contrib + gene_scales_contrib + factor_precisions_contrib

    def compute_local_entropies(self, batch_indices):
        mean = self.variational_parameters['local']['obs_weights']['mean'][batch_indices]
        log_std = self.variational_parameters['local']['obs_weights']['log_std'][batch_indices]
        obs_weights_contrib = jnp.sum(obs_weights_logq_val_and_grad(mean, log_std)[0])

        log_alpha = self.variational_parameters['local']['cell_scales']['log_alpha'][batch_indices]
        log_beta = self.variational_parameters['local']['cell_scales']['log_beta'][batch_indices]
        cell_scales_contrib = jnp.sum(cell_scales_logq_val_and_grad(log_alpha, log_beta)[0])
        return obs_weights_contrib + cell_scales_contrib

    def compute_kernel_prior(self):
        parent = self.parent()
        if parent is None:
            return self.compute_root_prior()
        
        parent_state = self.parent().get_state_sample()
        direction_samples = self.get_direction_sample()

        direction_shape = self.node_hyperparams['direction_shape']
        inheritance_strength = self.node_hyperparams['inheritance_strength']

        direction_logpdf = mc_direction_logp_val_and_grad(direction_samples, parent_state, direction_shape, inheritance_strength)[0]
        direction_logpdf = jnp.sum(direction_logpdf,axis=1)

        state_samples = self.get_state_sample()
        state_logpdf = mc_state_logp_val_and_grad(state_samples, parent_state, direction_samples)[0]
        state_logpdf = jnp.sum(state_logpdf, axis=1)

        return jnp.mean(direction_logpdf + state_logpdf)
    
    def compute_root_direction_prior(self, parent_state):
        direction_samples = self.get_direction_sample()
        direction_shape = self.node_hyperparams['direction_shape']
        inheritance_strength = self.node_hyperparams['inheritance_strength']        
        return jnp.mean(jnp.sum(mc_direction_logp_val_and_grad(direction_samples, parent_state, direction_shape, inheritance_strength)[0], axis=1))
    
    def compute_root_state_prior(self, parent_state):
        direction_samples = jnp.sqrt(self.get_direction_sample())
        state_samples = self.get_state_sample()
        return jnp.mean(jnp.sum(mc_state_logp_val_and_grad(state_samples, parent_state, direction_samples)[0], axis=1))

    def compute_root_kernel_prior(self, samples):
        parent_state = samples[0]        
        logp = self.compute_root_direction_prior(parent_state)
        logp += self.compute_root_state_prior(parent_state)
        return logp

    def compute_root_prior(self):
        return 0.

    def compute_kernel_entropy(self):
        parent = self.parent()
        if parent is None:
            return self.compute_root_entropy()
        
        # Direction
        direction_logpdf = tfd.Gamma(np.exp(self.variational_parameters['kernel']['direction']['log_alpha']),
                                    jnp.exp(self.variational_parameters['kernel']['direction']['log_beta'])
                                    ).entropy()        
        direction_logpdf = jnp.sum(direction_logpdf) 

        # State
        state_logpdf = tfd.Normal(self.variational_parameters['kernel']['state']['mean'], 
                                jnp.exp(self.variational_parameters['kernel']['state']['log_std'])
                                ).entropy()
        state_logpdf = jnp.sum(state_logpdf) # Sum across features

        return direction_logpdf + state_logpdf
    
    def compute_root_entropy(self):
        # In this model the root nodes have no unknown parameters
        return 0.
    
    # ======== Functions for updating the variational parameters. =========
    def local_sample_and_grad(self, idx, key, n_samples):
        """Sample and take gradient of local parameters. Must be root"""
        mean = self.variational_parameters['local']['obs_weights']['mean'][idx]
        log_std = self.variational_parameters['local']['obs_weights']['log_std'][idx]
        key, *sub_keys = jax.random.split(key, n_samples+1)
        obs_weights_sample_grad = mc_sample_obs_weights_val_and_grad(jnp.array(sub_keys), mean, log_std)
        obs_weights_sample = obs_weights_sample_grad[0]
        # obs_weights_sample = obs_weights_sample.at[:,0,:].set(0.)

        log_alpha = self.variational_parameters['local']['cell_scales']['log_alpha'][idx]
        log_beta = self.variational_parameters['local']['cell_scales']['log_beta'][idx]
        key, *sub_keys = jax.random.split(key, n_samples+1)
        cell_scales_sample_grad = mc_sample_cell_scales_val_and_grad(jnp.array(sub_keys), log_alpha, log_beta)

        sample = [obs_weights_sample, cell_scales_sample_grad[0]]
        grad = [obs_weights_sample_grad[1], cell_scales_sample_grad[1]]

        return key, (sample, grad)

    def global_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of global parameters. Must be root"""
        mean = self.variational_parameters['global']['factor_weights']['mean']
        log_std = self.variational_parameters['global']['factor_weights']['log_std']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        factor_weights_sample_grad = mc_sample_factor_weights_val_and_grad(jnp.array(sub_keys), mean, log_std)
        factor_weights_sample = factor_weights_sample_grad[0]
        # factor_weights_sample = factor_weights_sample.at[:,:,0].set(0.)

        log_alpha = self.variational_parameters['global']['gene_scales']['log_alpha']
        log_beta = self.variational_parameters['global']['gene_scales']['log_beta']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        gene_scales_sample_grad = mc_sample_gene_scales_val_and_grad(jnp.array(sub_keys), log_alpha, log_beta)
        # gene_scales_sample = gene_scales_sample_grad[0]
        # gene_scales_sample = gene_scales_sample.at[jnp.arange(n_samples),0].set(1.)

        log_alpha = self.variational_parameters['global']['factor_precisions']['log_alpha']
        log_beta = self.variational_parameters['global']['factor_precisions']['log_beta']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        factor_precisions_sample_grad = mc_sample_factor_precisions_val_and_grad(jnp.array(sub_keys), log_alpha, log_beta)        

        sample = [factor_weights_sample, gene_scales_sample_grad[0], factor_precisions_sample_grad[0]]
        grad = [factor_weights_sample_grad[1], gene_scales_sample_grad[1], factor_precisions_sample_grad[1]]

        return key, (sample, grad)

    def compute_locals_prior_grad(self, sample):
        log_std = jnp.log(jnp.sqrt(self.node_hyperparams['obs_weight_variance']))
        obs_weights_grad = mc_obs_weights_logp_val_and_grad(sample[0], 0., log_std)[1]
        log_alpha = jnp.log(self.node_hyperparams['cell_scale_shape'])
        log_beta = jnp.log(self.node_hyperparams['cell_scale_shape'] * self.lib_ratio)
        cell_scales_grad = mc_cell_scales_logp_val_and_grad(sample[1], log_alpha, log_beta)[1]
        return obs_weights_grad, cell_scales_grad

    def compute_globals_prior_grad(self, sample):
        factor_weights_grad = mc_factor_weights_logp_val_and_grad(sample[0], 0., sample[2])[1]
        log_alpha = jnp.log(self.node_hyperparams['gene_scale_shape'])
        log_beta = jnp.log(self.node_hyperparams['gene_scale_shape'] * 1./self.gene_means)
        gene_scales_grad = mc_gene_scales_logp_val_and_grad(sample[1], log_alpha, log_beta)[1]
        log_alpha = jnp.log(self.node_hyperparams['factor_precision_shape'])
        log_beta = jnp.log(1.)
        factor_precisions_grad = mc_factor_precisions_logp_val_and_grad(sample[2], log_alpha, log_beta)[1]        
        factor_precisions_grad += mc_factor_weights_logp_val_and_grad_wrt_precisions(sample[0], 0., sample[2])[1]
        return factor_weights_grad, gene_scales_grad, factor_precisions_grad

    def compute_locals_entropy_grad(self, idx):
        mean = self.variational_parameters['local']['obs_weights']['mean'][idx]
        log_std = self.variational_parameters['local']['obs_weights']['log_std'][idx]
        obs_weights_grad = obs_weights_logq_val_and_grad(mean, log_std)[1]

        log_alpha = self.variational_parameters['local']['cell_scales']['log_alpha'][idx]
        log_beta = self.variational_parameters['local']['cell_scales']['log_beta'][idx]
        cell_scales_grad = cell_scales_logq_val_and_grad(log_alpha, log_beta)[1]

        return obs_weights_grad, cell_scales_grad    
    
    def compute_globals_entropy_grad(self):
        mean = self.variational_parameters['global']['factor_weights']['mean']
        log_std = self.variational_parameters['global']['factor_weights']['log_std']        
        factor_weights_grad = factor_weights_logq_val_and_grad(mean, log_std)[1]

        log_alpha = self.variational_parameters['global']['gene_scales']['log_alpha']
        log_beta = self.variational_parameters['global']['gene_scales']['log_beta']        
        gene_scales_grad = gene_scales_logq_val_and_grad(log_alpha, log_beta)[1]

        log_alpha = self.variational_parameters['global']['factor_precisions']['log_alpha']
        log_beta = self.variational_parameters['global']['factor_precisions']['log_beta']        
        factor_precisions_grad = factor_precisions_logq_val_and_grad(log_alpha, log_beta)[1]        

        return factor_weights_grad, gene_scales_grad, factor_precisions_grad
    
    def state_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of state"""
        mu = self.variational_parameters['kernel']['state']['mean']
        log_std = self.variational_parameters['kernel']['state']['log_std']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_state_val_and_grad(jnp.array(sub_keys), mu, log_std)

    def direction_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of direction"""
        log_alpha = self.variational_parameters['kernel']['direction']['log_alpha']
        log_beta = self.variational_parameters['kernel']['direction']['log_beta']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_direction_val_and_grad(jnp.array(sub_keys), log_alpha, log_beta)

    def state_sample_and_grad(self, key, n_samples):
        """Sample and take gradient of state"""
        mu = self.variational_parameters['kernel']['state']['mean']
        log_std = self.variational_parameters['kernel']['state']['log_std']
        key, *sub_keys = jax.random.split(key, n_samples+1)
        return key, mc_sample_state_val_and_grad(jnp.array(sub_keys), mu, log_std)

    def compute_direction_prior_grad(self, direction, parent_direction, parent_state):
        """Gradient of logp(direction|parent_state) wrt this direction"""
        direction_shape = self.node_hyperparams['direction_shape']
        inheritance_strength = self.node_hyperparams['inheritance_strength']
        return mc_direction_logp_val_and_grad(direction, parent_state, direction_shape, inheritance_strength)[1]

    def compute_direction_prior_child_grad_wrt_direction(self, child_direction, direction, state):
        """Gradient of logp(child_alpha|alpha) wrt this direction"""
        return 0. # no influence under this model

    def compute_direction_prior_child_grad_wrt_state(self, child_direction, direction, state):
        """Gradient of logp(child_alpha|alpha) wrt this direction"""
        direction_shape = self.node_hyperparams['direction_shape']
        inheritance_strength = self.node_hyperparams['inheritance_strength']
        return mc_direction_logp_val_and_grad_wrt_parent(child_direction, state, direction_shape, inheritance_strength)[1]

    def compute_state_prior_grad(self, state, parent_state, direction):
        """Gradient of logp(state|parent_state,direction) wrt this psi"""
        return mc_state_logp_val_and_grad(state, parent_state, direction)[1]

    def compute_state_prior_child_grad(self, child_state, state, child_direction):
        """Gradient of logp(child_state|state,child_direction) wrt this state"""
        return mc_state_logp_val_and_grad_wrt_parent(child_state, state, child_direction)[1]

    def compute_root_state_prior_child_grad(self, child_state, state, child_direction):
        """Gradient of logp(child_psi|psi,child_alpha) wrt this psi"""
        return self.compute_state_prior_child_grad(child_state, state, child_direction)

    def compute_state_prior_grad_wrt_direction(self, state, parent_state, direction):
        """Gradient of logp(state|parent_state,direction) wrt this direction"""
        return mc_state_logp_val_and_grad_wrt_direction(state, parent_state, direction)[1]

    def compute_direction_entropy_grad(self):
        """Gradient of logq(alpha) wrt this alpha"""
        log_alpha = self.variational_parameters['kernel']['direction']['log_alpha']
        log_beta = self.variational_parameters['kernel']['direction']['log_beta']        
        return direction_logq_val_and_grad(log_alpha, log_beta)[1]

    def compute_state_entropy_grad(self):
        """Gradient of logq(psi) wrt this psi"""
        mu = self.variational_parameters['kernel']['state']['mean']
        log_std = self.variational_parameters['kernel']['state']['log_std']        
        return state_logq_val_and_grad(mu, log_std)[1]

    def compute_ll_state_grad(self, x, weights, state):
        """Gradient of logp(x|psi,noise) wrt this psi"""
        obs_weights = self.get_obs_weights_sample()
        factor_weights = self.get_factor_weights_sample()
        gene_scales = self.get_gene_scales_sample()
        cell_scales = self.get_cell_scales_sample()
        cnvs = self.cnvs
        return mc_ll_val_and_grad_state(x, weights, state, cnvs, obs_weights, factor_weights, cell_scales, gene_scales)[1]

    def compute_ll_state_grad_suff(self, state):
        """Gradient of logp(x|state,noise) wrt this state using suff stats"""
        cnv = self.cnvs
        gene_scales = self.get_gene_scales_sample()
        return mc_ll_state_suff_val_and_grad(state, cnv, gene_scales, 
                                             self.suff_stats['B_g']['total'], self.suff_stats['D_g']['total'])[1]

    def compute_ll_locals_grad(self, x, idx, weights):
        """Gradient of logp(x|psi,locals,globals) wrt locals"""
        state = self.get_state_sample()
        obs_weights = self.get_obs_weights_sample()[:,idx]
        factor_weights = self.get_factor_weights_sample()
        gene_scales = self.get_gene_scales_sample()
        cell_scales = self.get_cell_scales_sample()[:,idx]
        cnvs = self.cnvs
        obs_weights_grad = mc_ll_val_and_grad_obs_weights(x, weights, state, cnvs, obs_weights, factor_weights, cell_scales, gene_scales)[1]
        cell_scales_grad = mc_ll_val_and_grad_cell_scales(x, weights, state, cnvs, obs_weights, factor_weights, cell_scales, gene_scales)[1]
        return obs_weights_grad, cell_scales_grad

    def compute_ll_globals_grad(self, x, idx, weights):
        """Gradient of logp(x|psi,locals,globals) wrt globals"""
        state = self.get_state_sample()
        obs_weights = self.get_obs_weights_sample()[:,idx]
        factor_weights = self.get_factor_weights_sample()
        gene_scales = self.get_gene_scales_sample()
        cell_scales = self.get_cell_scales_sample()[:,idx]
        cnvs = self.cnvs
        factor_weights_grad = mc_ll_val_and_grad_factor_weights(x, weights, state, cnvs, obs_weights, factor_weights, cell_scales, gene_scales)[1]
        gene_scales_grad = mc_ll_val_and_grad_gene_scales(x, weights, state, cnvs, obs_weights, factor_weights, cell_scales, gene_scales)[1]
        return factor_weights_grad, gene_scales_grad
    
    def update_direction_params(self, direction_params_grad, direction_sample_grad, direction_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(direction_params_grad[0] * direction_sample_grad, axis=0)
        direction_log_alpha_grad = mc_grad + direction_params_entropy_grad[0]
        self.variational_parameters['kernel']['direction']['log_alpha'] += direction_log_alpha_grad * step_size

        mc_grad = jnp.mean(direction_params_grad[1] * direction_sample_grad, axis=0)
        direction_log_beta_grad = mc_grad + direction_params_entropy_grad[1]
        self.variational_parameters['kernel']['direction']['log_beta'] += direction_log_beta_grad * step_size

        self.variational_parameters['kernel']['direction']['log_alpha'] = self.apply_clip(self.variational_parameters['kernel']['direction']['log_alpha'], minval=MIN_ALPHA)
        self.variational_parameters['kernel']['direction']['log_beta'] = self.apply_clip(self.variational_parameters['kernel']['direction']['log_beta'], maxval=MAX_BETA)

    def update_state_params(self, state_params_grad, state_sample_grad, state_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(state_params_grad[0] * state_sample_grad, axis=0)
        loc_mean_grad = mc_grad + state_params_entropy_grad[0]
        self.variational_parameters['kernel']['state']['mean'] += loc_mean_grad * step_size

        mc_grad = jnp.mean(state_params_grad[1] * state_sample_grad, axis=0)
        loc_log_std_grad = mc_grad + state_params_entropy_grad[1]
        self.variational_parameters['kernel']['state']['log_std'] += loc_log_std_grad * step_size

        self.variational_parameters['kernel']['state']['mean']  = self.apply_clip(self.variational_parameters['kernel']['state']['mean'], minval=-5., maxval=5.)
        self.variational_parameters['kernel']['state']['log_std']  = self.apply_clip(self.variational_parameters['kernel']['state']['log_std'], minval=-5., maxval=5.)

    def update_cell_scales_params(self, idx, local_params_grad, local_sample_grad, local_params_entropy_grad, ent_anneal=1., step_size=0.001):
        mc_grad = jnp.mean(local_params_grad[0] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[0]
        new_param = self.variational_parameters['local']['cell_scales']['log_alpha'][idx] + param_grad * step_size
        self.variational_parameters['local']['cell_scales']['log_alpha'] = self.variational_parameters['local']['cell_scales']['log_alpha'].at[idx].set(new_param)

        mc_grad = jnp.mean(local_params_grad[1] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[1]
        new_param = self.variational_parameters['local']['cell_scales']['log_beta'][idx] + param_grad * step_size
        self.variational_parameters['local']['cell_scales']['log_beta'] = self.variational_parameters['local']['cell_scales']['log_beta'].at[idx].set(new_param)

        self.variational_parameters['local']['cell_scales']['log_alpha'] = self.apply_clip(self.variational_parameters['local']['cell_scales']['log_alpha'], minval=MIN_ALPHA)
        self.variational_parameters['local']['cell_scales']['log_beta'] = self.apply_clip(self.variational_parameters['local']['cell_scales']['log_beta'], maxval=MAX_BETA)

    def update_obs_weights_params(self, idx, local_params_grad, local_sample_grad, local_params_entropy_grad, ent_anneal=1., step_size=0.001):
        mc_grad = jnp.mean(local_params_grad[0] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[0]
        new_param = self.variational_parameters['local']['obs_weights']['mean'][idx] + param_grad * step_size
        self.variational_parameters['local']['obs_weights']['mean'] = self.variational_parameters['local']['obs_weights']['mean'].at[idx].set(new_param)

        mc_grad = jnp.mean(local_params_grad[1] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[1]
        new_param = self.variational_parameters['local']['obs_weights']['log_std'][idx] + param_grad * step_size
        self.variational_parameters['local']['obs_weights']['log_std'] = self.variational_parameters['local']['obs_weights']['log_std'].at[idx].set(new_param)

    def update_local_params(self, idx, local_params_grad, local_sample_grad, local_params_entropy_grad, ent_anneal=1., step_size=.001, param_names=["obs_weights", "cell_scales"], **kwargs):
        if param_names is None:
            param_names=["obs_weights", "cell_scales"]
        if "obs_weights" in param_names:
            self.update_obs_weights_params(idx, local_params_grad[0], local_sample_grad[0], local_params_entropy_grad[0], ent_anneal=ent_anneal, step_size=step_size)
        if "cell_scales" in param_names:
            self.update_cell_scales_params(idx, local_params_grad[1], local_sample_grad[1], local_params_entropy_grad[1], ent_anneal=ent_anneal, step_size=step_size)

    def update_gene_scales_params(self, global_params_grad, global_sample_grad, global_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(global_params_grad[0] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[0]
        self.variational_parameters['global']['gene_scales']['log_alpha'] += param_grad * step_size

        mc_grad = jnp.mean(global_params_grad[1] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[1]
        self.variational_parameters['global']['gene_scales']['log_beta'] += param_grad * step_size

        self.variational_parameters['global']['gene_scales']['log_alpha'] = self.apply_clip(self.variational_parameters['global']['gene_scales']['log_alpha'], minval=MIN_ALPHA)
        self.variational_parameters['global']['gene_scales']['log_beta'] = self.apply_clip(self.variational_parameters['global']['gene_scales']['log_beta'], maxval=MAX_BETA)

    def update_factor_weights_params(self, global_params_grad, global_sample_grad, global_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(global_params_grad[0] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[0]
        self.variational_parameters['global']['factor_weights']['mean'] += param_grad * step_size

        mc_grad = jnp.mean(global_params_grad[1] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[1]
        self.variational_parameters['global']['factor_weights']['log_std'] += param_grad * step_size

    def update_factor_precisions_params(self, global_params_grad, global_sample_grad, global_params_entropy_grad, step_size=0.001):
        mc_grad = jnp.mean(global_params_grad[0] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[0]
        self.variational_parameters['global']['factor_precisions']['log_alpha'] += param_grad * step_size

        mc_grad = jnp.mean(global_params_grad[1] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[1]
        self.variational_parameters['global']['factor_precisions']['log_beta'] += param_grad * step_size

        self.variational_parameters['global']['factor_precisions']['log_alpha'] = self.apply_clip(self.variational_parameters['global']['factor_precisions']['log_alpha'], minval=MIN_ALPHA)
        self.variational_parameters['global']['factor_precisions']['log_beta'] = self.apply_clip(self.variational_parameters['global']['factor_precisions']['log_beta'], maxval=MAX_BETA)

    def update_global_params(self, global_params_grad, global_sample_grad, global_params_entropy_grad, step_size=0.001, 
                             param_names=["factor_weights", "gene_scales", "factor_precisions"], **kwargs):
        if param_names is None:
            param_names=["factor_weights", "gene_scales", "factor_precisions"]
        if "factor_weights" in param_names:
            self.update_factor_weights_params(global_params_grad[0], global_sample_grad[0], global_params_entropy_grad[0], step_size=step_size)
        if "gene_scales" in param_names:
            self.update_gene_scales_params(global_params_grad[1], global_sample_grad[1], global_params_entropy_grad[1], step_size=step_size)
        if "factor_precisions" in param_names:           
            self.update_factor_precisions_params(global_params_grad[2], global_sample_grad[2], global_params_entropy_grad[2], step_size=step_size)

    def initialize_global_opt_states(self, param_names=["factor_weights", "gene_scales", "factor_precisions"]):
        states = dict()
        if param_names is None:
            param_names=["factor_weights", "gene_scales", "factor_precisions"]
        if "factor_weights" in param_names:
            factor_weights_states = self.initialize_factor_weights_states()
            states["factor_weights"] = factor_weights_states
        if "gene_scales" in param_names:    
            gene_scales_states = self.initialize_gene_scales_states()
            states["gene_scales"] = gene_scales_states
        if "factor_precisions" in param_names:
            factor_precisions_states = self.initialize_factor_precisions_states()
            states["factor_precisions"] = factor_precisions_states
        return states
    
    def initialize_factor_weights_states(self):
        n_factors = self.node_hyperparams['n_factors']
        m = jnp.zeros((n_factors,self.n_genes))
        v = jnp.zeros((n_factors,self.n_genes))
        state1 = (m,v)
        m = jnp.zeros((n_factors,self.n_genes))
        v = jnp.zeros((n_factors,self.n_genes))
        state2 = (m,v)
        states = (state1, state2)
        return states
    
    def initialize_gene_scales_states(self):
        m = jnp.zeros((self.n_genes,))
        v = jnp.zeros((self.n_genes,))
        state1 = (m,v)
        m = jnp.zeros((self.n_genes,))
        v = jnp.zeros((self.n_genes,))
        state2 = (m,v)
        states = (state1, state2)
        return states    
    
    def initialize_factor_precisions_states(self):
        n_factors = self.node_hyperparams['n_factors']
        m = jnp.zeros((n_factors,1))
        v = jnp.zeros((n_factors,1))
        state1 = (m,v)
        m = jnp.zeros((n_factors,1))
        v = jnp.zeros((n_factors,1))
        state2 = (m,v)
        states = (state1, state2)
        return states    
        
    def update_factor_weights_adaptive(self, global_params_grad, global_sample_grad, global_params_entropy_grad, i, states, b1=0.9,
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

    def update_gene_scales_adaptive(self, global_params_grad, global_sample_grad, global_params_entropy_grad, i, states, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001):
        mc_grad = jnp.mean(global_params_grad[0] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[0]
        
        m, v = states[0]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state1 = (m, v)
        self.variational_parameters['global']['gene_scales']['log_alpha'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        mc_grad = jnp.mean(global_params_grad[1] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[1]
        
        m, v = states[1]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        self.variational_parameters['global']['gene_scales']['log_beta'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        self.variational_parameters['global']['gene_scales']['log_alpha'] = self.apply_clip(self.variational_parameters['global']['gene_scales']['log_alpha'], minval=MIN_ALPHA)
        self.variational_parameters['global']['gene_scales']['log_beta'] = self.apply_clip(self.variational_parameters['global']['gene_scales']['log_beta'], maxval=MAX_BETA)

        states = (state1, state2)
        return states

    def update_factor_precisions_adaptive(self, global_params_grad, global_sample_grad, global_params_entropy_grad, i, states, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001):
        mc_grad = jnp.mean(global_params_grad[0] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[0]
        
        m, v = states[0]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state1 = (m, v)
        self.variational_parameters['global']['factor_precisions']['log_alpha'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        mc_grad = jnp.mean(global_params_grad[1] * global_sample_grad, axis=0)
        param_grad = mc_grad + global_params_entropy_grad[1]
        
        m, v = states[1]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        self.variational_parameters['global']['factor_precisions']['log_beta'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        self.variational_parameters['global']['factor_precisions']['log_alpha'] = self.apply_clip(self.variational_parameters['global']['factor_precisions']['log_alpha'], minval=MIN_ALPHA)
        self.variational_parameters['global']['factor_precisions']['log_beta'] = self.apply_clip(self.variational_parameters['global']['factor_precisions']['log_beta'], maxval=MAX_BETA)

        states = (state1, state2)
        return states

    def update_global_params_adaptive(self, global_params_grad, global_sample_grad, global_params_entropy_grad, i, states, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001, param_names=["factor_weights", "gene_scales", "factor_precisions"], **kwargs):
        if param_names is None:
            param_names=["factor_weights", "gene_scales", "factor_precisions"]        
        if "factor_weights" in param_names:
            factor_weights_states = self.update_factor_weights_adaptive(global_params_grad[0], global_sample_grad[0], global_params_entropy_grad[0], 
                                                    i=i, states=states["factor_weights"], b1=b1, b2=b2, eps=eps, step_size=step_size)
            states["factor_weights"] = factor_weights_states
        if "gene_scales" in param_names:    
            gene_scales_states = self.update_gene_scales_adaptive(global_params_grad[1], global_sample_grad[1], global_params_entropy_grad[1],
                                                    i=i, states=states["gene_scales"], b1=b1, b2=b2, eps=eps, step_size=step_size)
            states["gene_scales"] = gene_scales_states
        if "factor_precisions" in param_names:    
            factor_precisions_states = self.update_factor_precisions_adaptive(global_params_grad[2], global_sample_grad[2], global_params_entropy_grad[2],
                                                    i=i, states=states["factor_precisions"], b1=b1, b2=b2, eps=eps, step_size=step_size)        
            states["factor_precisions"] = factor_precisions_states
        return states
    

    def initialize_local_opt_states(self, param_names=["obs_weights", "cell_scales"]):
        states = dict()
        if param_names is None:
            param_names=["obs_weights", "cell_scales"]             
        if "obs_weights" in param_names:
            obs_weights_states = self.initialize_obs_weights_states()
            states["obs_weights"] = obs_weights_states
        if "cell_scales" in param_names:
            cell_scales_states = self.initialize_cell_scales_states()
            states["cell_scales"] = cell_scales_states
        return states

    def initialize_obs_weights_states(self):
        n_obs = self.tssb.ntssb.num_data
        n_factors = self.node_hyperparams['n_factors']
        m = jnp.zeros((n_obs,n_factors))
        v = jnp.zeros((n_obs,n_factors))
        state1 = (m,v)
        m = jnp.zeros((n_obs,n_factors))
        v = jnp.zeros((n_obs,n_factors))
        state2 = (m,v)
        states = (state1, state2)
        return states
    
    def initialize_cell_scales_states(self):
        n_obs = self.tssb.ntssb.num_data
        m = jnp.zeros((n_obs,1))
        v = jnp.zeros((n_obs,1))
        state1 = (m,v)
        m = jnp.zeros((n_obs,1))
        v = jnp.zeros((n_obs,1))
        state2 = (m,v)
        states = (state1, state2)
        return states    

    def update_obs_weights_adaptive(self, idx, local_params_grad, local_sample_grad, local_params_entropy_grad, i, states, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001, ent_anneal=1.):
        """
        states are not indexed
        """
        mc_grad = jnp.mean(local_params_grad[0] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[0]
        
        m, v = states[0]
        new_m = (1 - b1) * param_grad + b1 * m[idx] # First  moment estimate.
        new_v = (1 - b2) * jnp.square(param_grad) + b2 * v[idx]  # Second moment estimate.
        m = m.at[idx].set(new_m)
        v = v.at[idx].set(new_v)
        mhat = new_m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = new_v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state1 = (m, v)
        new_param = self.variational_parameters['local']['obs_weights']['mean'][idx] + step_size * mhat / (jnp.sqrt(vhat) + eps)
        self.variational_parameters['local']['obs_weights']['mean'] = self.variational_parameters['local']['obs_weights']['mean'].at[idx].set(new_param)

        mc_grad = jnp.mean(local_params_grad[1] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[1]
        
        m, v = states[1]      
        new_m = (1 - b1) * param_grad + b1 * m[idx]  # First  moment estimate.
        new_v = (1 - b2) * jnp.square(param_grad) + b2 * v[idx]  # Second moment estimate.
        m = m.at[idx].set(new_m)
        v = v.at[idx].set(new_v)
        mhat = new_m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = new_v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        new_param = self.variational_parameters['local']['obs_weights']['log_std'][idx] + step_size * mhat / (jnp.sqrt(vhat) + eps)
        self.variational_parameters['local']['obs_weights']['log_std'] = self.variational_parameters['local']['obs_weights']['log_std'].at[idx].set(new_param)

        states = (state1, state2)
        return states

    def update_cell_scales_adaptive(self, idx, local_params_grad, local_sample_grad, local_params_entropy_grad, i, states, b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001, ent_anneal=1.):
        """
        states are already indexed, as are the gradients
        """
        mc_grad = jnp.mean(local_params_grad[0] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[0]
        
        m, v = states[0]
        new_m = (1 - b1) * param_grad + b1 * m[idx] # First  moment estimate.
        new_v = (1 - b2) * jnp.square(param_grad) + b2 * v[idx]  # Second moment estimate.
        m = m.at[idx].set(new_m)
        v = v.at[idx].set(new_v)
        mhat = new_m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = new_v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state1 = (m, v)
        new_param = self.variational_parameters['local']['cell_scales']['log_alpha'][idx] + step_size * mhat / (jnp.sqrt(vhat) + eps)
        self.variational_parameters['local']['cell_scales']['log_alpha'] = self.variational_parameters['local']['cell_scales']['log_alpha'].at[idx].set(new_param)

        mc_grad = jnp.mean(local_params_grad[1] * local_sample_grad, axis=0)
        param_grad = mc_grad + ent_anneal * local_params_entropy_grad[1]
        
        m, v = states[1]
        new_m = (1 - b1) * param_grad + b1 * m[idx] # First  moment estimate.
        new_v = (1 - b2) * jnp.square(param_grad) + b2 * v[idx]  # Second moment estimate.
        m = m.at[idx].set(new_m)
        v = v.at[idx].set(new_v)
        mhat = new_m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = new_v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        new_param = self.variational_parameters['local']['cell_scales']['log_beta'][idx] + step_size * mhat / (jnp.sqrt(vhat) + eps)
        self.variational_parameters['local']['cell_scales']['log_beta'] = self.variational_parameters['local']['cell_scales']['log_beta'].at[idx].set(new_param)

        self.variational_parameters['local']['cell_scales']['log_alpha'] = self.apply_clip(self.variational_parameters['local']['cell_scales']['log_alpha'], minval=MIN_ALPHA)
        self.variational_parameters['local']['cell_scales']['log_beta'] = self.apply_clip(self.variational_parameters['local']['cell_scales']['log_beta'], maxval=MAX_BETA)

        states = (state1, state2)
        return states

    def update_local_params_adaptive(self, idx, local_params_grad, local_sample_grad, local_params_entropy_grad, i, states, ent_anneal=1., b1=0.9,
        b2=0.999, eps=1e-8, step_size=0.001, param_names=["obs_weights", "cell_scales"], **kwargs):
        if param_names is None:
            param_names = ["obs_weights", "cell_scales"]

        if "obs_weights" in param_names:
            obs_weights_states = self.update_obs_weights_adaptive(idx, local_params_grad[0], local_sample_grad[0], local_params_entropy_grad[0], 
                                                   i=i, states=states["obs_weights"], b1=b1, b2=b2, eps=eps, step_size=step_size, ent_anneal=ent_anneal)
            states["obs_weights"] = obs_weights_states
        if "cell_scales" in param_names:
            cell_scales_states = self.update_cell_scales_adaptive(idx, local_params_grad[1], local_sample_grad[1], local_params_entropy_grad[1],
                                                 i=i, states=states["cell_scales"], b1=b1, b2=b2, eps=eps, step_size=step_size, ent_anneal=ent_anneal)
            states["cell_scales"] = cell_scales_states
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
        m = jnp.zeros((self.n_genes,))
        v = jnp.zeros((self.n_genes,))
        state1 = (m,v)
        m = jnp.zeros((self.n_genes,))
        v = jnp.zeros((self.n_genes,))
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
        self.variational_parameters['kernel']['direction']['log_alpha'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        mc_grad = jnp.mean(direction_params_grad[1] * direction_sample_grad, axis=0)
        param_grad = mc_grad + direction_params_entropy_grad[1]
        
        m, v = states[1]
        m = (1 - b1) * param_grad + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(param_grad) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        state2 = (m, v)
        self.variational_parameters['kernel']['direction']['log_beta'] += step_size * mhat / (jnp.sqrt(vhat) + eps)

        states = (state1, state2)
        self.direction_states = states