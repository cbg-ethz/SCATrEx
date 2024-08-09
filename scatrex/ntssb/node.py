import numpy as np
from numpy.random import *
import jax.numpy as jnp

from abc import ABC, abstractmethod
from functools import reduce


class AbstractNode(ABC):
    def __init__(
        self, observed_parameters, parent=None, tssb=None, label="", seed=42,
    ):
        self.data = set([])
        self._children = set([])
        self.tssb = tssb
        self.label = label
        self.event_str = ""
        self.seed = seed

        self.params = dict()
        self.observed_parameters = observed_parameters
        self.variational_parameters = dict(globals=dict(), locals=dict())
        self.variational_parameters = {'delta_1': 1., 'delta_2': 1., # nu stick
                                       'sigma_1': 1., 'sigma_2': 1., # psi stick
                                       'q_z': [], # prob of assigning each cell to this node
                                       'kernel' : dict(), # model-specific
                                       'q_rho': [], # prob of assigning the root node of each child TSSB to this node
                                       'E_log_phi': 0., # auxiliary quantity
                                       'sum_E_log_1_nu': 0., # auxiliary quantity
                                       }
        self.samples = None
        self.pivot_prior_prob = 1. # prior prob of assigning the root node of each child TSSB to this node

        self.data_weights = 0.
        self.weight_until_here = 0.
        self.psi_stick_kl = 0.
        self.ancestors_and_this_E_log_1_nu = 0.
        self.ancestors_and_this_E_log_phi = 0.
        self.psi_not_prev_sum = 0.
        self.ll = 0.

        if parent is not None:
            parent.add_child(self)
            self._parent = parent
            n_siblings = len(list(parent.children()))
            self.label = parent.label + "-" + str(n_siblings - 1)

            # Init with parent
            self.data_weights = np.array(parent.data_weights)
            self.ll = np.array(parent.ll)
        else:
            self._parent = None

    @abstractmethod
    def compute_loglikelihood(self):
        return

    @abstractmethod
    def combine_params(self):
        return

    @abstractmethod
    def sample_kernel(self):
        return
    
    @abstractmethod
    def compute_kernel_prior(self):
        return
    
    @abstractmethod
    def compute_root_kernel_prior(self):
        return

    @abstractmethod
    def compute_kernel_entropy(self):
        return
    
    @abstractmethod
    def remove_noise(self):
        return

    def set_parent(self, parent, reset=False):
        if self._parent is not None and self._parent._children is not None:
            self._parent._children.remove(self)
        if parent is not None:
            parent.add_child(self)
        self._parent = parent

        self.set_mean()

        if reset:
            self.reset_variational_parameters()

    def reset_parameters(self):
        pass

    def kill(self):
        if self._parent is not None:
            self._parent._children.remove(self)

        self._parent = None
        self._children = None

    def spawn(self, observed_parameters, seed=42):
        return self.__class__(
            observed_parameters, parent=self, tssb=self.tssb, seed=seed,
        )

    def get_observed_parameters(self):
        return self.observed_parameters
    
    def get_params(self):
        return self.params

    def has_data(self):
        if len(self.data):
            return True
        else:
            for child in self._children:
                if child.has_data():
                    return True
        return False

    def num_data(self):
        return reduce(
            lambda x, y: x + y,
            map(lambda c: c.num_data(), self._children),
            len(self.data),
        )

    def num_local_data(self):
        return len(self.data)

    def add_datum(self, id):
        self.data.add(id)

    def add_data(self, l):
        self.data.update(l)

    def remove_datum(self, id):
        self.data.remove(id)

    def remove_data(self):
        self.data.clear()

    def resample_params(self):
        pass

    def resample_root_params(self):
        pass

    def add_child(self, child):
        self._children.add(child)

    def remove_child(self, child):
        self._children.remove(child)

    def children(self):
        return self._children

    def get_data(self):
        return self.tssb.ntssb.data[list(self.data), :]

    def get_tssb_root(self):
        tssb = self.tssb

        def descend(root):
            if root["node"] == self:
                return root
            for child in root["children"]:
                out = descend(child)
                if out:
                    return out

        return descend(tssb.root)

    def logprob(self, x):
        return 0

    def data_log_likelihood(self):
        return self.complete_loglh()

    def data_log_likelihood_global_params(self, global_params):
        return self.complete_logprob_global_params(global_params)

    def sample(self, num_data=1):
        return rand(num_data, 2)

    def parent(self):
        return self._parent

    def global_param(self, key):
        if self.parent() is None:
            return self.__dict__[key]
        else:
            return self.parent().global_param(key)

    def get_ancestors(self, all=True):
        if self._parent is None or (not all and self.tssb != self._parent.tssb):
            return [self]
        else:
            ancestors = self._parent.get_ancestors(all=all)
            ancestors.append(self)
            return ancestors

    def get_descendants(self):
        l = []

        def descend(node):
            l.append(node)
            for child in node.children():
                descend(child)

        descend(self)
        return l

    def parameter_log_prior(self):
        return 0

    def set_event_string(self):
        pass

    def get_top_obs(self, q=70, idx=None):
        """
        Get data which is very well explained by this node's parameter
        """
        if idx is None:
            idx = jnp.arange(self.tssb.ntssb.num_data)
        if len(idx) == 0:
            return np.array([])
        lls = self.compute_loglikelihood(idx)
        top_obs = idx[np.where(lls > np.percentile(lls, q=q))[0]]
        return top_obs
    
    def reset_opt(self):
        # For adaptive optimization
        self.direction_states = self.initialize_direction_states()
        self.state_states = self.initialize_state_states()
        self.event_states = self.initialize_event_states()

    def init_new_node_kernel(self, **kwargs):
        return