import numpy as np
from numpy.random import *


class AbstractNode(object):
    def __init__(
        self, is_observed, observed_parameters, parent=None, tssb=None, label=""
    ):
        self.data = set([])
        self._children = set([])
        self.tssb = tssb
        self.is_observed = is_observed
        self.observed_parameters = observed_parameters
        self.label = label
        self.event_str = ""

        self.params = dict()
        self.variational_parameters = dict(globals=dict(), locals=dict())

        if parent is not None:
            parent.add_child(self)
            self._parent = parent
            n_siblings = len(list(parent.children()))
            self.label = parent.label + "-" + str(n_siblings - 1)
        else:
            self._parent = None

    def set_parent(self, parent, reset=False):
        if self._parent is not None and self._parent._children is not None:
            self._parent._children.remove(self)
        if parent is not None:
            parent.add_child(self)
        self._parent = parent

        if not self.is_observed:
            # Make sure we use the right observed parameters
            self.observed_parameters = parent.observed_parameters
            self.inherit_parameters()

        self.unobserved_factors = parent.unobserved_factors + self.unobserved_factors
        self.set_mean()

        if reset:
            self.reset_variational_parameters()

    def inherit_parameters(self):
        pass

    def reset_parameters(self):
        pass

    def kill(self):
        if self._parent is not None:
            self._parent._children.remove(self)

        self._parent = None
        self._children = None

    def spawn(self, is_observed, observed_parameters):
        return self.__class__(
            is_observed, observed_parameters, parent=self, tssb=self.tssb
        )

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
        if self._parent is None or (not all and self.is_observed):
            return [self]
        else:
            ancestors = self._parent.get_ancestors(all=all)
            ancestors.append(self)
            return ancestors

    def parameter_log_prior(self):
        return 0

    def tssb_caller(self):
        if self.parent() is None:
            return self.tssb
        else:
            return self.parent().tssb_caller()

    def set_event_string(self):
        pass
