import numpy as np
from .node import TrajectoryNode
from ...ntssb.observed_tree import *
from anndata import AnnData

class TrajectoryTree(ObservedTree):
    def __init__(self, **kwargs):
        super(TrajectoryTree, self).__init__(**kwargs)
        self.node_constructor = TrajectoryNode

    def sample_kernel(self, parent_params, mean_dist=1., angle_concentration=1., loc_variance=.1, seed=42, depth=1., **kwargs):
        rng = np.random.default_rng(seed=seed)
        parent_loc = parent_params[0]
        parent_angle = parent_params[1]
        angle_concentration = angle_concentration * depth
        sampled_angle = rng.vonmises(parent_angle, angle_concentration)
        sampled_loc = rng.normal(mean_dist, loc_variance)
        sampled_loc = parent_loc + np.array([np.cos(sampled_angle)*np.abs(sampled_loc), np.sin(sampled_angle)*np.abs(sampled_loc)])
        return [sampled_loc, sampled_angle]
    
    def sample_root(self, **kwargs):
        return [np.array([0., 0.]), 0.]

    def get_param_size(self):
        return self.tree["param"][0].size

    def get_params(self):
        params = []
        for node in self.tree_dict:
            params.append(self.tree_dict[node]["param"][0])
        return np.array(params, dtype=np.float)

    def param_distance(self, paramA, paramB):
        return np.sqrt(np.sum((paramA[0]-paramB[0])**2))

    def create_adata(self, var_names=None):
        params = []
        params_labels = []
        for node in self.tree_dict:
            if self.tree_dict[node]["size"] != 0:
                params_labels.append(
                    [self.tree_dict[node]["label"]] * self.tree_dict[node]["size"]
                )
                params.append(
                    np.vstack(
                        [self.tree_dict[node]["param"][0]] * self.tree_dict[node]["size"]
                    )
                )
        params = pd.DataFrame(np.vstack(params))
        params_labels = np.concatenate(params_labels).tolist()
        if var_names is not None:
            params.columns = var_names
        self.adata = AnnData(params)
        self.adata.obs["node"] = params_labels
        self.adata.uns["node_colors"] = [
            self.tree_dict[node]["color"]
            for node in self.tree_dict
            if self.tree_dict[node]["size"] != 0
        ]
        self.adata.uns["node_sizes"] = np.array(
            [
                self.tree_dict[node]["size"]
                for node in self.tree_dict
                if self.tree_dict[node]["size"] != 0
            ]
        )
        self.adata.var["bulk"] = np.mean(self.adata.X, axis=0)