import numpy as np
import jax

def tree_to_dict(tree, param_key='param', root_par='-1'):
    """Converts a tree in a recursive dictionary to a flat dictionary with parent keys
    """
    tree_dict = dict()

    def descend(node, par_id):
        label = node['label']
        tree_dict[label] = dict()
        tree_dict[label]["parent"] = par_id
        for key in node:
            if key == param_key:
                key = 'param'
                tree_dict[label]['param'] = node[param_key]
            elif key == 'children':
                tree_dict[label][key] = [c['label'] for c in node[key]]
            elif key != 'parent':
                tree_dict[label][key] = node[key]
        for child in node['children']:
            descend(child, node['label'])

    descend(tree, root_par)

    return tree_dict

def dict_to_tree(tree_dict, root_name, param_key='param'):
    """Converts a tree in a flat dictionary with parent keys to a recursive dictionary
    """        
    # Make children in the flat dict
    for i in tree_dict:
        tree_dict[i]["children"] = []
    for i in tree_dict:
        for j in tree_dict:
            if tree_dict[j]["parent"] == i:
                tree_dict[i]["children"].append(j)

    # Make tree
    root = {}
    root['label'] = root_name
    for key in tree_dict[root_name]:
        # if isinstance(tree_dict[root_name][key], list):
        #     if isinstance(tree_dict[root_name][key][0], dict):
        #         continue
        root[key] = tree_dict[root_name][key]
    root['children'] = []

    # Recursively construct tree
    def descend(super_tree, label):
        for i, child in enumerate(tree_dict[label]["children"]):
            d = {}
            for key in tree_dict[child]:
                # if isinstance(tree_dict[child][key], list):
                #     if isinstance(tree_dict[child][key][0], dict):
                #         continue
                d[key] = tree_dict[child][key]
            d['children'] = []
            super_tree["children"].append(d)
            descend(super_tree["children"][-1], child)

    descend(root, root_name)
    return root

def condense_tree(tree, min_weight=0.1):
    """
    Traverse and choose with some prob whether to keep each node in the tree
    """
    def descend(root):
        to_keep = []
        for child in root['children']:
            descend(child)
            to_keep.append(int(child['weight'] > min_weight))
        if len(to_keep) > 0:
            to_remove_idx = [i for i in range(len(to_keep)) if to_keep[i] == 0]
            if 'weight' in root:
                root['weight'] += np.sum([r['weight'] for i, r in enumerate(root['children']) if to_keep[i]==0])
            if 'size' in root:
                root['size'] += int(np.sum([r['size'] for i, r in enumerate(root['children']) if to_keep[i]==0]))
            # Set children of source as children of target
            for child_to_remove in list(np.array(root['children'])[to_remove_idx]):
                for child_of_to_remove in child_to_remove['children']:
                    to_keep.append(1)
                    root['children'].append(child_of_to_remove)
            # Remove child
            root['children'] = list(np.array(root['children'])[np.where(np.array(to_keep))[0]])
    descend(tree)   

def subsample_tree(tree, keep_prob=0.5, seed=42):
    """
    Traverse and choose with some prob whether to keep each node in the tree
    """
    def descend(root, key):
        to_keep = []
        for child in root['children']:
            key, subkey = jax.random.split(key)
            descend(child, key)
            to_keep.append(int(jax.random.bernoulli(subkey, keep_prob)))
        if len(to_keep) > 0:
            to_remove_idx = [i for i in range(len(to_keep)) if to_keep[i] == 0]
            if 'weight' in root:
                root['weight'] += np.sum([r['weight'] for i, r in enumerate(root['children']) if to_keep[i]==0])
            if 'size' in root:
                root['size'] += int(np.sum([r['size'] for i, r in enumerate(root['children']) if to_keep[i]==0]))
            # Set children of source as children of target
            for child_to_remove in list(np.array(root['children'])[to_remove_idx]):
                for child_of_to_remove in child_to_remove['children']:
                    to_keep.append(1)
                    root['children'].append(child_of_to_remove)
            # Remove child
            root['children'] = list(np.array(root['children'])[np.where(np.array(to_keep))[0]])
    key = jax.random.PRNGKey(seed)
    descend(tree, key)    

def convert_phylogeny_to_clonal_tree(threshold):
    # Converts a phylogenetic tree to a clonal tree by choosing the main clades
    # according to some threshold
    raise NotImplementedError

def obs_rmse(ntssb1, ntssb2, param='observed'):
    ntssb1_obs = np.zeros(ntssb1.data.shape)
    ntssb2_obs = np.zeros(ntssb2.data.shape)

    ntssb1_nodes = ntssb1.get_nodes()
    for node in ntssb1_nodes:
        idx = np.where(ntssb1.assignments == node)
        ntssb1_obs[idx] = node.get_param(param)
    
    ntssb2_nodes = ntssb2.get_nodes()
    for node in ntssb2_nodes:
        idx = np.where(ntssb2.assignments == node)
        ntssb2_obs[idx] = node.get_param(param)

    return np.mean(np.sqrt(np.mean((ntssb1_obs - ntssb2_obs)**2, axis=1)))

def ntssb_distance(ntssb1, ntssb2):
    pdist1 = ntssb1.get_pairwise_obs_distances()
    pdist2 = ntssb2.get_pairwise_obs_distances()
    n_obs = pdist1.shape[0]
    return np.sqrt(2./(n_obs*(n_obs-1)) * np.sum((pdist1-pdist2)**2))


def subtree_to_tree_distance(subtree, tree):
    """subtree is a TSSB in the NTSSB
    tree is a dictionary containing the true tree that should be there
    To check that the substructure we find is close to the real structure and not just something 
    completely different
    """
    subtree_nodes = []
    tree_nodes = []
    dists = np.zeros((len(subtree_nodes), len(tree_nodes)))
    for i, subtree_node in enumerate(subtree_nodes):
        for j, tree_node in enumerate(tree_nodes):
            dists[i,j] = compute_distance(subtree_node, tree_node)
    return np.sqrt(np.mean(dists**2))

def print_tree(tree, tab='  '):
    def descend(root, depth=0):
        tabs = [tab] * depth
        tabs = ''.join(tabs)
        print(f"{tabs}{root['label']}")
        for child in root['children']:
            descend(child, depth=depth+1)
    descend(tree)