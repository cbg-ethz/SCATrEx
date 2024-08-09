"""
Embed a tree in a scatter plot of the data.
"""
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ..utils.tree_utils import tree_to_dict


def plot_full_tree(tree, ax=None, figsize=(6,6), subtree_parent_probs=None, edge_labels=True, font_size=12, **kwargs):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    def descend(root, graph, pos={}):
        if subtree_parent_probs is not None:
            pos_out = plot_tree(root['node'], G=graph, ax=ax, alpha=1., draw=False, parent_probs=subtree_parent_probs[root['label']], edge_labels=edge_labels, font_size=font_size, **kwargs) # Draw subtree
        else:
            pos_out = plot_tree(root['node'], G=graph, ax=ax, alpha=1., draw=False, edge_labels=edge_labels, font_size=font_size, **kwargs) # Draw subtree
        pos.update(pos_out)
        for child in root['children']:
            descend(child, graph, pos)

        def sub_descend(sub_root, graph):
            parent = sub_root['label']
            for i, super_child in enumerate(root['children']):
                child = super_child['label']
                prob = sub_root['pivot_probs'][i]
                graph.add_edge(parent, child, alpha=prob, ls='--')
                nx.draw_networkx_edges(graph, pos, edgelist=[(parent, child)], edge_color=sub_root['color'], alpha=prob, style='--')
                if edge_labels and prob > 0.01:
                    nx.draw_networkx_edge_labels(graph, pos, font_color=sub_root['color'], edge_labels={(parent, child):f"{prob:.3f}"}, font_size=int(font_size/2), alpha=float(prob))
            for child in sub_root['children']:
                sub_descend(child, graph)

        if len(root['children']) > 0:
            sub_descend(root['node'], graph) # Draw pivot edges

    G = nx.DiGraph()
    descend(tree, G)

    ax.margins(0.20) # Set margins for the axes so that nodes aren't clipped
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_tree(tree, G = None, param_key='param', data=None, labels=True, alpha=0.5, font_size=12, node_size=1500, edge_width=1., arrows=True, draw=True, ax=None, parent_probs=None, edge_labels=True):
    """
    parent_probs is a pandas dataframe containing the probability of each node being the child of every other node
    """
    tree_dict = tree_to_dict(tree, param_key=param_key)

    # Get all positions
    pos = {}
    pos[tree['label']] = tree[param_key]
    for node in tree_dict:
        if tree_dict[node]['parent'] != '-1':
            pos[node] = tree_dict[node]['param']

    # Draw graph
    node_options = {'alpha': alpha,
                     'node_size': node_size,}
    edge_options = {'width': edge_width,
                    'node_size':node_size,
                    'arrows': arrows}
    label_options = {'alpha': alpha,
                     'font_size': font_size,}

    if ax is None:
        fig = plt.figure(figsize=(6,6))

    if G is None:
        G = nx.DiGraph()
    
    for node in tree_dict:
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=tree_dict[node]['color'],
                                **node_options)
        if tree_dict[node]['parent'] != '-1':
            if parent_probs is not None:
                for parent in tree_dict:
                    G.add_edge(parent, node)
                    nx.draw_networkx_edges(G, pos, edgelist=[(parent, node)], edge_color=tree_dict[parent]['color'], alpha=parent_probs.loc[node, parent]*alpha, **edge_options)
                    if edge_labels and parent_probs.loc[node, parent] > 0.01:
                        nx.draw_networkx_edge_labels(G, pos, edge_labels={(parent, node):f'{parent_probs.loc[node, parent]:.3f}'}, font_color=tree_dict[parent]['color'], 
                                                     font_size=int(font_size/2), alpha=parent_probs.loc[node, parent]*alpha)
            else:
                parent = tree_dict[node]['parent']
                G.add_edge(parent, node)
                nx.draw_networkx_edges(G, pos, edgelist=[(parent, node)], edge_color=tree_dict[parent]['color'], alpha=alpha, **edge_options)
                
    if labels:
        labs = dict(zip(list(tree_dict.keys()), list(tree_dict.keys())))
        nx.draw_networkx_labels(G, pos, labs, **label_options)

    ax = plt.gca()
    ax.margins(0.20) # Set margins for the axes so that nodes aren't clipped
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if draw:
        plt.show()
    else:
        return pos

def plot_nested_tree(tree, top=True, param_key='param', out_alpha=0.4, in_alpha=1., large_node_size=5000, small_node_size=500, draw=True, ax=None, **kwargs):
    tree_dict = tree_to_dict(tree, param_key=param_key)

    if top:
        # Plot main tree with transparency, large nodes and without labels
        ax = plot_tree(tree, param_key=param_key, labels=False, node_size=large_node_size, alpha=out_alpha, draw=False, **kwargs)

    for subtree in tree_dict: # Do this in a tree traversal so that we add the pivots
        # Plot each subtree
        plot_tree(tree_dict[subtree]['node'], param_key='mean', labels=False, node_size=small_node_size, alpha=in_alpha, ax=ax, draw=False, **kwargs)

    if draw:
        plt.show()


def plot_tree_proj(
    proj,
    tree,
    pca_obj=None,
    title="Tree",
    ax=None,
    fontsize=16,
    line_width=0.001,
    head_width=0.003,
    node_logit=None,
    weights=False,
    fc="k",
    ec="k",
    s=10,
    legend_fontsize=16,
    figsize=None,
    save=None,
):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    inf_node = np.array([assignment.label for assignment in tree.assignments])

    # Plot data points with estimated assignments
    # ass_logits = jnp.array([node.data_ass_logits for node in tree.get_nodes()]).reshape(tree.num_data, -1)
    # ass_probs = jnn.softmax(ass_logits, axis=1)
    # inf_node = jnp.argmax(ass_probs, axis=1)
    nodes, mixture = tree.get_node_mixture()
    b_idx = None
    if node_logit is not None:
        b_idx = np.where(np.array([node.label for node in nodes]) == node_logit)[0][0]
    if b_idx is not None:
        plt.scatter(
            proj[:, 0], proj[:, 1], s=s, alpha=0.3, c=nodes[b_idx].data_ass_logits
        )
    for i, n in enumerate(nodes):
        l = n.label
        idx = np.where(inf_node == l)[0]
        if b_idx is None:
            lab = l
            if weights:
                lab = f"{lab.replace('-', '')} ({mixture[i]:.2f})"
            color = n.tssb.color
            shape = "." if n.is_observed else "*"
            plt.scatter(
                proj[idx, 0],
                proj[idx, 1],
                label=lab.replace("-", ""),
                s=s,
                alpha=0.3,
                color=color,
                marker=shape,
            )
        if pca_obj is not None:
            mean = pca_obj.transform(
                np.log(n.node_mean.reshape(1, -1) * 1e4 + 1)
            ).ravel()
        else:
            mean = np.mean(proj[idx], axis=0)
        mean_1 = mean[0]
        mean_2 = mean[1]
        plt.scatter(mean_1, mean_2, color="gray", s=0.1)
        ax.annotate(
            l.replace("-", ""),
            xy=[mean_1, mean_2],
            textcoords="data",
            fontsize=fontsize,
        )
        if n.parent() is not None:
            p = n.parent()
            if pca_obj is not None:
                pmean = pca_obj.transform(
                    np.log(p.node_mean.reshape(1, -1) * 1e4 + 1)
                ).ravel()
            else:
                pidx = np.where(inf_node == p.label)[0]
                pmean = np.mean(proj[pidx], axis=0)
            pmean_1 = pmean[0]
            pmean_2 = pmean[1]
            px, py = pmean_1, pmean_2
            nx, ny = mean_1, mean_2
            dx, dy = nx - px, ny - py
            plt.arrow(
                px, py, dx, dy, fc=fc, ec=ec, width=line_width, head_width=head_width
            )

    plt.legend()
    # b_pca = pca_obj.transform(n.baseline_caller().reshape(1,-1)).ravel()
    # plt.scatter(b_pca[0], b_pca[1], color='gray', marker='x')
    # plt.legend(bbox_to_anchor=[1, 1], fontsize=legend_fontsize)
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(title)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    return ax
