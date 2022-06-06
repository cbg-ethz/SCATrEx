"""
Embed a tree in a scatter plot of the data.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
