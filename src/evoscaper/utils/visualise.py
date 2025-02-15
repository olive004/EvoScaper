

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import functools
import networkx as nx

from evoscaper.utils.math import bin_to_nearest_edge


def save_plot():
    """
    Decorator to automatically save matplotlib figures after function execution.

    Args:
        output_dir (str): Directory to save plots
        create_dir (bool): Create output directory if it doesn't exist
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, save_path=None, **kwargs):
            # Execute original function
            result = func(*args, **kwargs)

            # Save figure
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path, transparent=True, dpi=300)
            plt.close()  # Close to prevent memory leaks

            return result
        return wrapper
    return decorator


@save_plot()
def vis_sampled_histplot(analytic, all_species: List[str], output_species: List[str], category_array: bool,
                         title: str, x_label: str, multiple='fill', show=False, f=sns.histplot, **kwargs):
    if f == sns.histplot:
        for k, v in zip(('element', 'bins', 'log_scale'), ('step', 20, [True, False])):
            kwargs.setdefault(k, v)

    # category_array = np.array(sorted(y_datanormaliser.metadata[y_datanormaliser.cols_separate[0]]["category_map"].values())).repeat(
    #     len(analytic)//len(y_datanormaliser.metadata[y_datanormaliser.cols_separate[0]]["category_map"]))

    fig = plt.figure(figsize=(13, 4))
    fig.subplots_adjust(wspace=0.6)
    for i, output_specie in enumerate(output_species):
        title_curr = title + f': species {output_specie}'
        output_idx = all_species.index(output_specie)
        df_s = pd.DataFrame(columns=[x_label, 'VAE conditional input'],
                            data=np.concatenate([analytic[:, output_idx][:, None], category_array], axis=-1))
        df_s['VAE conditional input'] = df_s['VAE conditional input'].astype(
            float).apply(lambda x: f'{x:.2f}')
        ax = plt.subplot(1, 2, i+1)
        f(df_s, x=x_label,
          multiple=multiple, hue='VAE conditional input', palette='viridis',
          **kwargs)

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title(title_curr)

    if show:
        plt.show()


@save_plot()
def vis_training(saves):
    metrics = ['train_loss', 'val_loss', 'val_accuracy']
    plt.figure(figsize=(6*len(metrics), 5))
    for i, m in enumerate(metrics):
        ax = plt.subplot(1, len(metrics), i+1)
        plt.plot(list(map(float, saves.keys())), np.array(
            [float(v[m]) for v in saves.values()]))
        if 'loss' in m:
            plt.yscale('log')
        plt.ylabel(m)
        plt.xlabel('step')
    plt.suptitle('Training process')


@save_plot()
def vis_parity(y, pred_y):
    g = sns.scatterplot(x=pred_y.flatten(), y=y.flatten(), alpha=0.1, hue=np.sqrt(
        np.abs(pred_y.flatten() - y.flatten())), palette='viridis')
    g.legend_.set_title('Sqare root of difference')
    plt.title(
        f'Actual vs. predicted decoded circuits\nR2: {r2_score(y.flatten(), pred_y.flatten()):.2f}')
    plt.xlabel('Predicted circuit interactions')
    plt.ylabel('Actual circuit interactions')

    print('The R2 score is ', r2_score(y.flatten(), pred_y.flatten()))
    print('The R2 score with weighted variance is ', r2_score(
        y.flatten(), pred_y.flatten(), multioutput='variance_weighted'))


@save_plot()
def vis_recon_distribution(x, x_fake, n_to_sample: int):
    fig = plt.figure(figsize=(7*3, 8))
    fig.subplots_adjust(wspace=0.4)

    ax = plt.subplot(2, 3, 1)
    g = sns.histplot(x_fake, element='step', bins=30,
                     palette='viridis', multiple='fill')
    plt.title('Fake circuits')
    sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))

    ax = plt.subplot(2, 3, 2)
    g = sns.histplot(np.where(x_fake > x.max(), x.max(), np.where(x_fake < x.min(), x.min(), x_fake)),
                     element='step', bins=30,
                     palette='viridis', multiple='fill')
    plt.title('Fake circuits clipped')
    sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))

    ax = plt.subplot(2, 3, 3)
    g = sns.histplot(x.reshape(np.prod(x.shape[:-1]), x.shape[-1])[:n_to_sample],
                     element='step', bins=30, palette='viridis', multiple='fill')
    plt.title('Real circuits')
    sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))

    ax = plt.subplot(2, 3, 4)
    g = sns.histplot(x_fake, element='step', bins=30, palette='viridis',
                     multiple='layer', fill=False, log_scale=[False, True])
    plt.title('Fake circuits')
    sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))

    ax = plt.subplot(2, 3, 5)
    g = sns.histplot(np.where(x_fake > x.max(), x.max(), np.where(x_fake < x.min(), x.min(), x_fake)),
                     element='step', bins=30, palette='viridis',
                     multiple='layer', fill=False, log_scale=[False, True])
    plt.title('Fake circuits clipped')
    sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))

    ax = plt.subplot(2, 3, 6)
    x_hist = x.reshape(np.prod(x.shape[:-1]), x.shape[-1])[:n_to_sample]
    g2 = sns.histplot(x_hist, element='step', bins=30, palette='viridis',
                      multiple='layer', fill=False, log_scale=[False, True])
    plt.title('Real circuits')
    sns.move_legend(g2, 'upper left', bbox_to_anchor=(1, 1))

    g.set_ylim(g2.get_ylim())

    plt.suptitle(f'Interactions for CVAE: {n_to_sample} circuits')


@save_plot()
def vis_recon_scatter(x, x_fake, show_max: int = 2000):
    fig = plt.figure(figsize=(13, 4))

    ax = plt.subplot(1, 3, 1)
    sns.scatterplot(x_fake[:show_max], alpha=0.1)
    plt.title('Fake circuits')

    ax = plt.subplot(1, 3, 2)
    fake_circuits_1 = np.where(
        x_fake[:show_max] > x.max(), x.max(), x_fake[:show_max])
    sns.scatterplot(np.where(fake_circuits_1 < x.min(),
                    x.min(), fake_circuits_1), alpha=0.1)
    plt.title('Fake circuits within [min, max]')

    ax = plt.subplot(1, 3, 3)
    sns.scatterplot(
        x.reshape(np.prod(x.shape[:-1]), x.shape[-1])[:show_max], alpha=0.1)
    plt.title('Real circuits')

    plt.suptitle(f'CVAE: {show_max} circuits')


@save_plot()
def vis_histplot_combined_realfake(
        n_categories, df, x_cols, objective_col, y_datanormaliser, y_methods_preprocessing,
        fake_circuits, z, sampled_cond, is_onehot, multiple='fill', fill: bool = True, show_max=300):
    k = objective_col[0]
    df = df[~df[k].isna()]
    df[k + '_nearest_edge'] = bin_to_nearest_edge(
        df[k].to_numpy(), n_bins=n_categories)

    fig = plt.figure(figsize=(10*2, n_categories*5))
    for i, (zi, fake, edge) in enumerate(zip(z, fake_circuits, sorted(df[k + '_nearest_edge'].unique()))):
        real = df[df[k + '_nearest_edge']
                  == edge][x_cols].to_numpy()

        ax = plt.subplot(n_categories, 2, 2*i+1)
        g1 = sns.histplot(fake, element='step', bins=30,
                          palette='viridis', multiple=multiple, fill=fill)
        sc = np.array(sorted(np.unique(sampled_cond)))[:, None]
        sc = y_datanormaliser.create_chain_preprocessor_inverse(
            y_methods_preprocessing)(sc)
        if is_onehot:
            plt.title(
                f'{len(fake)} fake circuits: {k} = {y_datanormaliser.metadata[k]["category_map"][i]:.2f}')
        else:
            plt.title(
                f'{len(fake)} fake circuits: {k} = {str(sc.flatten()[i])[:6]}')
        sns.move_legend(g1, 'upper left', bbox_to_anchor=(1, 1))

        ax = plt.subplot(n_categories, 2, 2*i+2)
        g2 = sns.histplot(real, element='step', bins=30,
                          palette='magma', multiple=multiple, fill=fill)
        plt.title(
            f'{len(real)} real circuits: {k} = {str(edge)[:6]}')
        sns.move_legend(g2, 'upper left', bbox_to_anchor=(1, 1))

    plt.suptitle(f'CVAE: circuit comparison fake vs. real ({k})', fontsize=14)


def create_network_inset(fig, ax, pos=(0.3, -0.1), width=0.5, height=0.5,
                         edge_weights=None, node_color='lightblue', n_nodes=3):
    """
    Creates a 3-node network with curved bidirectional edges and extended self-loops.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes to place the inset in
    pos : tuple
        Position of the inset (x, y) in axes coordinates
    width, height : float
        Size of the inset in axes coordinates
    edge_weights : dict
        Dictionary of edge weights to determine opacity. Format: {(node1, node2): weight}
    node_color : str
        Color of the network nodes
    """
    # Create inset axes
    inset_axes = ax.inset_axes([pos[0], pos[1], width, height])

    # Create directed network
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])

    # Add bidirectional edges and self-loops
    # edges = [(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1),
    #          (1, 1), (2, 2), (3, 3)]
    edges = [tuple(sorted(ii)) for ii in zip(*[(i + 1).tolist()
                                               for i in np.triu_indices(n_nodes)])]
    G.add_edges_from(edges)

    # Set default edge weights if none provided
    if edge_weights is None:
        edge_weights = {(i, j): 0.7 for i, j in edges}

    # Position nodes in an equilateral triangle
    scale = 0.9  # Reduced to give more room for loops
    pos_nodes = {
        1: (-scale/2, -scale/2),
        2: (scale/2, -scale/2),
        3: (0, scale/2)
    }

    # Draw nodes
    node_size = 400
    nx.draw_networkx_nodes(G, pos_nodes, node_color=node_color,
                           node_size=node_size, ax=inset_axes)

    # Define edge styles
    curved_rad = 0.3     # Curvature of edges between nodes
    loop_rad = 0.5      # Much larger radius for extended self-loops

    # Draw edges with varying opacity and arrowheads
    for (u, v) in edges:
        weight = edge_weights.get((u, v), 0.7)

        if u == v:  # Self-loop
            # Customize self-loop appearance based on node position
            rad = loop_rad
            # if u == 3:  # Top node
            #     rad = loop_rad * 0.9  # Slightly smaller for top node

            # Draw self-loops with extended radius
            nx.draw_networkx_edges(G, pos_nodes, edgelist=[(u, v)],
                                   alpha=weight, width=1.5,
                                   arrowsize=10,
                                   connectionstyle=f'arc3, rad={rad}',
                                   arrowstyle='-|>',
                                   node_size=int(node_size * 1.8),
                                   ax=inset_axes)
        else:  # Curved edges between different nodes
            # Alternate curve direction for bidirectional edges
            rad = curved_rad if (u < v) else -curved_rad
            weight = weight / 2

            nx.draw_networkx_edges(G, pos_nodes, edgelist=[(u, v)],
                                   alpha=weight, width=1.5,
                                   arrowsize=10,
                                   connectionstyle=f'arc3, rad={rad}',
                                   arrowstyle='<|-|>',
                                   ax=inset_axes)

    # Add node labels
    nx.draw_networkx_labels(G, pos_nodes, ax=inset_axes)

    # Configure inset appearance
    inset_axes.set_xticks([])
    inset_axes.set_yticks([])
    inset_axes.set_facecolor('none')

    # Remove the border box
    for spine in inset_axes.spines.values():
        spine.set_visible(False)

    # Set fixed aspect ratio and expanded limits for larger loops
    inset_axes.set_aspect('equal')
    inset_axes.set_xlim(-1.2, 1.2)
    inset_axes.set_ylim(-1.2, 1.2)

    return inset_axes
