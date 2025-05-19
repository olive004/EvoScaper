

from typing import List
import numpy as np
import jax
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
        def wrapper(*args, save_path=None, show=False, **kwargs):
            # Execute original function
            result = func(*args, **kwargs)

            # Save figure
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path, transparent=True, dpi=300)

            if show:
                plt.show()

            plt.close()  # Close to prevent memory leaks

            return result
        return wrapper
    return decorator


@save_plot()
def vis_sampled_histplot(analytic, all_species: List[str], output_species: List[str], category_array: bool,
                         title: str, x_label: str, multiple='fill', f=sns.histplot,
                         include_hue_vlines=False, vline_uniqs=None, hue_label='Conditional input', 
                         figsize=(13, 5), **kwargs):
    if f == sns.histplot:
        for k, v in zip(('element', 'bins', 'log_scale'), ('step', 20, [True, False])):
            kwargs.setdefault(k, v)

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.6)
    for i, output_specie in enumerate(output_species):
        title_curr = title + f': species ${output_specie}$'
        output_idx = all_species.index(output_specie)
        df_s = pd.DataFrame(columns=[x_label] + [f'Conditional input {ii}' for ii in range(category_array.shape[-1])],
                            data=np.concatenate([analytic[:, output_idx][:, None], category_array], axis=-1))
        for ii in range(category_array.shape[-1]):
            df_s[f'Conditional input {ii}'] = df_s[f'Conditional input {ii}'].astype(
                float).apply(lambda x: f'{x:.1f}')
        df_s['Conditional input'] = df_s[[f'Conditional input {ii}' for ii in range(
            category_array.shape[-1])]].apply(lambda x: ', '.join(x), axis=1)

        ax = plt.subplot(1, 2, i+1)
        f(df_s, x=x_label,
          multiple=multiple, hue='Conditional input', palette='viridis',
          **kwargs)

        c_uniq = sorted(
            np.unique(category_array[:, 0])) if vline_uniqs is None else vline_uniqs
        if include_hue_vlines:
            colors = sns.color_palette('viridis', len(c_uniq))
            for ih, hue_val in enumerate(c_uniq):
                # , label=f'{hue_val} mean')
                plt.axvline(hue_val, linestyle='--', color=colors[ih])

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=hue_label)
        if i != (len(output_species) - 1):
            ax.legend_.remove()
        plt.title(title_curr)


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


def make_nx_weights(energies, n_nodes, vmin=None, vmax=None):
    vmin = vmin if vmin is not None else energies.min()
    vmax = vmax if vmax is not None else energies.max()
    energies_mod = np.interp(
        energies, (vmin, vmax), (1, 0))
    keys = [tuple(sorted(ii)) for ii in zip(*[(i + 1).tolist()
                                              for i in np.triu_indices(n_nodes)])]
    weights = dict(zip(keys, energies_mod.round(2).tolist()))
    return weights


def create_network_inset(fig, ax, pos=(0.3, -0.1), width=0.5, height=0.5,
                         edge_weights=None, node_color='lightblue', n_nodes=3,
                         linewidth=1.5):
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
    G.add_nodes_from(np.arange(n_nodes) + 1)

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
        1: (0, scale/3),
        2: (-scale/2, -scale/2),
        3: (scale/2, -scale/2)
    }

    # Draw nodes
    node_size = 400
    nx.draw_networkx_nodes(G, pos_nodes, node_color=node_color,
                           node_size=node_size, ax=inset_axes,
                           edgecolors='black')

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
                                   alpha=weight, width=linewidth,
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
                                   alpha=weight, width=linewidth,
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


# Visualise 2D latent space with custom labels
# This function is used to visualize the 2D latent space of a model using t-SNE or UMAP.

def make_sort_hue(hue, sort, sort_random=False, sort_flip_prop=4):
    """ A sort_flip_prop of 4 means that the first 1/4 of the hue values are sent to the
    2/4 - 3/4 position in the sorted idxs, making the two most extreme quarters next to each other """
    if sort:
        idxs_hue = np.argsort(hue)[::-1]
        ib = len(idxs_hue)//sort_flip_prop
        ij = (sort_flip_prop - 1) * ib
        idxs_hue[:ij] = np.concatenate([idxs_hue[ib:ij], idxs_hue[:ib][::-1]])
    else:
        idxs_hue = np.arange(len(hue))
    if sort_random:
        idxs_hue = np.array(jax.random.choice(jax.random.PRNGKey(
            0), idxs_hue, idxs_hue.shape, replace=False))
    return idxs_hue


def visualize_dimred_2d_custom_labels(dimred_result, cond, x_bin, labels_cond, labels_x: list, method='TSNE', save_path=None,
                                      sort=False, s=10, use_h=False, x_type='energies', sort_random=False):
    ncols = 4 if len(labels_cond) > 1 else (x_bin.shape[-1] + 1)
    nrows = len(labels_cond)
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        5*ncols, 4*nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = axes[None, :]
        cond = cond[:, None]

    # Cond plots on the left
    for i, l in enumerate(labels_cond):
        ax_main = fig.add_subplot(axes[i, 0])  # Span both rows
        idxs_hue = make_sort_hue(
            hue=cond[:, i], sort=sort, sort_random=sort_random)
        scatter = ax_main.scatter(
            dimred_result[idxs_hue, 0], dimred_result[idxs_hue, 1], c=cond[idxs_hue, i], cmap='viridis', alpha=0.5, s=s)
        ax_main.set_title(
            f'{method} clusters {l}', fontsize=14)
        ax_main.set_xlabel(f'{method} Dimension 1', fontsize=12)
        ax_main.set_ylabel(f'{method} Dimension 2', fontsize=12)
        plt.colorbar(scatter, ax=ax_main, label=l)

    # Interaction plots on the right
    for i in range(x_bin.shape[-1]):
        row = i // (ncols-1)
        col = i % (ncols-1) + 1
        ax = fig.add_subplot(axes[row, col])
        idxs_hue = make_sort_hue(
            hue=x_bin[:, i], sort=sort, sort_random=sort_random)
        scatter = ax.scatter(
            dimred_result[idxs_hue, 0], dimred_result[idxs_hue, 1], c=x_bin[idxs_hue, i], cmap='plasma', alpha=0.5, s=s)
        # ax.set_title(' + '.join(labels_x[i]), fontsize=14)
        ax.set_title(labels_x[i], fontsize=14)
        ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
        # if i == (x_bin.shape[-1] - 1):
        plt.colorbar(
            scatter, ax=ax, label=f'Energy (kcal/mol)' if x_type == 'energies' else x_type)

    plt.suptitle(
        f'{method} visualization of latent space ({"h" if use_h else "z"})', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True)
    plt.show()


def visualize_dimred_adapt_sp(dimred_result, cond, x_bin, labels_cond, labels_x: list, method='TSNE', save_path=None,
                              sort=False, s=10, use_h=False, sort_random=False):
    """ labels_cond and cond should be [adaptation, log sensitivity, log precision] """

    # labels_cond = [c.replace('Log', r'$Log_{10}$') for c in labels_cond]

    def small_plot(ax, hue, cbar_label, title, idxs_hue, cmap):
        scatter = ax.scatter(
            dimred_result[idxs_hue, 0], dimred_result[idxs_hue, 1], c=hue, cmap=cmap, alpha=0.5, s=s//4)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f'{method} Dimension 1', fontsize=10)
        ax.set_ylabel(f'{method} Dimension 2', fontsize=10)
        plt.colorbar(scatter, ax=ax, label=cbar_label)

    fig = plt.figure(figsize=(23, 7))
    gs = fig.add_gridspec(2, 5, width_ratios=[2, 1, 1, 1, 1])

    # Main plot on the left
    ax_main = fig.add_subplot(gs[:, 0])  # Span both rows
    scatter = ax_main.scatter(
        dimred_result[:, 0], dimred_result[:, 1], c=cond[:, 0], cmap='viridis', alpha=0.5, s=s)
    ax_main.set_title(
        f'{method} clusters by prompt ({labels_cond[0]})', fontsize=16)
    ax_main.set_xlabel(f'{method} Dimension 1', fontsize=12)
    ax_main.set_ylabel(f'{method} Dimension 2', fontsize=12)
    plt.colorbar(scatter, ax=ax_main, label=labels_cond[0].capitalize())

    # Smaller plots for sens + prec
    for i, l in enumerate(labels_cond[1:]):
        row = i
        col = 1
        ax = fig.add_subplot(gs[row, col])
        idxs_hue = make_sort_hue(hue=cond[:, i+1], sort=sort)
        small_plot(ax, hue=cond[idxs_hue, i+1], cbar_label=l,
                   title=l, idxs_hue=idxs_hue, cmap='viridis')

    # Smaller plots on the right
    for i in range(x_bin.shape[-1]):
        row = i // 3
        col = i % 3 + 2
        ax = fig.add_subplot(gs[row, col])
        idxs_hue = make_sort_hue(hue=x_bin[:, i], sort=sort)
        small_plot(ax, hue=x_bin[idxs_hue, i], cbar_label=f'Energy (kcal/mol)',
                   title=labels_x[i], idxs_hue=idxs_hue, cmap='plasma')
        #    title=' + '.join(labels_x[i]), idxs_hue=idxs_hue, cmap='plasma')

    plt.suptitle(
        f'{method} visualization of latent space ({"h" if use_h else "z"})', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True)
    plt.show()
