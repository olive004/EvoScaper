

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import functools
import os

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
def vis_sampled_histplot(analytic, y_datanormaliser, model_brn, output_species: List[str],
                         title: str, x_label: str, multiple='fill'):
    category_array = np.array(sorted(y_datanormaliser.metadata["category_map"].values())).repeat(
        len(analytic)//len(y_datanormaliser.metadata["category_map"]))

    fig = plt.figure(figsize=(13, 4))
    fig.subplots_adjust(wspace=0.6)
    for i, output_specie in enumerate(output_species):
        title = title + f': species {output_specie}'
        output_idx = [ii.name for ii in model_brn.species].index(output_specie)
        df_s = pd.DataFrame(columns=[x_label, 'VAE conditional input'],
                            data=np.concatenate([analytic[:, output_idx][:, None], category_array[:, None]], axis=-1))
        df_s['VAE conditional input'] = df_s['VAE conditional input'].astype(
            float).apply(lambda x: f'{x:.2f}')
        ax = plt.subplot(1, 2, i+1)
        sns.histplot(df_s, x=x_label, element='step',
                     log_scale=[True, False], multiple=multiple, bins=20, hue='VAE conditional input', palette='viridis')

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title(title)


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
    g = sns.scatterplot(x=y.flatten(), y=pred_y.flatten(), alpha=0.1, hue=np.sqrt(
        np.abs(pred_y.flatten() - y.flatten())), palette='viridis')
    g.legend_.set_title('Sqare root of difference')
    plt.title(
        f'Actual vs. predicted decoded circuits\nR2: {r2_score(y.flatten(), pred_y.flatten()):.2f}')
    plt.xlabel('Actual circuit binding energy')
    plt.ylabel('Predicted circuit binding energy')

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
        fake_circuits, z, sampled_cond, is_onehot, show_max=300):
    df = df[~df[objective_col].isna()]
    df[objective_col + '_nearest_edge'] = bin_to_nearest_edge(
        df[objective_col].to_numpy(), n_bins=n_categories)
    
    fig = plt.figure(figsize=(10*2, n_categories*5))
    for i, (zi, fake, edge) in enumerate(zip(z, fake_circuits, sorted(df[objective_col + '_nearest_edge'].unique()))):
        real = df[df[objective_col + '_nearest_edge']
                  == edge][x_cols].to_numpy()

        ax = plt.subplot(n_categories, 2, 2*i+1)
        g1 = sns.histplot(fake, element='step', bins=30,
                          palette='viridis', multiple='fill')
        sc = np.array(sorted(np.unique(sampled_cond)))[:, None]
        sc = y_datanormaliser.create_chain_preprocessor_inverse(
            y_methods_preprocessing)(sc)
        if is_onehot:
            plt.title(
                f'{len(fake)} fake circuits: {objective_col} = {y_datanormaliser.metadata["category_map"][i]:.2f}')
        else:
            plt.title(
                f'{len(fake)} fake circuits: {objective_col} = {str(sc.flatten()[i])[:6]}')
        sns.move_legend(g1, 'upper left', bbox_to_anchor=(1, 1))

        ax = plt.subplot(n_categories, 2, 2*i+2)
        g2 = sns.histplot(real, element='step', bins=30,
                          palette='magma', multiple='fill')
        plt.title(
            f'{len(real)} real circuits: {objective_col} = {str(edge)[:6]}')
        sns.move_legend(g2, 'upper left', bbox_to_anchor=(1, 1))

    plt.suptitle(f'CVAE: circuit comparison fake vs. real', fontsize=14)
