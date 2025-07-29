

import itertools
import numpy as np
from evoscaper.utils.math import bin_array
from evoscaper.model.vae import sample_z
import jax
import os
import umap
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation
from synbio_morpher.utils.data.data_format_tools.common import write_json
from evoscaper.utils.evolution import calculate_ruggedness_core
from evoscaper.utils.math import make_flat_triangle


def get_model_latent_space_dimred(p, rng, encoder, h2mu, h2logvar,
                                  x, cond, use_h,
                                  y_datanormaliser, y_methods_preprocessing,
                                  config_dataset, x_datanormaliser, x_methods_preprocessing,
                                  hpos_all,
                                  method='UMAP', n_show=1000, random_state=0,
                                  perplexity=30):

    # Initialize a PRNG key with a seed
    rng_key = jax.random.PRNGKey(random_state)

    h_all = encoder(p, rng, np.concatenate([x, cond], axis=-1))
    h_all = h_all.reshape(np.prod(h_all.shape[:-1]), -1)
    mu = h2mu(p, rng, h_all)
    logvar = h2logvar(p, rng, h_all)
    z_all = sample_z(mu, logvar, rng, deterministic=True)
    if use_h:
        emb = h_all
    else:
        emb = z_all

    cond_rev_all = np.concatenate([y_datanormaliser.create_chain_preprocessor_inverse(y_methods_preprocessing)(
        cond[..., i], col=c).flatten() for i, c in enumerate(config_dataset.objective_col)]).reshape(np.prod(cond.shape[:-1]), -1).squeeze()
    # cond_rev_all = y_datanormaliser.create_chain_preprocessor_inverse(y_methods_preprocessing)(cond, col=config_dataset.objective_col[0]).reshape(np.prod(cond.shape[:-1]), -1).squeeze()
    x_rev_all = x_datanormaliser.create_chain_preprocessor_inverse(
        x_methods_preprocessing)(x).reshape(np.prod(x.shape[:-1]), -1).squeeze()

    x_bin_all, edges, labels = bin_array(x_rev_all, num_bins=10)
    x_bin_all = np.round(x_bin_all, 1)

    cond_binned = cond
    if not hpos_all['prep_y_categorical']:
        cond_binned = bin_array(
            cond_rev_all, num_bins=hpos_all['prep_y_categorical_n_bins'])[0]
        if cond.shape[-1] != cond_binned.shape[-1]:
            cond_binned = cond_binned.reshape(-1, cond.shape[-1])

    cond_unique = [np.unique(cond_binned[..., i])
                   for i in range(cond_binned.shape[-1])]
    cond_unique = np.array(list(itertools.product(*cond_unique)))

    idxs_show = []
    for c in cond_unique:
        idxs_show.extend(np.where((cond_binned != c).sum(axis=-1) == 0)
                         [0][:np.max([n_show//len(cond_unique), 5])])
    idxs_show = np.array(idxs_show)
    if len(idxs_show) > n_show:
        idxs_show = jax.random.choice(
            rng_key, idxs_show, (n_show,), replace=False)

    if method == 'UMAP':
        reducer = umap.UMAP(n_neighbors=100, n_components=2, random_state=random_state,
                            #  metric='euclidean', n_epochs=2000, learning_rate=0.1, init='spectral')
                            metric='euclidean', n_epochs=2000, learning_rate=0.1, init='pca')
    else:
        reducer = TSNE(n_components=2, perplexity=perplexity,
                       random_state=random_state)
    result_dimred = reducer.fit_transform(emb[idxs_show])

    return result_dimred, idxs_show, cond_unique, cond_binned, x_bin_all, x_rev_all, cond_rev_all, emb, h_all, z_all


def load_stitch_analytics(dir_src_rugg):
    fn_analytics = os.path.join(dir_src_rugg, 'analytics.json')

    if os.path.exists(fn_analytics):
        analytics_rugg = load_json_as_dict(fn_analytics)
        for k, v in analytics_rugg.items():
            analytics_rugg[k] = np.array(v)
        if 'adaptation' not in analytics_rugg.keys():
            analytics_rugg['adaptation'] = calculate_adaptation(
                analytics_rugg['sensitivity'], analytics_rugg['precision'], alpha=2)
    else:
        # Stitch together ruggedness from batches
        analytics_rugg = {}
        batch_dirs = [c for c in os.listdir(dir_src_rugg) if c.startswith('batch')]
        batch_dirs.sort(key=lambda x: int(x.split('_')[1]))
        # for fn_analytic in ['analytics.json', 'analytics2.json']:
        for dir_batch in batch_dirs:
            if (not os.path.exists(os.path.join(dir_src_rugg, dir_batch))) or (
                    len(os.listdir(os.path.join(dir_src_rugg, dir_batch))) == 0):
                continue
            analytics_batch = load_json_as_dict(os.path.join(
                dir_src_rugg, dir_batch, 'analytics.json'))
            for k, v in analytics_batch.items():
                if k not in analytics_rugg:
                    analytics_rugg[k] = np.array(v)
                else:
                    analytics_rugg[k] = np.concatenate(
                        [analytics_rugg[k], np.array(v)], axis=0)

        analytics_rugg['Log sensitivity'] = np.log10(
            analytics_rugg['sensitivity'])
        analytics_rugg['Log precision'] = np.log10(analytics_rugg['precision'])
        if 'adaptation' not in analytics_rugg.keys():
            analytics_rugg['adaptation'] = calculate_adaptation(
                analytics_rugg['sensitivity'], analytics_rugg['precision'], alpha=2)

        write_json(analytics_rugg, fn_analytics)

    analytics_rugg.pop('RMSE', None)
    return analytics_rugg


def norm_rugg(rugg, rugg_training, y_datanormaliser, col_rugg='Log ruggedness (adaptation)'):
    # Robust scaling
    median = y_datanormaliser.metadata[col_rugg]['median']
    iqr = y_datanormaliser.metadata[col_rugg]['iqr']

    robust_scaled = (rugg - median) / iqr

    max_val = np.nanmax(rugg_training)
    min_val = np.nanmin(rugg_training)

    # Prevent division by zero
    scale = max_val - min_val
    scale = np.where(scale == 0, 1.0, scale)

    # Map to desired feature range
    min_range, max_range = (0, 1)
    rugg_norm = ((robust_scaled - y_datanormaliser.metadata[col_rugg]['min_val']) / y_datanormaliser.metadata[col_rugg]['scale']) * \
        (max_range - min_range) + min_range

    return rugg_norm


def load_rugg(all_fake_circuits, config_rugg, analytics_rugg):
    n_samples = all_fake_circuits.shape[0]
    n_interactions = make_flat_triangle(all_fake_circuits[0]).shape[-1]
    n_perturbs = n_interactions + config_rugg['resimulate_analytics']
    eps = config_rugg['eps_perc'] * np.abs(all_fake_circuits).max()

    ruggedness = {}
    for analytic in analytics_rugg.keys():
        ruggedness[analytic] = calculate_ruggedness_core(analytics_rugg, None, analytic,
                                                         config_rugg['resimulate_analytics'], n_samples, n_perturbs, eps)

    if config_rugg['resimulate_analytics']:
        # n_max = n_samples * n_perturbs
        analytics_og = {k: np.array(v).reshape(
            n_samples, n_perturbs, -1)[:, -1, :] for k, v in analytics_rugg.items()}
    else:
        analytics_og = {}

    k_rugg = 'Log ruggedness (adaptation)'

    ruggedness[k_rugg] = np.log10(ruggedness['Log sensitivity'])

    return ruggedness, analytics_og, n_samples, n_perturbs, n_interactions, eps, k_rugg


def make_df_rugg(analytics_og, ruggedness, idx_output, all_sampled_cond,
                 y_datanormaliser, y_methods_preprocessing, config_dataset, k_rugg):

    df_rugg = pd.DataFrame()
    
    for col in ['overshoot', 'initial_steady_states', 'steady_states', 'Log sensitivity', 'Log precision', 'adaptation', 'response_time']:
        df_rugg[col] = analytics_og[col][..., idx_output]
        
    df_rugg['Prompt Adaptation'] = all_sampled_cond[...,
                                                    config_dataset.objective_col.index('adaptation')].flatten()
    df_rugg['Prompt Ruggedness'] = all_sampled_cond[...,
                                                    config_dataset.objective_col.index(k_rugg)].flatten()
    df_rugg['Prompt Ruggedness Unnorm'] = y_datanormaliser.create_chain_preprocessor_inverse(
        y_methods_preprocessing)(all_sampled_cond[..., config_dataset.objective_col.index(k_rugg)].flatten(), col=k_rugg)
    
    for col_rugg in ['adaptation', 'Log sensitivity', 'Log precision']:
        df_rugg[f'Log ruggedness ({col_rugg})'] = np.where(
            ruggedness[col_rugg][..., idx_output] == 0, np.nan, np.log10(ruggedness[col_rugg][..., idx_output]))

    df_rugg['Log ruggedness (adaptation) norm'] = y_datanormaliser.create_chain_preprocessor(
        y_methods_preprocessing)(df_rugg['Log ruggedness (adaptation)'].values, col=k_rugg, use_precomputed=True)

    for col_bin in ['Log ruggedness (adaptation)', 'adaptation']:
        df_rugg[f'{col_bin} bin'] = pd.cut(
            df_rugg[col_bin], bins=10)
        df_rugg[f'{col_bin} bin'] = df_rugg[f'{col_bin} bin'].apply(
            lambda x: x.mid).astype(float).round(2)
        
    df_rugg.loc[np.isinf(df_rugg['Log precision']), 'Log precision'] = np.nan

    return df_rugg
