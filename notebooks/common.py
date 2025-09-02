

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
from evoscaper.utils.normalise import calc_minmax
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def cluster_parameter_groups(df, eps=0.5, cols=['UMAP 1', 'UMAP 2'], min_cluster_size=5, min_samples=1, method='HDBSCAN',
                             n_true_clusters=5):
    if method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    elif method == 'KMeans':
        clusterer = KMeans(n_clusters=n_true_clusters, random_state=0, n_init='auto')
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = clusterer.fit_predict(df[cols])
    return df


def scale_norm(x, key, data, vmin=0, vmax=1):
    """Scale and normalize the data."""
    return calc_minmax(
        x, min_val=data[key].min(), scale=data[key].max() - data[key].min(),
        max_range=vmax, min_range=vmin)
    
    
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


def load_stitch_ys(dir_src_rugg, idx_output, n_samples, time_steps = 800):
    
    def check_expansion(ys_out, t_len):
        if ys_out.shape[1] < t_len:
            ys_out = np.concatenate(
                [ys_out, np.ones((ys_out.shape[0], t_len - ys_out.shape[1]), dtype=np.float32) * np.nan], axis=1)
        return ys_out
    
    batch_dirs = [c for c in os.listdir(dir_src_rugg) if c.startswith('batch')]
    batch_dirs.sort(key=lambda x: int(x.split('_')[1]))

    fn_ys = os.path.join(dir_src_rugg, 'ys.npy')
    fn_ys_out = os.path.join(dir_src_rugg, 'ys_out.npy')
    if os.path.exists(fn_ys):
        ys_out = np.load(fn_ys)[..., idx_output]
        ts = np.load(fn_ys.replace('ys', 'ts'))
    elif os.path.exists(fn_ys_out):
        ys_out = np.load(fn_ys_out)
        ts = np.load(fn_ys_out.replace('ys_out', 'ts'))
    else:
        ys_out = np.ones((n_samples, time_steps), dtype=np.float32) * np.nan
        t_max = 0
        for i, b in enumerate(batch_dirs):
            ys_i = np.load(os.path.join(dir_src_rugg, b, 'ys.npy'))
            if i == 0: 
                ts = np.load(os.path.join(dir_src_rugg, b, 'ts.npy'))

            t_len = ys_i.shape[1]
            if t_len > t_max:
                t_max = t_len
            ys_out = check_expansion(ys_out, t_len)
            ys_out[i * len(ys_i) : (i+1) * len(ys_i), :t_len] = ys_i[..., idx_output].astype(np.float32)
            if t_len < time_steps:
                ys_out[i * len(ys_i) : (i+1) * len(ys_i), t_len:] = ys_i[:, -1, idx_output].astype(np.float32)[:, None]
        if t_max < time_steps:
            ys_out = ys_out[:, :t_max]

        ys_out = ys_out.reshape(*ys_out.shape, 1)
        np.save(fn_ys_out, ys_out)
        np.save(fn_ys_out.replace('ys_out', 'ts'), ts)
    
    return ys_out, ts


def load_stitch_analytics(dir_src_rugg, last_idx=None):
    
    def add_properties(analytics_rugg):
        analytics_rugg['Log sensitivity'] = np.log10(
            analytics_rugg['sensitivity'])
        analytics_rugg['Log precision'] = np.log10(analytics_rugg['precision'])
        if 'adaptation' not in analytics_rugg.keys():
            analytics_rugg['adaptation'] = calculate_adaptation(
                analytics_rugg['sensitivity'], analytics_rugg['precision'], alpha=2)
        return analytics_rugg
        
    fn_analytics = os.path.join(dir_src_rugg, 'analytics.json')

    last_idx = last_idx if last_idx is not None else slice(None, None, 1)
    if os.path.exists(fn_analytics):
        analytics_rugg = load_json_as_dict(fn_analytics)
        for k, v in analytics_rugg.items():
            analytics_rugg[k] = np.array(v) #[..., last_idx]
        analytics_rugg = add_properties(analytics_rugg)
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
                    analytics_rugg[k] = np.array(v)[..., last_idx]
                else:
                    analytics_rugg[k] = np.concatenate(
                        [analytics_rugg[k], np.array(v)[..., last_idx]], axis=0)
        analytics_rugg = add_properties(analytics_rugg)

        write_json(analytics_rugg, fn_analytics)

    analytics_rugg.pop('RMSE', None)
    return analytics_rugg


def norm_rugg(rugg, rugg_training, y_datanormaliser, min_range=0, max_range=1, col_rugg='Log ruggedness (adaptation)'):
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
