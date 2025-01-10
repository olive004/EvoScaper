

from typing import Iterable
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from evoscaper.utils.normalise import make_chain_f
from evoscaper.utils.dataclasses import NormalizationSettings, FilterSettings
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation


def init_data(data, x_cols: Iterable[str], objective_cols: Iterable[str], OUTPUT_SPECIES: list,
              TOTAL_DS_MAX, BATCH_SIZE, rng,
              x_norm_settings: NormalizationSettings,
              y_norm_settings: NormalizationSettings,
              filter_settings: FilterSettings
              ):

    df = prep_data(data, OUTPUT_SPECIES, objective_cols, x_cols, filter_settings)
    TOTAL_DS, N_BATCHES, BATCH_SIZE = adjust_total_ds(df, BATCH_SIZE, TOTAL_DS_MAX)
    df = df.iloc[jax.random.choice(rng, np.arange(len(df)), [TOTAL_DS], replace=False)]

    x, cond, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing = make_xy(df, TOTAL_DS, x_cols, objective_cols,
                                                                                                            x_norm_settings, y_norm_settings)

    return df, x, cond, TOTAL_DS, N_BATCHES, BATCH_SIZE, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing


def adjust_total_ds(df, BATCH_SIZE, TOTAL_DS_MAX):
    TOTAL_DS = int(np.min([TOTAL_DS_MAX, len(df)]))
    if TOTAL_DS < BATCH_SIZE:
        print(f'TOTAL_DS is less than BATCH_SIZE: {TOTAL_DS} < {BATCH_SIZE}')
        BATCH_SIZE = TOTAL_DS
    else:
        TOTAL_DS = int(TOTAL_DS // BATCH_SIZE * BATCH_SIZE)
    if TOTAL_DS == 0:
        raise ValueError('TOTAL_DS is 0')
    N_BATCHES = int(TOTAL_DS // BATCH_SIZE)
    return TOTAL_DS, N_BATCHES, BATCH_SIZE


def prep_data(data, output_species, col_y, cols_x, filter_settings: FilterSettings):

    data = embellish_data(data)
    df = filter_invalids(data, output_species, cols_x,
                         col_y, filter_settings)
    df = reduce_repeat_samples(
        df, cols_x, n_same_circ_max=filter_settings.filt_n_same_x_max, nbin=filter_settings.filt_n_same_x_max_bins)
    return df


def embellish_data(data, transform_sensitivity_nans=True, zero_log_replacement=-10.0):
    if 'adaptation' not in data.columns:
        data['adaptation'] = calculate_adaptation(
            s=data['sensitivity_wrt_species-6'].values,
            p=data['precision_wrt_species-6'].values)
    if transform_sensitivity_nans:
        data['sensitivity_wrt_species-6'] = np.where(np.isnan(
            data['sensitivity_wrt_species-6']), 0, data['sensitivity_wrt_species-6'])
    def make_log(k, data):
        data[f'Log {k.split("_")[0]}'] = zero_log_replacement
        data.loc[data[k] != 0, f'Log {k.split("_")[0]}'] = np.log10(data[data[k] != 0][k])
        return data
    data = make_log('sensitivity_wrt_species-6', data)
    data = make_log('precision_wrt_species-6', data)
    return data


# Make xy
def make_xy(df, TOTAL_DS, X_COLS, objective_cols: Iterable[str],
            x_norm_settings, y_norm_settings):
    
    x, x_datanormaliser, x_methods_preprocessing = make_x(
        df, X_COLS, x_norm_settings)
    cond, y_datanormaliser, y_methods_preprocessing = make_y(
        df, objective_cols, y_norm_settings)

    # shuffled_indices = jax.random.permutation(rng, x.shape[0])
    # x, cond = x[shuffled_indices], cond[shuffled_indices]

    if x.shape[0] < TOTAL_DS:
        print(
            f'WARNING: The filtered data is not as large as the requested total dataset size: {x.shape[0]} vs. requested {TOTAL_DS}')

    return x, cond, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing


def make_x(df, X_COLS, x_norm_settings):
    x = [df[i].values[:, None] for i in X_COLS]
    x = np.concatenate(x, axis=-1).squeeze()

    x_datanormaliser, x_methods_preprocessing = make_chain_f(x_norm_settings)

    x = x_datanormaliser.create_chain_preprocessor(x_methods_preprocessing)(x)
    return x, x_datanormaliser, x_methods_preprocessing

    
def get_conds(col, df, y_datanormaliser, y_methods_preprocessing):
    cond = df[col].to_numpy()[:, None]
    cond = y_datanormaliser.create_chain_preprocessor(
        y_methods_preprocessing)(cond, col=col)
    return cond


def concat_conds(objective_cols, df, y_datanormaliser, y_methods_preprocessing):
    cond = get_conds(objective_cols[0], df, y_datanormaliser, y_methods_preprocessing)
    for k in objective_cols[1:]:
        cond = np.concatenate([cond, get_conds(k, df, y_datanormaliser, y_methods_preprocessing)], axis=-1)
    return cond


def make_y(df, objective_cols, y_norm_settings):

    y_datanormaliser, y_methods_preprocessing = make_chain_f(y_norm_settings, cols=objective_cols)
    cond = concat_conds(objective_cols, df, y_datanormaliser, y_methods_preprocessing)
    
    return cond, y_datanormaliser, y_methods_preprocessing


def make_training_data(x, cond, train_split, n_batches, batch_size):
    def f_reshape(i): return i.reshape(n_batches, batch_size, i.shape[-1])
    x, cond, y = f_reshape(x), f_reshape(cond), f_reshape(x)

    if n_batches == 1:
        def f_train(i): return i[:, :int(train_split * batch_size)]
        def f_val(i): return i[:, int(train_split * batch_size):]
    else:
        def f_train(i): return i[:int(np.max([train_split * n_batches, 1]))]
        def f_val(i): return i[int(np.max([train_split * n_batches, 1])):]
    x_train, cond_train, y_train = f_train(x), f_train(cond), f_train(y)
    x_val, cond_val, y_val = f_val(x), f_val(cond), f_val(y)

    return x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val


# Balance preprocess

def filter_invalids(data, OUTPUT_SPECIES, X_COLS, objective_cols, filter_settings: FilterSettings):

    filt = data['sample_name'].isin(OUTPUT_SPECIES)
    if filter_settings.filt_x_nans:
        filt = filt & data[X_COLS].notna().all(axis=1)
    if filter_settings.filt_y_nans:
        for k in objective_cols:
            filt = filt & data[k].notna() & (
            np.abs(data[k]) < np.inf)
    if filter_settings.filt_sensitivity_nans:
        filt = filt & (np.abs(data['sensitivity_wrt_species-6'])
                       < np.inf) & data['sensitivity_wrt_species-6'].notna()
    if filter_settings.filt_precision_nans:
        filt = filt & (np.abs(data['precision_wrt_species-6'])
                       < np.inf) & data['precision_wrt_species-6'].notna()

    df = data[filt]
    df = df.reset_index(drop=True)

    return df


def reduce_repeat_samples(df, cols, n_same_circ_max: int = 1, nbin=None):
    df = df.reset_index(drop=True)
    df_lowres = df if nbin is None else transform_to_histogram_bins(
        df, cols, num_bins=nbin)
    df_lowres = df_lowres.groupby(cols, as_index=False).head(n_same_circ_max)
    return df.loc[df_lowres.index].reset_index(drop=True)


def reduce_repeat_samples_old(rng, df, X_COLS, n_same_circ_max: int = 1, nbin=None):
    df = df.reset_index(drop=True)

    n_same_circ_max = 100
    nbin = 300
    def agg_func(x): return np.sum(x, axis=1)
    def agg_func(x): return tuple(x)

    df.loc[:, X_COLS] = df[X_COLS].apply(lambda x: np.round(x, 1))
    df_bal = balance_dataset(rng, df, cols=X_COLS, nbin=nbin,
                             bin_max=n_same_circ_max, use_log=False, func1=agg_func)
    df_bal = df_bal.reset_index(drop=True)
    return df_bal


def pre_balance(df, cols, use_log, func1):
    d = np.log10(df[cols].to_numpy()) if use_log else df[cols].to_numpy()
    if func1 is not None:
        d = func1(d)
    if use_log:  # keep -inf
        d = np.where(d < -100.0, -100.0, d)
        d = np.where(d > 100.0, 100.0, d)
    return d

# Indexes


def find_idxs_keep(rng, bin_edges, i, d, bin_max, to_keep):
    edge_lo, edge_hi = bin_edges[i], bin_edges[i+1]
    inds = np.where((d >= edge_lo) & (d <= edge_hi))[0]
    to_keep = np.concatenate([to_keep, jax.random.choice(
        rng, inds, [bin_max], replace=False)]).astype(int)
    return to_keep


def find_idxs_keep_jax(edge_lo, edge_hi, rng, d, bin_max):
    inds = jnp.where((d >= edge_lo) & (d <= edge_hi))[0]
    to_keep = jax.random.choice(
        rng, inds, [bin_max], replace=False).astype(int)
    return to_keep


def get_mask(idxs, n, sign_bool=True):
    mask = (np.ones(n) * int(not (sign_bool))).astype(bool)
    try:
        mask[idxs] = True if sign_bool else False
    except IndexError as e:
        raise IndexError(f'IndexError: {e} - idxs: {idxs}, n: {n}')
    return mask


def keep_idxs(df, to_keep):
    df = df.iloc[get_mask(to_keep, len(df))]
    return df


def rem_idxs(df, to_rem):
    df = df.iloc[get_mask(to_rem, len(df), sign_bool=0)]
    return df


# Balance


def transform_to_histogram_bins(df: pd.DataFrame, cols, num_bins: int = 30) -> pd.DataFrame:
    # Create a copy of the DataFrame to avoid modifying the original
    transformed_df = df.copy()

    # Compute quantile bin edges
    # Adding 0 and 1 to capture the full range of the data
    quantile_edges = np.concatenate([
        [df[cols].min()],  # Minimum value
        df[cols].quantile(np.linspace(0, 1, num_bins + 1)[1:-1]),
        [df[cols].max()]  # Maximum value
    ])

    # Ensure unique bin edges (some distributions might have repeated values)
    quantile_edges = np.unique(quantile_edges)

    # Create a mapping from original values to their quantile bin edge
    def map_to_quantile_edge(value):
        # Find the bin edge that this value falls into
        bin_index = np.digitize(value, quantile_edges) - 1
        return quantile_edges[bin_index]

    # Apply the transformation
    transformed_df[cols] = df[cols].apply(map_to_quantile_edge)

    return transformed_df


def balance_dataset(rng, df, cols, nbin, bin_max, use_log, func1=None):
    d = pre_balance(df, cols, use_log, func1)

    hist, bin_edges = np.histogram(d, bins=nbin)

    # i = np.where(hist > bin_max)[0][0]
    # ok = transform_to_histogram_bins(df, cols, 15)
    # to_keep = np.concatenate(jax.vmap(partial(find_idxs_keep_jax, rng=rng, d=np.array(d), bin_max=bin_max))(bin_edges[i], bin_edges[i+1]))
    to_keep = np.array([])
    for i in np.where(hist > bin_max)[0]:
        to_keep = find_idxs_keep(rng, bin_edges, i, d, bin_max, to_keep)
        # assert len(to_keep) == np.sum(np.where((hist[:i+1] - bin_max) > 0, hist[:i+1], 0)) - over_count*bin_max, 'something wrong'

    if to_keep.size > 0:
        df = keep_idxs(df, to_keep)
    else:
        print('No indices were removed.')
    return df


def balance_dataset2d(df, cols1, cols2, nbins, bin_max, use_log, func1, func2):
    d1 = pre_balance(df, cols1, use_log, func1)
    d2 = pre_balance(df, cols2, use_log, func2)
    hist, bin_edges_x, bin_edges_y = np.histogram2d(d1, d2, bins=nbins)

    to_keep = np.array([])
    for ix, iy in zip(*np.where(hist > bin_max)):
        to_keep = find_idxs_keep(bin_edges_x, ix, d1, bin_max, to_keep)
        to_keep = find_idxs_keep(bin_edges_y, iy, d2, bin_max, to_keep)

    to_keep = list(set(to_keep))
    df = keep_idxs(df, to_keep)
    return df
