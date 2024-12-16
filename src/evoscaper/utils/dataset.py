

import numpy as np
import jax
from evoscaper.utils.normalise import make_chain_f
from evoscaper.utils.dataclasses import NormalizationSettings, FilterSettings
from sklearn.utils import shuffle
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation


def init_data(data, x_cols: list, y_col: str, OUTPUT_SPECIES: list,
              TOTAL_DS_MAX, BATCH_SIZE, rng,
              x_norm_settings: NormalizationSettings,
              y_norm_settings: NormalizationSettings,
              filter_settings: FilterSettings
              ):

    df = prep_data(data, OUTPUT_SPECIES, y_col, x_cols, filter_settings)

    TOTAL_DS = int(np.min([TOTAL_DS_MAX, len(df)]))
    if TOTAL_DS < BATCH_SIZE:
        print(f'TOTAL_DS is less than BATCH_SIZE: {TOTAL_DS} < {BATCH_SIZE}')
        BATCH_SIZE = TOTAL_DS
    else:
        TOTAL_DS = int(TOTAL_DS // BATCH_SIZE * BATCH_SIZE)
    if TOTAL_DS == 0:
        raise ValueError('TOTAL_DS is 0')
    N_BATCHES = int(TOTAL_DS // BATCH_SIZE)

    x, cond, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing = make_xy(df, rng, TOTAL_DS, x_cols, y_col,
                                                                                                            x_norm_settings, y_norm_settings)

    return df, x, cond, TOTAL_DS, N_BATCHES, BATCH_SIZE, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing


def prep_data(data, OUTPUT_SPECIES, OBJECTIVE_COL, X_COLS, filter_settings):

    data = embellish_data(data)
    df = filter_invalids(data, OUTPUT_SPECIES, X_COLS,
                         OBJECTIVE_COL, filter_settings)
    df = reduce_repeat_samples(df, X_COLS)
    return df


def embellish_data(data, transform_sensitivity_nans=True, zero_log_replacement=-10.0):
    if 'adaptability' not in data.columns:
        data['adaptability'] = calculate_adaptation(
            s=data['sensitivity_wrt_species-6'].values,
            p=data['precision_wrt_species-6'].values)
    if transform_sensitivity_nans:
        data['sensitivity_wrt_species-6'] = np.where(np.isnan(
            data['sensitivity_wrt_species-6']), 0, data['sensitivity_wrt_species-6'])
    data['Log sensitivity'] = np.where(data['sensitivity_wrt_species-6'] == 0, zero_log_replacement, np.log10(data['sensitivity_wrt_species-6']))
    return data


# Make xy
def make_xy(df, rng, TOTAL_DS, X_COLS, OBJECTIVE_COL,
            x_norm_settings, y_norm_settings):

    x, x_datanormaliser, x_methods_preprocessing = make_x(
        df, TOTAL_DS, X_COLS, x_norm_settings)
    cond, y_datanormaliser, y_methods_preprocessing = make_y(
        df, OBJECTIVE_COL, TOTAL_DS, y_norm_settings)
    # x, cond = shuffle(x, cond, random_state=rng)
    shuffled_indices = jax.random.permutation(rng, x.shape[0])
    x, cond = x[shuffled_indices], cond[shuffled_indices]

    if x.shape[0] < TOTAL_DS:
        print(
            f'WARNING: The filtered data is not as large as the requested total dataset size: {x.shape[0]} vs. requested {TOTAL_DS}')

    return x, cond, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing


def make_x(df, TOTAL_DS, X_COLS, x_norm_settings):
    x = [df[i].iloc[:TOTAL_DS].values[:, None] for i in X_COLS]
    x = np.concatenate(x, axis=1).squeeze()

    x_datanormaliser, x_methods_preprocessing = make_chain_f(x_norm_settings)

    x = x_datanormaliser.create_chain_preprocessor(x_methods_preprocessing)(x)
    return x, x_datanormaliser, x_methods_preprocessing


def make_y(df, OBJECTIVE_COL, TOTAL_DS, y_norm_settings):

    cond = df[OBJECTIVE_COL].iloc[:TOTAL_DS].to_numpy()[:, None]
    y_datanormaliser, y_methods_preprocessing = make_chain_f(y_norm_settings)
    cond = y_datanormaliser.create_chain_preprocessor(
        y_methods_preprocessing)(cond)
    return cond, y_datanormaliser, y_methods_preprocessing


# Balance preprocess

def filter_invalids(data, OUTPUT_SPECIES, X_COLS, OBJECTIVE_COL, filter_settings: FilterSettings):

    filt = data['sample_name'].isin(OUTPUT_SPECIES)
    if filter_settings.filt_x_nans:
        filt = filt & data[X_COLS].notna().all(axis=1)
    if filter_settings.filt_y_nans:
        filt = filt & data[OBJECTIVE_COL].notna() & (
            np.abs(data[OBJECTIVE_COL]) < np.inf)
    if filter_settings.filt_sensitivity_nans:
        filt = filt & (np.abs(data['sensitivity_wrt_species-6'])
                       < np.inf) & data['sensitivity_wrt_species-6'].notna()
    if filter_settings.filt_precision_nans:
        filt = filt & (np.abs(data['precision_wrt_species-6'])
                       < np.inf) & data['precision_wrt_species-6'].notna()

    df = data[filt]
    # df.loc[:, OBJECTIVE_COL] = df[OBJECTIVE_COL].apply(
    #     lambda x: np.round(x, 1))
    df = df.reset_index(drop=True)

    return df


def reduce_repeat_samples(df, X_COLS):
    df = df.reset_index(drop=True)

    n_same_circ_max = 100
    nbin = 300
    def agg_func(x): return np.sum(x, axis=1)
    def agg_func(x): return tuple(x)

    df.loc[:, X_COLS] = df[X_COLS].apply(lambda x: np.round(x, 1))
    df_bal = balance_dataset(df, cols=X_COLS, nbin=nbin,
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


def find_idxs_keep(bin_edges, i, d, bin_max, to_keep):
    edge_lo, edge_hi = bin_edges[i], bin_edges[i+1]
    inds = np.where((d >= edge_lo) & (d <= edge_hi))[0]
    to_keep = np.concatenate([to_keep, np.random.choice(
        inds, bin_max, replace=False)]).astype(int)
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

def balance_dataset(df, cols, nbin, bin_max, use_log, func1=None):
    d = pre_balance(df, cols, use_log, func1)

    hist, bin_edges = np.histogram(d, bins=nbin)

    to_keep = np.array([])
    for i in np.where(hist > bin_max)[0]:
        to_keep = find_idxs_keep(bin_edges, i, d, bin_max, to_keep)
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
