

import numpy as np
from evoscaper.utils.math import convert_to_scientific_exponent
import jax
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation


def init_data(data, OUTPUT_SPECIES, X_COLS, TOTAL_DS, TOTAL_DS_MAX, BATCH_SIZE, SEED,
              USE_X_NEG, USE_X_LOGSCALE, SCALE_X_MINMAX,
              OBJECTIVE_COL, USE_Y_LOGSCALE, SCALE_Y_MINMAX, USE_Y_CATEGORICAL):

    data = embellish_data(data)
    df = filter_invalids(data, OUTPUT_SPECIES, OBJECTIVE_COL)
    df = reduce_repeat_samples(df, X_COLS)

    if TOTAL_DS is None:
        TOTAL_DS = len(df)

    TOTAL_DS = int(np.min([TOTAL_DS, TOTAL_DS_MAX, len(df)]))
    TOTAL_DS = int(TOTAL_DS // BATCH_SIZE * BATCH_SIZE)
    N_BATCHES = int(TOTAL_DS // BATCH_SIZE)

    x, cond, x_scaling, x_unscaling, y_scaling, y_unscaling = make_xy(df, SEED, TOTAL_DS, X_COLS, USE_X_NEG,
                                                                      USE_X_LOGSCALE, SCALE_X_MINMAX, OBJECTIVE_COL,
                                                                      USE_Y_LOGSCALE, SCALE_Y_MINMAX, USE_Y_CATEGORICAL)
    N_HEAD = x.shape[-1]

    return df, x, cond, TOTAL_DS, N_BATCHES, x_scaling, x_unscaling, y_scaling, y_unscaling


def embellish_data(data):
    if 'adaptability' not in data.columns:
        data['adaptability'] = calculate_adaptation(
            s=data['sensitivity_wrt_species-6'].values,
            p=data['precision_wrt_species-6'].values)
    return data


# Make xy

def make_xy(df, SEED, TOTAL_DS, X_COLS, USE_X_NEG, USE_X_LOGSCALE, SCALE_X_MINMAX,
            OBJECTIVE_COL, USE_Y_LOGSCALE, SCALE_Y_MINMAX, USE_Y_CATEGORICAL):

    x, x_scaling, x_unscaling = make_x(df, TOTAL_DS, X_COLS, USE_X_NEG,
                                       USE_X_LOGSCALE, SCALE_X_MINMAX)
    cond, y_scaling, y_unscaling = make_y(df, OBJECTIVE_COL, USE_Y_LOGSCALE,
                                          SCALE_Y_MINMAX, USE_Y_CATEGORICAL, TOTAL_DS)

    x, cond = shuffle(x, cond, random_state=SEED)

    if x.shape[0] < TOTAL_DS:
        print(
            f'WARNING: The filtered data is not as large as the requested total dataset size: {x.shape[0]} vs. requested {TOTAL_DS}')

    return x, cond, x_scaling, x_unscaling, y_scaling, y_unscaling


def make_x(df, TOTAL_DS, X_COLS, USE_X_NEG, USE_X_LOGSCALE, SCALE_X_MINMAX):
    x = [df[i].iloc[:TOTAL_DS].values[:, None] for i in X_COLS]
    x = np.concatenate(x, axis=1).squeeze()

    x_scaling, x_unscaling = [], []
    if USE_X_NEG:
        x_scaling.append(lambda x: -x)
        x_unscaling.append(lambda x: -x)
        x = x_scaling[-1](x)

    if USE_X_LOGSCALE:
        x_scaling.append(np.log10)
        x_unscaling.append(lambda x: np.power(10, x))
        x = x_scaling[-1](x)

    if SCALE_X_MINMAX:
        xscaler = MinMaxScaler().fit(x)
        x_scaling.append(xscaler.transform)
        x_unscaling.append(xscaler.inverse_transform)
        x = x_scaling[-1](x)

    x_unscaling = x_unscaling[::-1]

    return x, x_scaling, x_unscaling


def make_y(df, OBJECTIVE_COL, USE_Y_LOGSCALE, SCALE_Y_MINMAX, USE_Y_CATEGORICAL, TOTAL_DS):

    cond = df[OBJECTIVE_COL].iloc[:TOTAL_DS].to_numpy()[:, None]

    y_scaling, y_unscaling = [], []

    if USE_Y_CATEGORICAL:

        vectorized_convert_to_scientific_exponent = np.vectorize(
            convert_to_scientific_exponent)
        numerical_resolution = 2
        cond_map = {k: numerical_resolution for k in np.arange(int(f'{cond[cond != 0].min():.0e}'.split(
            'e')[1])-1, np.max([int(f'{cond.max():.0e}'.split('e')[1])+1, 0 + 1]))}
        cond_map[-6] = 1
        cond_map[-5] = 1
        cond_map[-4] = 4
        cond_map[-3] = 2
        cond_map[-1] = 3
        cond = jax.tree_util.tree_map(partial(
            vectorized_convert_to_scientific_exponent, numerical_resolution=cond_map), cond)
        cond = np.interp(cond, sorted(np.unique(cond)), np.arange(
            len(sorted(np.unique(cond))))).astype(int)
        cond = y_scaling[-1](cond)

    if USE_Y_LOGSCALE:
        zero_log_replacement = -10.0
        cond = np.where(cond != 0, np.log10(cond), zero_log_replacement)
        y_scaling.append(lambda x: np.where(
            x != 0, np.log10(x), zero_log_replacement))
        y_unscaling.append(lambda x: np.where(
            x != zero_log_replacement, np.power(10, x), 0))
        cond = y_scaling[-1](cond)

    if SCALE_Y_MINMAX:
        yscaler = MinMaxScaler().fit(cond)
        y_scaling.append(yscaler.transform)
        y_unscaling.append(yscaler.inverse_transform)
        cond = y_scaling[-1](cond)

    y_unscaling = y_unscaling[::-1]

    return cond, y_scaling, y_unscaling


# Balance preprocess

def filter_invalids(data, OUTPUT_SPECIES, OBJECTIVE_COL):

    filt = data['sample_name'].isin(OUTPUT_SPECIES) & ~data['precision_wrt_species-6'].isna(
    ) & ~data['sensitivity_wrt_species-6'].isna() & (data['precision_wrt_species-6'] < np.inf) & data[OBJECTIVE_COL].notna()

    df = data[filt]
    df.loc[:, 'adaptability'] = df['adaptability'].apply(np.float32)
    df.loc[:, 'adaptability'] = df['adaptability'].apply(
        lambda x: np.round(x, 1))
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
