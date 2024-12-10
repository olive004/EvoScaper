

import numpy as np
from evoscaper.utils.normalise import make_chain_f, NormalizationSettings
from sklearn.utils import shuffle
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation


def init_data(data, OBJECTIVE_COL, OUTPUT_SPECIES, X_COLS,
              TOTAL_DS_MAX, BATCH_SIZE, SEED,
              PREP_X_NEG,
              PREP_X_LOGSCALE,
              PREP_X_STANDARDISE,
              PREP_X_MINMAX,
              PREP_X_ROBUST_SCALING,
              PREP_X_CATEGORICAL,
              PREP_X_CATEGORICAL_ONEHOT,
              PREP_X_CATEGORICAL_NBINS,
              PREP_X_CATEGORICAL_METHOD,
              PREP_Y_NEG,
              PREP_Y_LOGSCALE,
              PREP_Y_STANDARDISE,
              PREP_Y_MINMAX,
              PREP_Y_ROBUST_SCALING,
              PREP_Y_CATEGORICAL,
              PREP_Y_CATEGORICAL_ONEHOT,
              PREP_Y_CATEGORICAL_NBINS,
              PREP_Y_CATEGORICAL_METHOD
              ):

    df = prep_data(data, OUTPUT_SPECIES, OBJECTIVE_COL, X_COLS)

    TOTAL_DS = int(np.min([TOTAL_DS_MAX, len(df)]))
    TOTAL_DS = int(TOTAL_DS // BATCH_SIZE * BATCH_SIZE)
    N_BATCHES = int(TOTAL_DS // BATCH_SIZE)

    x_norm_settings = NormalizationSettings(
        negative=PREP_X_NEG,
        log=PREP_X_LOGSCALE,
        standardise=PREP_X_STANDARDISE,
        min_max=PREP_X_MINMAX,
        robust=PREP_X_ROBUST_SCALING,
        categorical=PREP_X_CATEGORICAL,
        categorical_onehot=PREP_X_CATEGORICAL_ONEHOT,
        categorical_n_bins=PREP_X_CATEGORICAL_NBINS,
        categorical_method=PREP_X_CATEGORICAL_METHOD
    )
    y_norm_settings = NormalizationSettings(
        negative=PREP_Y_NEG,
        log=PREP_Y_LOGSCALE,
        standardise=PREP_Y_STANDARDISE,
        min_max=PREP_Y_MINMAX,
        robust=PREP_Y_ROBUST_SCALING,
        categorical=PREP_Y_CATEGORICAL,
        categorical_onehot=PREP_Y_CATEGORICAL_ONEHOT,
        categorical_n_bins=PREP_Y_CATEGORICAL_NBINS,
        categorical_method=PREP_Y_CATEGORICAL_METHOD
    )
    x, cond, x_scaling, x_unscaling, y_scaling, y_unscaling = make_xy(df, SEED, TOTAL_DS, X_COLS, OBJECTIVE_COL,
                                                                      x_norm_settings, y_norm_settings)

    return df, x, cond, TOTAL_DS, N_BATCHES, x_scaling, x_unscaling, y_scaling, y_unscaling


def prep_data(data, OUTPUT_SPECIES, OBJECTIVE_COL, X_COLS):

    data = embellish_data(data)
    df = filter_invalids(data, OUTPUT_SPECIES, OBJECTIVE_COL)
    df = reduce_repeat_samples(df, X_COLS)
    return df


def embellish_data(data):
    if 'adaptability' not in data.columns:
        data['adaptability'] = calculate_adaptation(
            s=data['sensitivity_wrt_species-6'].values,
            p=data['precision_wrt_species-6'].values)
    data['Log sensitivity'] = np.log10(data['sensitivity_wrt_species-6'])
    return data


# Make xy
def make_xy(df, SEED, TOTAL_DS, X_COLS, OBJECTIVE_COL,
            x_norm_settings, y_norm_settings):

    x, x_datanormaliser, x_methods_preprocessing = make_x(
        df, TOTAL_DS, X_COLS, x_norm_settings)
    cond, y_datanormaliser, y_methods_preprocessing = make_y(
        df, OBJECTIVE_COL, TOTAL_DS, y_norm_settings)
    x, cond = shuffle(x, cond, random_state=SEED)

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

def filter_invalids(data, OUTPUT_SPECIES, OBJECTIVE_COL):

    filt = data['sample_name'].isin(OUTPUT_SPECIES) & ~data['precision_wrt_species-6'].isna() & ~data['sensitivity_wrt_species-6'].isna(
    ) & (np.abs(data['precision_wrt_species-6']) < np.inf) & data[OBJECTIVE_COL].notna() & (np.abs(data[OBJECTIVE_COL]) < np.inf) & (np.abs(data['sensitivity_wrt_species-6']) < np.inf)

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
