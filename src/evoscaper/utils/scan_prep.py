

from typing import List
import pandas as pd
import itertools
import numpy as np
import os
from evoscaper.scripts.cvae_scan import cvae_scan_multi
from evoscaper.utils.preprocess import make_datetime_str
from synbio_morpher.utils.data.data_format_tools.common import write_json
from bioreaction.misc.misc import flatten_listlike, load_json_as_dict

import jax

USE_ONLY_ONE_GPU = False

if USE_ONLY_ONE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0 or 1

# devices = jax.devices()
# if devices:
#     devices = [devices[0]]
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[0].id)

jax.config.update('jax_platform_name', 'gpu')

jax.devices()


def keep_equal(df):
    pairs = {
        'enc_ls': 'dec_ls',
        'num_enc_layers': 'num_dec_layers',
        'factor_expanding_ls': 'factor_contracting_ls',
    }
    for k1, k2 in pairs.items():
        if k1 in df.columns and k2 in df.columns:
            df[k2] = df[k1]
    return df


def add_combinatorial_keys(df_hpos, hpos_to_vary_together, basic_setting):
    keys_vary_together = sorted(hpos_to_vary_together.keys())
    for i, v in enumerate(itertools.product(*[hpos_to_vary_together[h] for h in keys_vary_together])):
        curr = basic_setting.assign(
            **{h: [vv] if type(vv) == tuple else vv for h, vv in zip(keys_vary_together, v)})
        df_hpos = pd.concat([df_hpos, curr], ignore_index=True)
    return df_hpos


def add_single_hpos(df_hpos, hpos_to_vary_from_og, basic_setting):
    for h, v in hpos_to_vary_from_og.items():
        try:
            df_hpos = pd.concat(
                [df_hpos] + [basic_setting.assign(**{h: vv}) for vv in v], ignore_index=True)
        except ValueError:
            for vv in v:
                b = basic_setting.copy()
                b.loc[0, h] = vv
                df_hpos = pd.concat([df_hpos, b], ignore_index=True)
    return df_hpos


def postproc(df_hpos):
    df_hpos = keep_equal(df_hpos)
    # df_hpos.loc[df_hpos['objective_col'] ==
    #             'sensitivity', 'prep_y_logscale'] = True
    df_hpos['print_every'] = df_hpos['epochs'] // 50
    
    df_hpos.loc[df_hpos['x_type'] ==
                'binding_rates_dissociation', 'prep_x_negative'] = False
    df_hpos = df_hpos.drop_duplicates().reset_index(drop=True)
    return df_hpos


def load_basics(hpos_all: dict):
    hpos_flat = {}
    for k, v in hpos_all.items():
        if isinstance(v, dict):
            hpos_flat.update(v)
        else:
            hpos_flat[k] = v
    for k, v in hpos_flat.items():
        if isinstance(v, list):
            hpos_flat[k] = tuple(v)
    df_hpos = pd.DataFrame.from_dict(hpos_flat, orient='index').T
    assert df_hpos.columns.duplicated().sum() == 0, 'Change some column names, there are duplicates'
    basic_setting = df_hpos.copy()
    return basic_setting, df_hpos

def load_varying(hpos_varying: List[dict]):
    """ Settings that are lists should actually be tuples. """
    for i, v in enumerate(hpos_varying):
        if isinstance(v, dict):
            for k, vv in v.items():
                if isinstance(vv, list):
                    for ii, vv in enumerate(vv):
                        if isinstance(vv, list):
                            hpos_varying[i][k][ii] = tuple(vv)
    return hpos_varying


def expand_df_varying(df_hpos, basic_setting, hpos_to_vary_from_og: dict, hpos_to_vary_together: dict):
    for h in hpos_to_vary_from_og:
        df_hpos = add_single_hpos(df_hpos, h, basic_setting)
    for h in hpos_to_vary_together:
        df_hpos = add_combinatorial_keys(df_hpos, h, basic_setting)

    df_hpos = postproc(df_hpos)

    # Reorder columns
    cols_priority = list(set(flatten_listlike([list(h.keys(
    )) for h in hpos_to_vary_from_og] + [list(h.keys()) for h in hpos_to_vary_together])))
    df_hpos = df_hpos[cols_priority +
                    [c for c in df_hpos.columns if c not in cols_priority]]

    df_hpos.reset_index(drop=True)
    return df_hpos
