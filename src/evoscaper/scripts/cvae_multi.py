

import argparse
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

# jupyter nbconvert --to notebook --execute 03_cvae_multi.ipynb --output=03_cvae_multi_2.ipynb --ExecutePreprocessor.timeout=-1

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
    #             'sensitivity_wrt_species-6', 'prep_y_logscale'] = True
    df_hpos['print_every'] = df_hpos['epochs'] // 50
    
    df_hpos.loc[df_hpos['x_type'] ==
                'binding_rates_dissociation', 'prep_x_negative'] = False
    df_hpos = df_hpos.drop_duplicates().reset_index(drop=True)
    return df_hpos


def load_basics(hpos_all: dict):
    for k, v in hpos_all.items():
        if isinstance(v, list):
            hpos_all[k] = tuple(v)
    df_hpos = pd.DataFrame.from_dict(hpos_all, orient='index').T
    assert df_hpos.columns.duplicated().sum() == 0, 'Change some column names, there are duplicates'
    basic_setting = df_hpos.copy()
    return basic_setting, df_hpos

def load_varying(d: dict):
    for k, v in d.items():
        if isinstance(v, list):
            for i, vv in enumerate(v):
                if isinstance(vv, list):
                    d[k][i] = tuple(vv)
    return d


def expand_df_varying(df_hpos, hpos_to_vary_from_og: dict, hpos_to_vary_together: dict):
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


def main(fn_config):

    top_dir = os.path.join('notebooks', 'data', 'cvae_multi', make_datetime_str())
    os.makedirs(top_dir, exist_ok=True)

    settings = load_json_as_dict(fn_config)
    basic_setting, df_hpos = load_basics(settings['hpos_basic'])
    hpos_to_vary_from_og = load_varying(settings['hpos_to_vary_from_og'])
    hpos_to_vary_together = load_varying(settings['hpos_to_vary_together'])

    df_hpos = expand_df_varying(df_hpos, hpos_to_vary_from_og, hpos_to_vary_together)

    df_hpos_main = df_hpos #.iloc[61:111]

    fn_config_multisim = os.path.join(top_dir, 'config_multisim.json')
    config_multisim = {
        'signal_species': ('RNA_0',),
        'output_species': ('RNA_2',),
        'eval_n_to_sample': 1e3,
        'eval_cond_min': -0.2,
        'eval_cond_max': 1.2,
        'eval_n_categories': 2,
        'filenames_train_config': 'data/raw/summarise_simulation/2024_12_05_210221/ensemble_config_update.json',
        # 'filenames_train_table': f'{data_dir}/raw/summarise_simulation/2024_12_05_210221/tabulated_mutation_info.csv',
        'filenames_train_table': 'notebooks/data/simulate_circuits/2025_02_01__00_22_38/tabulated_mutation_info.json'
    }
    write_json(config_multisim, fn_config_multisim)
    cvae_scan_multi(df_hpos_main, fn_config_multisim, top_dir, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_config', type=str, default='notebooks/data/configs/cvae_multi/data_scan.json',
                        help='Path to basic and varyingsettings JSON file')
    args = parser.parse_args()
    fn_config = args.fn_config
    main(fn_config)
