

import argparse
import logging
import os

from typing import Optional
from evoscaper.scripts.cvae_scan import cvae_scan_multi
from evoscaper.utils.preprocess import make_datetime_str
from evoscaper.utils.scan_prep import load_basics, load_varying, expand_df_varying
from synbio_morpher.utils.data.data_format_tools.common import write_json
from bioreaction.misc.misc import load_json_as_dict
import pandas as pd
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


def main(fn_basic, fn_varying, fn_df_hpos_loaded: Optional[str]):

    top_dir = os.path.join('notebooks', 'data',
                           'cvae_multi', make_datetime_str())
    os.makedirs(top_dir, exist_ok=True)

    if fn_df_hpos_loaded is None:
        hpos_basic = load_json_as_dict(fn_basic)
        varying = load_json_as_dict(fn_varying)
        basic_setting, df_hpos = load_basics(hpos_basic)
        hpos_to_vary_from_og = load_varying(varying['hpos_to_vary_from_og'])
        hpos_to_vary_together = load_varying(varying['hpos_to_vary_together'])
        df_hpos = expand_df_varying(
            df_hpos, basic_setting, hpos_to_vary_from_og, hpos_to_vary_together)
    else:
        df_hpos = pd.read_json(fn_df_hpos_loaded)

    logging.warning('!! Using subset of hyperparameters! !! df_hpos.iloc[...]')
    df_hpos_main = df_hpos.iloc[[6, 10, 11, 14, 24, 25, 27,
                                 34, 40, 51, 57, 60, 66, 70, 75, 91, 100, 101, 114, 117]]

    fn_config_multisim = os.path.join(top_dir, 'config_multisim.json')
    config_multisim = {
        'fn_varying': fn_varying,
        'fn_basic': fn_basic,
        'fn_df_hpos_loaded': fn_df_hpos_loaded,
        'signal_species': ('RNA_0',),
        'output_species': ('RNA_2',),
        'eval_n_to_sample': int(1e4),
        'eval_cond_min': -0.2,
        'eval_cond_max': 1.2,
        'eval_n_categories': 10,
        'eval_batch_size': int(1e5),
        'filename_simulation_settings': 'notebooks/configs/cvae_multi/simulation_settings.json',

    }

    max_n_categories_multi = 5

    for k in config_multisim.keys():
        if k.startswith('eval'):
            if k == 'eval_n_categories':
                # If there are multiple objectives, limit the number of categories (they grow quadratically)
                # [config_multisim[k]] * len(df_hpos) if df_hpos['objective_cols'].nunique() == 1 else max_n_categories_multi
                df_hpos.loc[:, k] = config_multisim[k]
                df_hpos.loc[df_hpos['objective_col'].apply(lambda x: 1 if type(
                    x) == str else len(x)) > 1, k] = max_n_categories_multi
            else:
                df_hpos.loc[:, k] = [(config_multisim[k])
                                     for _ in range(len(df_hpos))]

    write_json(config_multisim, fn_config_multisim)
    logging.info(
        f'\nRunning CVAE scan with {len(df_hpos_main)} models and {config_multisim["eval_n_to_sample"] * len(df_hpos_main)} total samples\n')
    cvae_scan_multi(df_hpos_main, fn_config_multisim,
                    top_dir, debug=False, visualise=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--fn_basic', type=str, default='notebooks/configs/cvae_multi/hpos_basic_20250413.json',
    # parser.add_argument('--fn_basic', type=str, default='notebooks/configs/cvae_multi/hpos_basic_20250424.json',
    parser.add_argument('--fn_basic', type=str, default='notebooks/configs/cvae_multi/hpos_basic_20250707.json',
                        help='Path to basic settings JSON file')
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_datasize.json',
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_objectives3.json',
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_objectives_test.json',
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_contloss4.json',
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_objectives4.json',
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_adaptation.json',
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_adaptation2.json',
    # parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_adaptation3.json',
    parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_adaptation4.json',
                        help='Path to varying settings JSON file')
    parser.add_argument('--fn_df_hpos_loaded', type=str, default=None,
                        help='Path to dataframe of hyperparameters and results from previous run (json).')
    args = parser.parse_args()
    fn_varying = args.fn_varying
    fn_basic = args.fn_basic
    fn_df_hpos_loaded = args.fn_df_hpos_loaded
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_03__21_33_13/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_06__16_27_57/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_24__17_11_20/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_27__16_33_11/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_28__12_57_27/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_29__14_07_43/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_31__16_06_36/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_01__17_00_19/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_01__21_06_36/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_03_31__16_18_57/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_02__11_24_31/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_02__11_18_07/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_02__14_36_19/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_10__13_58_26/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_11__15_04_39/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_11__18_52_29/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_13__12_04_45/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_10__11_34_20/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_24__20_51_02/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_25__09_25_04/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_26__19_01_53/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_04_28__16_15_32/df_hpos.json'
    # fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_07_07__16_27_26/df_hpos.json'
    fn_df_hpos_loaded = 'notebooks/data/cvae_multi/2025_09_03__11_20_07/df_hpos.json'

    main(fn_basic, fn_varying, fn_df_hpos_loaded)
