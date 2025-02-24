

import argparse
import pandas as pd
import itertools
import numpy as np
import os
from evoscaper.scripts.cvae_scan import cvae_scan_multi
from evoscaper.utils.preprocess import make_datetime_str
from evoscaper.utils.scan_prep import load_basics, load_varying, expand_df_varying
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


def main(fn_basic, fn_varying):

    top_dir = os.path.join('notebooks', 'data', 'cvae_multi', make_datetime_str())
    os.makedirs(top_dir, exist_ok=True)

    hpos_basic = load_json_as_dict(fn_basic)
    varying = load_json_as_dict(fn_varying)
    basic_setting, df_hpos = load_basics(hpos_basic)
    hpos_to_vary_from_og = load_varying(varying['hpos_to_vary_from_og'])
    hpos_to_vary_together = load_varying(varying['hpos_to_vary_together'])

    df_hpos = expand_df_varying(df_hpos, hpos_to_vary_from_og, hpos_to_vary_together)

    df_hpos_main = df_hpos #.iloc[61:111]

    fn_config_multisim = os.path.join(top_dir, 'config_multisim.json')
    config_multisim = {
        'fn_varying': fn_varying,
        'fn_basic': fn_basic,
        'signal_species': ('RNA_0',),
        'output_species': ('RNA_2',),
        'eval_n_to_sample': 1e3,
        'eval_cond_min': -0.2,
        'eval_cond_max': 1.2,
        'eval_n_categories': 2,
        'eval_batch_size': int(1e6),
        'filenames_train_config': 'data/raw/summarise_simulation/2024_12_05_210221/ensemble_config_update.json',
        # 'filenames_train_table': f'{data_dir}/raw/summarise_simulation/2024_12_05_210221/tabulated_mutation_info.csv',
        'filenames_train_table': 'notebooks/data/simulate_circuits/2025_02_01__00_22_38/tabulated_mutation_info.json'
    }
    write_json(config_multisim, fn_config_multisim)
    cvae_scan_multi(df_hpos_main, fn_config_multisim, top_dir, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_varying', type=str, default='notebooks/data/configs/cvae_multi/data_scan.json',
                        help='Path to basic and varying settings JSON file')
    args = parser.parse_args()
    fn_varying = args.fn_varying
    main(fn_basic, fn_varying)
