

import argparse
import logging
import os
from evoscaper.scripts.cvae_scan import cvae_scan_multi
from evoscaper.utils.preprocess import make_datetime_str
from evoscaper.utils.scan_prep import load_basics, load_varying, expand_df_varying
from synbio_morpher.utils.data.data_format_tools.common import write_json
from bioreaction.misc.misc import load_json_as_dict

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

    df_hpos = expand_df_varying(df_hpos, basic_setting, hpos_to_vary_from_og, hpos_to_vary_together)

    df_hpos_main = df_hpos # .iloc[:2]

    fn_config_multisim = os.path.join(top_dir, 'config_multisim.json')
    config_multisim = {
        'fn_varying': fn_varying,
        'fn_basic': fn_basic,
        'signal_species': ('RNA_0',),
        'output_species': ('RNA_2',),
        'eval_n_to_sample': int(1e4),
        'eval_cond_min': -0.2,
        'eval_cond_max': 1.2,
        'eval_n_categories': 10,
        'eval_batch_size': int(1e6),
        'filename_simulation_settings': 'notebooks/configs/cvae_multi/simulation_settings.json',
    }
    max_n_categories_multi = 5
    
    for k in config_multisim.keys():
        if k.startswith('eval'):
            if k == 'eval_n_categories':
                df_hpos.loc[:, k] = config_multisim[k] # [config_multisim[k]] * len(df_hpos) if df_hpos['objective_cols'].nunique() == 1 else max_n_categories_multi
                df_hpos.loc[df_hpos['objective_col'].apply(lambda x: 1 if type(x) == str else len(x)) > 1, k] = max_n_categories_multi
            else:
                df_hpos.loc[:, k] = [(config_multisim[k]) for _ in range(len(df_hpos))]
        
    write_json(config_multisim, fn_config_multisim)
    logging.info(f'\nRunning CVAE scan with {len(df_hpos_main)} models and {config_multisim["eval_n_to_sample"] * len(df_hpos_main)} total samples\n')
    cvae_scan_multi(df_hpos_main, fn_config_multisim, top_dir, debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_basic', type=str, default='notebooks/configs/cvae_multi/hpos_basic.json',
                        help='Path to basic settings JSON file')
    parser.add_argument('--fn_varying', type=str, default='notebooks/configs/cvae_multi/scan_datasize.json',
                        help='Path to varying settings JSON file')
    args = parser.parse_args()
    fn_varying = args.fn_varying
    fn_basic = args.fn_basic
    main(fn_basic, fn_varying)
