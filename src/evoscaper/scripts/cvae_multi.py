

import pandas as pd
import itertools
import numpy as np
import os
from evoscaper.scripts.cvae_scan import cvae_scan_single
from evoscaper.utils.preprocess import make_datetime_str
from synbio_morpher.utils.data.data_format_tools.common import write_json
from bioreaction.misc.misc import flatten_listlike

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


def load_basics(fn):
    df_hpos = pd.read_csv(fn)
    df_hpos = pd.DataFrame.from_dict(hpos_all, orient='index').T
    assert df_hpos.columns.duplicated().sum() == 0, 'Change some column names, there are duplicates'
    basic_setting = df_hpos.copy()


def load_varying(fn):
    df_vary = pd.read_csv(fn)
    hpos_to_vary_from_og = [{
        'hidden_size': np.arange(2, 32, 2),
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4]
    }]
    hpos_to_vary_together = [{
        'objective_col': [('adaptation',), ('Log sensitivity',), ('Log sensitivity', 'Log precision')],
        'prep_y_categorical': [False, True],
        'use_kl_div': [True],
        'kl_weight': [5e-5, 1e-4, 2.5e-4, 4e-4, 5e-4],
        'threshold_early_val_acc': [0.995, 0.98, 0.96, 0.9],
    },
        {
        'use_contrastive_loss': [True],
        'temperature': [0.1, 0.5, 1, 1.5, 2, 4, 8],
        'threshold_similarity': [0.95, 0.9, 0.7, 0.5, 0.3, 0.1],
        'power_factor_distance': [3, 4],
        'threshold_early_val_acc': [0.995, 0.9]
    }
    ]

    df_hpos.loc[df_hpos['objective_col'] ==
                'sensitivity_wrt_species-6', 'prep_y_logscale'] = True

def main():
    pass


if __name__ == "__main__":
    main()
