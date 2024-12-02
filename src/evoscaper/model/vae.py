

import haiku as hk
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import jax.numpy as jnp
from jax import random
import logging


from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.parameter_prediction.simulator import RawSimulationHandling, make_piecewise_stepcontrol
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
from synbio_morpher.utils.common.setup import prepare_config, expand_config, expand_model_config
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols
from synbio_morpher.utils.misc.numerical import symmetrical_matrix_length
from synbio_morpher.utils.misc.type_handling import flatten_listlike, get_unique
from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded
from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.model.data_containers import BasicModel, QuantifiedReactions
from bioreaction.simulation.manager import simulate_steady_states
from functools import partial

from scipy.cluster.vq import whiten
from scipy.special import factorial
from sklearn.manifold import TSNE
import os
import sys
import numpy as np
import haiku as hk
import jax
import diffrax as dfx

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
                
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# if __package__ is None:

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

__package__ = os.path.basename(module_path)


from evoscaper.model.mlp import MLP
from evoscaper.model.shared import arrayise
from evoscaper.model.loss import loss_wrapper, compute_accuracy_regression, mse_loss
from evoscaper.utils.preprocess import drop_duplicates_keep_first_n
from evoscaper.utils.math import convert_to_scientific_exponent


class VAE(hk.Module):

    def __init__(self, encoder, decoder, embed_size: int, **hk_kwargs):
        # Inspired by Raptgen https://github.com/hmdlab/raptgen/blob/c4986ca9fa439b9389916c05829da4ff9c30d6f3/raptgen/models.py#L541

        super().__init__(**hk_kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.h2mu = hk.Linear(embed_size, name='h2mu')
        self.h2logvar = hk.Linear(embed_size, name='h2logvar')

    def reparameterize(self, mu, logvar, key, deterministic=False):
        # std = jnp.exp(0.5 * logvar)
        # eps = random.normal(key, std.shape)
        # z = mu + (std * eps if not deterministic else 0)
        return sample_z(mu, logvar, key, deterministic)

    def __call__(self,
                 input: Float[Array, " num_interactions"],
                 deterministic: bool = False,
                 logging: bool = True) -> Float[Array, " n_head"]:

        h = self.encoder(input)

        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        z = self.reparameterize(mu, logvar, hk.next_rng_key(), deterministic)

        y = self.decoder(z)

        return y


class CVAE(VAE):

    def __init__(self, encoder, decoder, embed_size: int, **hk_kwargs):
        super().__init__(encoder, decoder, embed_size, **hk_kwargs)

    def __call__(self, x: Array, cond: Array, deterministic: bool = False, logging: bool = True) -> Array:
        h = self.encoder(jnp.concatenate([x, cond], axis=-1))

        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        z = self.reparameterize(mu, logvar, hk.next_rng_key(), deterministic)
        z_cond = jnp.concatenate([z, cond], axis=-1)

        y = self.decoder(z_cond)
        return y
    
    
def VAE_fn(enc_layers: list, dec_layers: list, decoder_head, HIDDEN_SIZE, call_kwargs: dict = {}, ):
    encoder = MLP(layer_sizes=enc_layers, n_head=dec_layers[0], use_categorical=False, name='encoder')
    decoder = MLP(layer_sizes=dec_layers, n_head=decoder_head, use_categorical=False, name='decoder')
    model = CVAE(encoder=encoder, decoder=decoder, embed_size=HIDDEN_SIZE)
    
    def init(x: np.ndarray, cond: np.ndarray, deterministic: bool):
        h = model.encoder(np.concatenate([x, cond], axis=-1))

        mu = model.h2mu(h)
        logvar = model.h2logvar(h)
        z = model.reparameterize(mu, logvar, hk.next_rng_key(), deterministic)
        z_cond = np.concatenate([z, cond], axis=-1)

        y = model.decoder(z_cond)
        return y
        
    return init, (encoder, decoder, model, model.h2mu, model.h2logvar, model.reparameterize)


def sample_z(mu, logvar, key, deterministic=False):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(key, std.shape)
    z = mu + (std * eps if not deterministic else 0)
    return z


# def init_data(data, 
#               BATCH_SIZE, INPUT_SPECIES, MAX_TOTAL_DS, 
#               SCALE_X, SEED, TOTAL_DS, USE_CATEGORICAL, 
#               USE_X_LOGSCALE, X_TYPE, 
#               input_concat_axis, input_concat_diffs, target_circ_func):
#     filt = data['sample_name'] == INPUT_SPECIES

#     # Balance the dataset
#     df = data[filt]
#     df = drop_duplicates_keep_first_n(df, get_true_interaction_cols(
#         data, X_TYPE, remove_symmetrical=True), n=100)

#     TOTAL_DS = np.min([TOTAL_DS, MAX_TOTAL_DS, len(df)])
#     N_BATCHES = TOTAL_DS // BATCH_SIZE
#     TOTAL_DS = N_BATCHES * BATCH_SIZE

#     x_cols = [get_true_interaction_cols(data, X_TYPE, remove_symmetrical=True)]
#     if input_concat_diffs:
#         x_cols = x_cols + \
#             [[f'{i}_diffs' for i in get_true_interaction_cols(
#                 data, X_TYPE, remove_symmetrical=True)]]

#     x = [df[i].iloc[:TOTAL_DS].values[:, :, None] for i in x_cols]
#     x = np.concatenate(x, axis=input_concat_axis+1).squeeze()

#     x_scaling, x_unscaling = [], []
#     if USE_X_LOGSCALE:
#         x_scaling.append(np.log10)
#         x_unscaling.append(lambda x: np.power(10, x))

#     if SCALE_X:
#         xscaler = MinMaxScaler()
#         x_scaling.append(xscaler.fit_transform)
#         x_unscaling.append(xscaler.inverse_transform)

#     x_unscaling = x_unscaling[::-1]

#     for fn in x_scaling:
#         x = fn(x)
        
#     cond = df[target_circ_func].iloc[:TOTAL_DS].to_numpy()

#     if USE_CATEGORICAL:

#         vectorized_convert_to_scientific_exponent = np.vectorize(
#             convert_to_scientific_exponent)
#         numerical_resolution = 2
#         cond_map = {k: numerical_resolution for k in np.arange(int(f'{cond[cond != 0].min():.0e}'.split(
#             'e')[1])-1, np.max([int(f'{cond.max():.0e}'.split('e')[1])+1, 0 + 1]))}
#         cond_map[-6] = 1
#         cond_map[-5] = 1
#         cond_map[-4] = 4
#         cond_map[-3] = 2
#         cond_map[-1] = 3
#         cond = jax.tree_util.tree_map(partial(
#             vectorized_convert_to_scientific_exponent, numerical_resolution=cond_map), cond)
#         cond = np.interp(cond, sorted(np.unique(cond)), np.arange(
#             len(sorted(np.unique(cond))))).astype(int)
#     else:
#         zero_log_replacement = -10.0
#         cond = np.where(cond != 0, np.log10(cond), zero_log_replacement)

#     cond = cond[:, None]
#     N_HEAD = x.shape[-1]

#     x, cond = shuffle(x, cond, random_state=SEED)

#     if x.shape[0] < TOTAL_DS:
#         logging.warning(
#             f'WARNING: The filtered data is not as large as the requested total dataset size: {x.shape[0]} vs. requested {TOTAL_DS}')
        
#     return x, cond, x_scaling, x_unscaling, x_cols, df, filt, N_HEAD