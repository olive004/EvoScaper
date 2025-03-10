from evoscaper.model.mlp import MLPWithActivation
from evoscaper.model.shared import get_initialiser
from bioreaction.misc.misc import flatten_listlike
import haiku as hk
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import jax.numpy as jnp
from typing import Callable, List
import os
import sys
import numpy as np
import haiku as hk
import jax


# if __package__ is None:

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

__package__ = os.path.basename(module_path)


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

    def __call__(self, x: Array, cond: Array, deterministic: bool = False,
                 inference: bool = False,
                 return_all: bool = False,
                 logging: bool = True) -> Array:
        h = self.encoder(jnp.concatenate(
            [x, cond], axis=-1), inference=inference)

        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        z = self.reparameterize(mu, logvar, hk.next_rng_key(), deterministic)
        z_cond = jnp.concatenate([z, cond], axis=-1)

        y = self.decoder(z_cond, inference=inference)

        if return_all:
            return y, mu, logvar, h
        return y


def VAE_fn(enc_layers: List[int], dec_layers: List[int], decoder_head: int, HIDDEN_SIZE: int, decoder_activation_final: Callable, USE_CATEGORICAL=False, call_kwargs: dict = {},
           enc_init='HeNormal', dec_init='HeNormal', activation: Callable = jax.nn.leaky_relu, dropout_rate=None):
    encoder = MLPWithActivation(output_sizes=enc_layers + [HIDDEN_SIZE],
                                w_init=get_initialiser(enc_init),
                                activation=activation,
                                activation_final=jax.nn.log_softmax if USE_CATEGORICAL else jax.nn.leaky_relu,
                                dropout_rate=dropout_rate,
                                name='encoder')
    decoder = MLPWithActivation(output_sizes=[HIDDEN_SIZE] + dec_layers + [decoder_head],
                                w_init=get_initialiser(dec_init),
                                activation=activation,
                                activation_final=decoder_activation_final,
                                dropout_rate=dropout_rate,
                                name='decoder')
    # encoder = hk.Sequential(flatten_listlike([[hk.Linear(i), jax.nn.leaky_relu] for i in enc_layers + [HIDDEN_SIZE]]))
    # decoder = hk.Sequential(flatten_listlike([[hk.Linear(i), jax.nn.leaky_relu] for i in [HIDDEN_SIZE] + dec_layers]) + [hk.Linear(decoder_head), jax.nn.sigmoid if USE_SIGMOID_DECODER else jax.nn.leaky_relu])
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
    """ We use exp(0.5*logvar) instead of std because it is more numerically stable
    and add the 0.5 part because std^2 = exp(logvar) """
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, mu.shape) if not deterministic else 0
    z = mu + std * eps
    return z


def sample_z(mu, logvar, key, deterministic=False):
    """ We use exp(0.5*logvar) instead of std because it is more numerically stable
    and add the 0.5 part because std^2 = exp(logvar) """
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.uniform(key, mu.shape) if not deterministic else 0
    # eps = jax.random.normal(key, mu.shape) if not deterministic else 0
    z = mu + std * eps
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
