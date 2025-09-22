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
    # decoder = MLPWithActivation(output_sizes=[HIDDEN_SIZE] + dec_layers + [decoder_head],
    decoder = MLPWithActivation(output_sizes=dec_layers + [decoder_head],
                                w_init=get_initialiser(dec_init),
                                activation=activation,
                                activation_final=decoder_activation_final,
                                dropout_rate=dropout_rate,
                                name='decoder')
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
