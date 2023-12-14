

import haiku as hk
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import jax.numpy as jnp
from jax import random


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


def sample_z(mu, logvar, key, deterministic=False):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(key, std.shape)
    z = mu + (std * eps if not deterministic else 0)
    return z
