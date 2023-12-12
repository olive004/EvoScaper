

from typing import Type
import haiku as hk
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import jax.numpy as jnp


class VAE(hk.Module):

    def __init__(self, encoder, decoder, embed_size: int, **hk_kwargs):
        # Inspired by Raptgen https://github.com/hmdlab/raptgen/blob/c4986ca9fa439b9389916c05829da4ff9c30d6f3/raptgen/models.py#L541

        super().__init__(**hk_kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.h2mu = hk.Linear(embed_size)
        self.h2logvar = hk.Linear(embed_size)

    def reparameterize(self, mu, logvar, key, deterministic=False):
        std = jnp.exp(0.5 * logvar)
        eps = jnp.normal(key, std.shape)
        z = mu + (std * eps if not deterministic else 0)
        return z

    def __call__(self,
                 input: Float[Array, " num_interactions"],
                 key: Float, 
                 inference: bool = False,
                 deterministic: bool = False,
                 logging: bool = True) -> Float[Array, " n_head"]:

        h = self.encoder(input)

        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        z = self.reparameterize(mu, logvar, key, deterministic)

        y = self.decoder(z)

        return y
