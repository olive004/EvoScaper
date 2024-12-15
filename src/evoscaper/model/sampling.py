

from functools import partial
from typing import List, Optional
import jax
import jax.numpy as jnp
import numpy as np

from evoscaper.utils.normalise import DataNormalizer


def sample_reconstructions(params, rng, decoder,
                           n_categories: Optional[int], n_to_sample: int, hidden_size: int,
                           x_datanormaliser: DataNormalizer, x_methods_preprocessing: List[str],
                           use_binned_sampling: bool = True, use_onehot: bool = False,
                           cond_min: Optional[float] = 0, cond_max: Optional[float] = 1):

    if use_binned_sampling:
        if use_onehot:
            sampled_cond = np.repeat(np.arange(n_categories)[
                :, None], repeats=n_to_sample//n_categories, axis=1)
            sampled_cond = jax.nn.one_hot(sampled_cond, n_categories)
        else:
            sampled_cond = np.repeat(np.linspace(cond_min, cond_max, n_categories)[
                :, None], repeats=n_to_sample//n_categories, axis=1)[:, :, None]
    else:
        sampled_cond = jax.random.uniform(
            rng, (n_categories, n_to_sample, 1), minval=cond_min, maxval=cond_max)

    z = jax.random.normal(rng, (n_to_sample, hidden_size))
    z = np.repeat(z[None, :], repeats=n_categories, axis=0)
    z = np.concatenate([z, sampled_cond], axis=-1)

    x_fake = jax.vmap(partial(decoder, params=params, rng=rng))(inputs=z)
    x_fake = x_datanormaliser.create_chain_preprocessor_inverse(
        x_methods_preprocessing)(x_fake)

    return x_fake, z, sampled_cond
