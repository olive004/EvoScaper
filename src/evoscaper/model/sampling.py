

from functools import partial
from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import itertools

from evoscaper.utils.normalise import DataNormalizer


def sample_reconstructions(params, rng, decoder,
                           n_categories: int, n_to_sample: int, hidden_size: int,
                           x_datanormaliser: DataNormalizer, x_methods_preprocessing: List[str],
                           objective_cols: List[str],
                           use_binned_sampling: bool = True, use_onehot: bool = False,
                           cond_min: Optional[float] = 0, cond_max: Optional[float] = 1,
                           impose_final_range: Optional[Tuple[float]] = None):

    n_objectives = len(objective_cols)
    # n_to_sampel_per_cond = n_to_sample//(n_categories ** n_objectives)
    n_to_sampel_per_cond = n_to_sample

    if use_binned_sampling:
        if use_onehot:
            category_array = np.repeat(np.arange(n_categories)[
                :, None], repeats=n_to_sampel_per_cond, axis=1)
            category_array = jax.nn.one_hot(category_array, n_categories)
            sampled_cond = jax.nn.one_hot(category_array, n_categories)
            for k in objective_cols[1:]:
                sampled_cond2 = np.repeat(np.arange(n_categories)[
                    :, None], repeats=n_to_sampel_per_cond, axis=1)
                sampled_cond2 = jax.nn.one_hot(sampled_cond2, n_categories)
                sampled_cond = np.concatenate(
                    [sampled_cond, sampled_cond2], axis=-1)
        else:
            category_array = np.array(list(itertools.product(
                *([np.linspace(cond_min, cond_max, n_categories).tolist()] * n_objectives))))
            sampled_cond = np.repeat(category_array, repeats=n_to_sampel_per_cond, axis=1).reshape(
                n_categories ** n_objectives, n_to_sampel_per_cond, n_objectives)
    else:
        category_array = np.array(list(itertools.product(
            *([np.linspace(cond_min, cond_max, n_categories).tolist()] * n_objectives))))
        sampled_cond = np.repeat(category_array, repeats=n_to_sampel_per_cond, axis=1).reshape(
            n_categories ** n_objectives, n_to_sampel_per_cond, n_objectives)

        # category_array = np.linspace(cond_min, cond_max, n_categories)
        # sampled_cond = np.repeat(category_array[:, None], repeats=n_to_sample, axis=1)[:, :, None]

        # sampled_cond = jax.random.uniform(
        #     rng, (n_categories, n_to_sample, 1), minval=cond_min, maxval=cond_max)

    z = jax.random.normal(
        rng, (n_categories ** n_objectives, n_to_sampel_per_cond, hidden_size))
    z = np.concatenate([z, sampled_cond], axis=-1)

    # x_fake = jax.vmap(partial(decoder, params=params, rng=rng))(inputs=z)
    x_fake = decoder(inputs=z, params=params, rng=rng)
    x_fake = x_datanormaliser.create_chain_preprocessor_inverse(
        x_methods_preprocessing)(x_fake)

    if impose_final_range is not None:
        x_fake = jnp.clip(x_fake, impose_final_range[0], impose_final_range[1])

    return x_fake, z, sampled_cond
