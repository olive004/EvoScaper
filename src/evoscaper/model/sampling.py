

from functools import partial
from typing import List, Optional, Tuple, Union
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
                           cond_min: Union[float, List[float]] = 0., cond_max: Union[float, List[float]] = 1.,
                           clip_range: Optional[Tuple[float]] = None):

    n_objectives = len(objective_cols)
    n_to_sample_per_cond = np.max(
        [1, n_to_sample//(n_categories ** n_objectives)])

    # Same for all objectives
    categories = make_categories(
        n_categories, n_objectives, cond_min, cond_max)
    sampled_cond = make_sampled_conditions(
        categories, n_categories, n_to_sample_per_cond, n_objectives, use_binned_sampling, use_onehot, objective_cols)

    z = jax.random.normal(
        rng, (n_categories ** n_objectives, n_to_sample_per_cond, hidden_size))
    z = np.concatenate([z, sampled_cond], axis=-1)

    # x_fake = jax.vmap(partial(decoder, params=params, rng=rng))(inputs=z)
    x_fake = decoder(inputs=z, params=params, rng=rng)
    x_fake = x_datanormaliser.create_chain_preprocessor_inverse(
        x_methods_preprocessing)(x_fake)

    if clip_range is not None:
        x_fake = jnp.clip(x_fake, clip_range[0], clip_range[1])

    return x_fake, z, sampled_cond


def make_sampled_conditions(categories: np.ndarray, n_categories: int, n_to_sample_per_cond: int, n_objectives: int,
                            use_binned_sampling: bool, use_onehot: bool, objective_cols: List[str]):
    """ Sample the same range of conditions for all objectives """

    # Same for all objectives
    sampled_cond = np.repeat(categories, repeats=n_to_sample_per_cond, axis=0).reshape(
        n_categories ** n_objectives, n_to_sample_per_cond, n_objectives)

    sampled_cond = transform_sampled_conditions(
        sampled_cond, n_categories, n_to_sample_per_cond, use_binned_sampling, use_onehot, objective_cols)
    return sampled_cond


def make_categories(n_categories: int, n_objectives: int, cond_min: Union[float, List[float]], cond_max: Union[float, List[float]]):
    """ Sample a different range of conditions for all objectives """

    if (type(cond_max) is not list) or (type(cond_max) is not np.ndarray):

        categories = np.array(list(itertools.product(
            *([np.linspace(cond_min, cond_max, n_categories).tolist()] * n_objectives))))
    else:
        assert len(cond_min) == len(cond_max) == n_objectives, \
            f"cond_min and cond_max must be lists of length {n_objectives}, got {len(cond_min)} and {len(cond_max)}"
        categories = np.array(list(itertools.product(
            *([np.linspace(c_min, c_max, n_categories).tolist() for c_min, c_max in zip(cond_min, cond_max)] * n_objectives))))

    return categories


def transform_sampled_conditions(sampled_cond: np.ndarray, n_categories: int,
                                 n_to_sample_per_cond: int, use_binned_sampling: bool,
                                 use_onehot: bool, objective_cols: List[str]):
    """ Transform sampled conditions into the desired format """
    if use_binned_sampling:
        if use_onehot:
            category_array = np.repeat(np.arange(n_categories)[
                :, None], repeats=n_to_sample_per_cond, axis=1)
            sampled_cond = np.ndarray(
                jax.nn.one_hot(category_array, n_categories))
            for k in objective_cols[1:]:
                sampled_cond2 = np.repeat(np.arange(n_categories)[
                    :, None], repeats=n_to_sample_per_cond, axis=1)
                sampled_cond2 = jax.nn.one_hot(sampled_cond2, n_categories)
                sampled_cond = np.concatenate(
                    [sampled_cond, sampled_cond2], axis=-1)

    return sampled_cond
