

import jax.numpy as jnp
import numpy as np
import jax
from functools import partial


def calculate_ruggedness_from_perturbations(analytic_perturbed, analytic_og, eps):
    """ Calculate the ruggedness using the perturbations of the analytic landscape.
    The combined metric is the L2 norm of the perturbation for each unique interaction.

    Args:
        analytic_perturbed: The perturbed analytic landscape, shape [n_samples, n_perturbs, n_species].
        analytic_og: The original analytic landscape, [n_samples, 1, n_species].
        eps: The perturbation size.
    """

    dp = (analytic_perturbed - analytic_og) / eps

    ruggedness = jnp.sqrt(jnp.nansum(jnp.square(dp), axis=0))

    return ruggedness


def calculate_ruggedness_from_perturbations_alt(analytic_perturbed, analytic_og, eps):
    """ Calculate the ruggedness using the perturbations of the analytic landscape.

    Args:
        analytic_perturbed: The perturbed analytic landscape, shape [n_samples, n_perturbs, n_species].
        analytic_og: The original analytic landscape, [n_samples, 1, n_species].
        eps: The perturbation size.
    """

    ratios = jnp.where(analytic_og == 0, 1 + (analytic_perturbed - analytic_og),
                       analytic_perturbed / analytic_og)

    ruggedness = jnp.sum(jnp.log10(ratios), axis=0)

    return ruggedness


def calculate_ruggedness_core(analytics_perturbed, analytics_original, analytic,
                              resimulate_analytics, n_samples, n_perturbs, eps,
                              use_alt_algo=False):

    analytic_perturbed = jnp.array(
        analytics_perturbed[analytic]).reshape(n_samples, n_perturbs, -1)
    if resimulate_analytics:
        analytic_perturbed = analytic_perturbed[:, :-1, :]
        analytic_og = analytic_perturbed[:, -1, :]
    elif analytics_original is not None:
        analytic_og = np.array(analytics_original[analytic][:n_samples])
    else:
        analytic_og = np.zeros_like(analytic_perturbed[:, -1, :])

    # If loaded from previous data where not all analytics were saved
    if analytic_perturbed.shape[-1] != analytic_og.shape[-1]:
        analytic_perturbed = analytic_perturbed[..., -analytic_og.shape[-1]:]

    f = calculate_ruggedness_from_perturbations_alt if use_alt_algo else calculate_ruggedness_from_perturbations
    ruggedness = jax.vmap(partial(f, eps=eps))(
        analytic_perturbed, analytic_og[:, None, :])

    return ruggedness
