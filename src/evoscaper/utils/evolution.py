

import jax.numpy as jnp


def calculate_ruggedness_from_perturbations(analytic_perturbed, analytic_og, eps):

    dp = (analytic_perturbed - analytic_og) / eps

    ruggedness = jnp.sqrt(jnp.nansum(jnp.square(dp), axis=0))

    return ruggedness
