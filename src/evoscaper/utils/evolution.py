

import jax.numpy as jnp


def calculate_ruggedness_from_perturbations(analytic_perturbed, analytic_og, eps):
    """ Calculate the ruggedness using the perturbations of the analytic landscape.
    The combined metric is the L2 norm of the perturbation for each unique interaction.
    
    Args:
        analytic_perturbed: The perturbed analytic landscape.
        analytic_og: The original analytic landscape.
        eps: The perturbation size.
    """

    dp = (analytic_perturbed - analytic_og) / eps

    ruggedness = jnp.sqrt(jnp.nansum(jnp.square(dp), axis=0))

    return ruggedness
