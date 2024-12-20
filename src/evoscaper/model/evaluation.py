

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple


def estimate_mutual_information_knn(
    z: jnp.ndarray,  # Latent representations
    c: jnp.ndarray,  # Conditional variables
    k: int = 5  # Number of nearest neighbors
) -> float:
    """
    Estimate mutual information using K-Nearest Neighbors approach.

    Args:
        z: Latent representations (N x latent_dim)
        c: Conditional variables (N x condition_dim)
        k: Number of nearest neighbors for estimation

    Returns:
        Estimated mutual information
    """
    def pairwise_distances(x):
        """Compute pairwise distances between points."""
        x_sq = jnp.sum(x**2, axis=1)
        distances = x_sq[:, jnp.newaxis] + \
            x_sq[jnp.newaxis, :] - 2 * jnp.dot(x, x.T)
        return jnp.sqrt(jnp.maximum(distances, 0))

    def knn_distances(x):
        """Find k-th nearest neighbor distances."""
        dist_matrix = pairwise_distances(x)
        sorted_dists = jnp.sort(dist_matrix, axis=1)
        return sorted_dists[:, k]

    # Combine latent representations and conditions
    joint = jnp.concatenate([z, c], axis=-1)

    # Estimate entropy terms
    z_entropy = jnp.log(knn_distances(z) + 1e-10)
    c_entropy = jnp.log(knn_distances(c) + 1e-10)
    joint_entropy = jnp.log(knn_distances(joint) + 1e-10)

    # Mutual information estimation
    mi = jnp.mean(z_entropy + c_entropy - joint_entropy)

    return mi
