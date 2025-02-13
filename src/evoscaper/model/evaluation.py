

import jax.numpy as jnp


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
    eps = 1e-10
    z_entropy = jnp.log(knn_distances(z) + eps)
    c_entropy = jnp.log(knn_distances(c) + eps)
    joint_entropy = jnp.log(knn_distances(joint) + eps)

    # Mutual information estimation
    mi = jnp.mean(z_entropy + c_entropy - joint_entropy)

    return mi


def calc_prompt_adherence(pred, real, perc_recall):
    """ Arrays should be of shape (n_prompts, n_samples, n_objectives) """

    diff = jnp.abs(pred - real)
    thresh_recall = jnp.expand_dims(jnp.max(diff, axis=1) * perc_recall, axis=1)
    # n_prompts_uniq = pred.shape[0]

    n_positives_inclass = jnp.nansum(diff < thresh_recall, axis=1)
    n_positives_all = jnp.nansum(diff < thresh_recall)
    n_positive_preds = pred.shape[1]

    precision = n_positives_inclass / n_positive_preds
    recall = n_positives_inclass / n_positives_all # * n_prompts_uniq
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, jnp.where(jnp.isnan(f1), 0, f1)
