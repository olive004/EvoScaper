

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
import jax.numpy as jnp
import numpy as np


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
    n_prompts_uniq = pred.shape[0]

    n_positives_inclass = jnp.nansum(diff < thresh_recall, axis=1)
    n_positives_all = jnp.nansum(diff < thresh_recall)
    n_positive_preds = pred.shape[1]

    precision = n_positives_inclass / n_positive_preds
    recall = n_positives_inclass / n_positives_all #* (n_prompts_uniq * perc_recall)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, jnp.where(jnp.isnan(f1), 0, f1)


def calculate_kl_divergence(feature, sampled_cond_binned, nbins=30):
    """
    Calculate KL divergence between histograms of feature values for different conditions.
    Feature could be a single dimension of the latent space or a single dimension of the data,
    for example one of the unique interactions in a circuit.
    """
    unique_conds = np.unique(sampled_cond_binned)
    if len(unique_conds) > 20:
        print(f'{len(unique_conds)} unique conditions, this may take a while')

    # Define consistent bins across all histograms
    min_val, max_val = np.min(feature), np.max(feature)
    # 30 bins requires 31 bin edges
    bins = np.linspace(min_val, max_val, nbins + 1)

    # Pre-compute all histograms using dictionary comprehension
    histograms = {
        cond: np.histogram(
            feature[sampled_cond_binned == cond], bins=bins, density=True)[0] + 1e-10
        for cond in unique_conds
    }

    # Generate all unique condition pairs using numpy's upper triangular indices
    indices = np.triu_indices(len(unique_conds), k=1)
    cond_pairs = unique_conds[np.array(indices)].T

    # Calculate KL divergence for all pairs using list comprehension
    kl_divs = np.array([entropy(histograms[cond1], histograms[cond2])
                        for cond1, cond2 in cond_pairs])
    kl_divs = np.concatenate([cond_pairs, kl_divs[:, None]], axis=-1)

    return kl_divs


def calculate_kl_divergence_aves(sample, sampled_cond_binned):
    kl_divs_avg = []
    for feature in sample.T:
        kl_divs = calculate_kl_divergence(feature, sampled_cond_binned)
        avg_kl = np.nanmean(kl_divs[:, 2])
        kl_divs_avg.append(avg_kl)
    return kl_divs_avg
    
    
def conditional_latent_entropy(z_samples, conditions, n_bins=20):
    """
    Calculate entropy of latent distributions conditioned on different prompts.
    Lower entropy indicates more specific prompt response.

    Parameters:
    -----------
    z_samples : torch.Tensor or numpy.ndarray
        Latent vectors from the VAE (shape: n_samples × latent_dim)
    conditions : list or numpy.ndarray
        Condition/prompt labels for each sample
    n_bins : int
        Number of bins for discretizing the latent space

    Returns:
    --------
    float
        Average conditional entropy across all latent dimensions
    dict
        Per-condition entropy values
    """

    unique_conditions = np.unique(conditions)
    latent_dim = z_samples.shape[1]

    # Calculate entropy for each condition and each latent dimension
    condition_entropies = {}
    avg_entropies = np.zeros(latent_dim)

    for condition in unique_conditions:
        condition_mask = (conditions == condition)
        condition_z = z_samples[condition_mask]

        # Skip if too few samples
        if len(condition_z) < 5:
            continue

        dim_entropies = []
        for dim in range(latent_dim):
            # Discretize the dimension
            hist, _ = np.histogram(
                condition_z[:, dim], bins=n_bins, density=True)
            # Add small epsilon to avoid log(0)
            hist = hist + 1e-10
            hist = hist / np.sum(hist)
            dim_entropy = entropy(hist)
            dim_entropies.append(dim_entropy)
            avg_entropies[dim] += dim_entropy * len(condition_z)

        condition_entropies[condition] = np.mean(dim_entropies)

    # Normalize by total samples
    avg_entropies = avg_entropies / len(z_samples)
    overall_entropy = np.mean(avg_entropies)

    return overall_entropy, condition_entropies


def latent_cluster_separation(z_samples, conditions):
    """
    Measure how well-separated the latent clusters are for different prompts.
    Higher values indicate more specific prompt encoding.

    Parameters:
    -----------
    z_samples : torch.Tensor or numpy.ndarray
        Latent vectors from the VAE
    conditions : list or numpy.ndarray
        Condition/prompt labels for each sample

    Returns:
    --------
    float
        Silhouette score measuring cluster separation
    """

    # Only calculate if we have enough samples and conditions
    unique_conditions = np.unique(conditions)
    if len(unique_conditions) < 2 or len(z_samples) < len(unique_conditions) * 3:
        return 0.0

    try:
        # Calculate silhouette score (how well clusters are separated)
        sil_score = silhouette_score(z_samples, conditions)
        return sil_score
    except ValueError:
        # Handle potential errors in silhouette calculation
        return 0.0
    
    
def kl_condition_prior(encoder_model, dataloader, condition_embedding_fn):
    """
    Measure KL divergence between condition-specific latent distributions and prior.
    Higher KL indicates more condition-specific encodings.

    Parameters:
    -----------
    encoder_model : torch.nn.Module
        VAE encoder model that outputs mean and log variance
    dataloader : torch.utils.data.DataLoader
        Data loader with input data and conditions
    condition_embedding_fn : callable
        Function to embed conditions into the format expected by the encoder

    Returns:
    --------
    dict
        Dictionary with per-condition KL divergence values
    """
    device = next(encoder_model.parameters()).device
    condition_kl_divs = {}
    condition_sample_counts = {}

    for batch in dataloader:
        # Unpack batch (adjust based on your specific dataloader)
        inputs, conditions = batch
        inputs = inputs.to(device)

        # Get condition embeddings
        condition_embeddings = condition_embedding_fn(conditions).to(device)

        # Get means and log variances from encoder
        mu, logvar = encoder_model(inputs, condition_embeddings)

        # Calculate KL divergence with standard normal prior
        # KL(N(μ,σ²) || N(0,1)) = 0.5 * (μ² + σ² - log(σ²) - 1)
        kl_divs = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        kl_divs = kl_divs.sum(dim=1)  # Sum over latent dimensions

        # Group by condition
        for i, condition in enumerate(conditions):
            cond_key = str(condition.item())
            if cond_key not in condition_kl_divs:
                condition_kl_divs[cond_key] = 0
                condition_sample_counts[cond_key] = 0

            condition_kl_divs[cond_key] += kl_divs[i].item()
            condition_sample_counts[cond_key] += 1

    # Calculate averages
    for cond_key in condition_kl_divs:
        if condition_sample_counts[cond_key] > 0:
            condition_kl_divs[cond_key] /= condition_sample_counts[cond_key]

    return condition_kl_divs


def mutual_information_latent_condition(z_samples, conditions, n_bins=20):
    """
    Estimate mutual information between latent space and conditions.
    Higher MI indicates stronger conditioning.

    Parameters:
    -----------
    z_samples : torch.Tensor or numpy.ndarray
        Latent vectors from the VAE
    conditions : list or numpy.ndarray
        Condition/prompt labels for each sample
    n_bins : int
        Number of bins for discretizing the latent space

    Returns:
    --------
    float
        Estimated mutual information
    list
        Per-dimension mutual information values
    """
    latent_dim = z_samples.shape[1]
    mi_per_dim = []

    for dim in range(latent_dim):
        # Discretize latent dimension
        z_dim = z_samples[:, dim]
        z_bins = np.linspace(z_dim.min(), z_dim.max(), n_bins + 1)
        z_binned = np.digitize(z_dim, z_bins[1:-1])

        # Calculate joint distribution P(z,c)
        joint_counts = np.zeros((n_bins, len(np.unique(conditions))))
        for z_val, c_val in zip(z_binned, conditions):
            c_idx = np.where(np.unique(conditions) == c_val)[0][0]
            joint_counts[z_val-1, c_idx] += 1

        # Convert to probability
        joint_prob = joint_counts / np.sum(joint_counts)

        # Calculate marginals
        z_prob = joint_prob.sum(axis=1)
        c_prob = joint_prob.sum(axis=0)

        # Calculate entropies (add small epsilon to avoid log(0))
        eps = 1e-10
        Hz = entropy(z_prob + eps)
        Hc = entropy(c_prob + eps)

        # Flatten and calculate joint entropy
        joint_prob_flat = joint_prob.flatten() + eps
        joint_prob_flat = joint_prob_flat / np.sum(joint_prob_flat)
        Hzc = entropy(joint_prob_flat)

        # Calculate mutual information I(Z;C) = H(Z) + H(C) - H(Z,C)
        mi = Hz + Hc - Hzc
        mi_per_dim.append(mi)

    return np.mean(mi_per_dim), mi_per_dim


def within_condition_variance_ratio(z_samples, conditions):
    """
    Calculate the ratio of within-condition variance to total variance.
    Lower values indicate more condition-specific latent representations.

    Parameters:
    -----------
    z_samples : torch.Tensor or numpy.ndarray
        Latent vectors from the VAE
    conditions : list or numpy.ndarray
        Condition/prompt labels for each sample

    Returns:
    --------
    float
        Ratio of within-condition variance to total variance
    """
    unique_conditions = np.unique(conditions)

    # Calculate total variance
    total_variance = np.var(z_samples, axis=0).sum()

    # Calculate within-condition variance
    within_variance = 0
    for condition in unique_conditions:
        condition_mask = (conditions == condition)
        condition_z = z_samples[condition_mask]
        if len(condition_z) > 1:  # Need at least 2 samples to calculate variance
            within_variance += np.var(condition_z,
                                      axis=0).sum() * len(condition_z)

    within_variance /= len(z_samples)

    # Return ratio (bounded between 0 and 1)
    if total_variance > 0:
        return within_variance / total_variance
    else:
        return 1.0  # If total variance is 0, return 1 (no separation)


def nearest_neighbor_condition_accuracy(z_samples, conditions, k=5):
    """
    Measure how well conditions can be identified from nearest neighbors in latent space.
    Higher accuracy indicates more condition-specific clusters.

    Parameters:
    -----------
    z_samples : torch.Tensor or numpy.ndarray
        Latent vectors from the VAE
    conditions : list or numpy.ndarray
        Condition/prompt labels for each sample
    k : int
        Number of nearest neighbors to consider

    Returns:
    --------
    float
        Average accuracy of condition prediction from nearest neighbors
    """
    # Ensure k is not larger than the number of samples minus one
    k = min(k, len(z_samples) - 1)

    # Find k-nearest neighbors for each sample
    # +1 because the sample itself is included
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(z_samples)
    distances, indices = nn.kneighbors(z_samples)

    # Calculate accuracy for each sample
    accuracies = []
    for i in range(len(z_samples)):
        # Skip the first index (which is the sample itself)
        neighbor_indices = indices[i, 1:]
        neighbor_conditions = conditions[neighbor_indices]

        # Calculate accuracy
        same_condition = (neighbor_conditions == conditions[i])
        accuracy = np.mean(same_condition)
        accuracies.append(accuracy)

    return np.mean(accuracies)
