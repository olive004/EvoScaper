

from scipy import stats
from sklearn.metrics import roc_auc_score
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
    thresh_recall = jnp.expand_dims(
        jnp.nanmax(diff, axis=1) * perc_recall, axis=1)
    n_prompts_uniq = pred.shape[0]

    n_positives_inclass = jnp.nansum(diff < thresh_recall, axis=1)
    n_positives_all = jnp.nansum(diff < thresh_recall)
    n_positive_preds = pred.shape[1]

    precision = n_positives_inclass / n_positive_preds
    # * (n_prompts_uniq * perc_recall)
    recall = n_positives_inclass / n_positives_all

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
    filt_inf = np.where(np.isfinite(feature))
    min_val, max_val = np.nanmin(feature[filt_inf]), np.nanmax(feature[filt_inf])
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
    # # Convert conditions to array if not already
    # conditions = np.array(conditions)
    # if len(conditions.shape) == 1:
    #     conditions = conditions[:, None]

    # # Get unique values for each condition dimension
    # unique_conditions = [np.unique(conditions[:, i]) for i in range(conditions.shape[-1])]
    # latent_dim = z_samples.shape[1]

    # # Calculate entropy for each condition combination and each latent dimension
    # condition_entropies = {}
    # avg_entropies = np.zeros(latent_dim)
    # total_samples = 0

    # # Iterate through all combinations of condition values
    # for cond_n in unique_conditions:
    #     # Create mask for this condition combination
    #     mask = (conditions[:, 0] == cond_n[0])
    #     for i, cond_n_i in enumerate(cond_n[1:]):
    #         mask = mask & (conditions[:, i+1] == cond_n_i)
    #     condition_mask = mask
    #     condition_z = z_samples[condition_mask]

    #     # Skip if too few samples
    #     if len(condition_z) < 5:
    #         continue

    #     total_samples += len(condition_z)
    #     dim_entropies = []

    #     for dim in range(latent_dim):
    #         # Discretize the dimension
    #         hist, _ = np.histogram(condition_z[:, dim], bins=n_bins, density=True)
    #         # Add small epsilon to avoid log(0)
    #         hist = hist + 1e-10
    #         hist = hist / np.sum(hist)
    #         dim_entropy = entropy(hist)
    #         dim_entropies.append(dim_entropy)
    #         avg_entropies[dim] += dim_entropy * len(condition_z)

    #     condition_entropies[tuple(cond_n)] = np.mean(dim_entropies)

    # # Normalize by total samples
    # avg_entropies = avg_entropies / total_samples
    # overall_entropy = np.mean(avg_entropies)

    # return overall_entropy, condition_entropies

    unique_conditions = np.unique(conditions).tolist()
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
    overall_entropy = np.nanmean(avg_entropies)

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
    # Get unique values for each condition dimension
    # unique_conditions = [np.unique(conditions[:, i]) for i in range(conditions.shape[-1])]

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


def variation_of_information(P, Q, epsilon=1e-9):
    # Calculate the entropy of P and Q
    H_P = -np.sum(P * np.log(P + epsilon))
    H_Q = -np.sum(Q * np.log(Q + epsilon))

    # Calculate the joint entropy of P and Q
    P_Q = np.outer(P, Q)
    H_PQ = -np.sum(P_Q * np.log(P_Q + epsilon))

    # Calculate the variation of information
    VI = H_P + H_Q - 2 * H_PQ

    return VI


def hellinger_distance(P, Q):
    # Calculate the Hellinger distance
    H = np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2))
    return H


def wasserstein_distance(P, Q):
    # Calculate the cumulative distribution functions (CDFs)
    cdf_P = np.cumsum(P)
    cdf_Q = np.cumsum(Q)

    # Calculate the Wasserstein distance
    W = np.sum(np.abs(cdf_P - cdf_Q))

    return W


def jensen_shannon_divergence(P, Q, epsilon=1e-9):
    # Calculate the average distribution
    M = 0.5 * (P + Q)

    # Calculate the Kullback-Leibler divergence
    def kl_divergence(P, Q):
        return np.sum(P * np.log((P + epsilon) / (Q + epsilon)))

    # Calculate the Jensen-Shannon Divergence
    JSD = 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

    return JSD


def bhattacharyya_distance(P, Q):
    # Calculate the Bhattacharyya coefficient
    BC = np.sum(np.sqrt(P * Q))

    # Calculate the Bhattacharyya distance
    BD = -np.log(BC + 1e-9)

    return BD


def cross_entropy(P, Q, epsilon=1e-9):
    return entropy(P, Q + epsilon)


def total_variation_distance(P, Q):
    return 0.5 * np.sum(np.abs(P - Q))


def kolmogorov_smirnov_statistic(P, Q):
    # Calculate the cumulative distribution functions (CDFs)
    cdf_P = np.cumsum(P)
    cdf_Q = np.cumsum(Q)

    # Calculate the Kolmogorov-Smirnov Statistic
    KS_statistic = np.max(np.abs(cdf_P - cdf_Q))

    return KS_statistic


def area_under_roc_curve(P, Q):
    # Ensure P and Q are binary labels
    P_binary = (P > 0.5).astype(int)
    Q_binary = (Q > 0.5).astype(int)

    # Calculate the AUC
    auc = roc_auc_score(P_binary, Q_binary)

    return auc


def KL_per_dist(P, Q):
    divergence = np.abs(np.sum(P*np.log(P/Q)))
    return divergence


def calculate_distributional_overlap(distributions, dist_type='kde'):
    epsilon = 1e-8
    distributions = np.interp(
        distributions, (np.nanmin(distributions), np.nanmax(distributions)), (epsilon, 1))
    distributions = distributions / np.nansum(distributions)  # , axis=1)[:, None]

    if dist_type == 'kde':
        f = calculate_kde_overlap_core
    elif dist_type == 'binned':
        f = calculate_binned_overlap
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    overlaps = np.zeros((len(distributions), len(distributions)))
    for i in range(len(distributions)):
        for j in range(i, len(distributions)):
            overlaps[i, j] = f(distributions[i], distributions[j])
    overlaps[np.tril_indices(len(distributions))
             ] = overlaps.T[np.tril_indices(len(distributions))]
    return overlaps


# def calculate_binned_overlap_normalized(dist1, dist2, n_bins=50, epsilon=1e-8):
#     """
#     Calculate the overlap of two distributions binned into 50 bins, normalized so that the overlap sums to 1.

#     Parameters:
#     -----------
#     dist1 : numpy.ndarray
#         First distribution
#     dist2 : numpy.ndarray
#         Second distribution
#     n_bins : int
#         Number of bins for discretizing the distributions
#     epsilon : float
#         Small value to avoid division by zero

#     Returns:
#     --------
#     float
#         Normalized overlap between the two distributions
#     """
#     # Define consistent bins across all distributions
#     min_val = np.min([dist1.min(), dist2.min()])
#     max_val = np.max([dist1.max(), dist2.max()])
#     bins = np.linspace(min_val, max_val, n_bins + 1)

#     # Calculate histograms for each distribution
#     hist1, _ = np.histogram(dist1, bins=bins, density=True)
#     hist2, _ = np.histogram(dist2, bins=bins, density=True)

#     # Normalize histograms to sum to 1
#     hist1 = hist1 / (np.sum(hist1) + epsilon)
#     hist2 = hist2 / (np.sum(hist2) + epsilon)

#     # Calculate overlap
#     overlap = np.sum(np.minimum(hist1, hist2))

#     return overlap


def calculate_binned_overlap(dist1, dist2, n_bins=50, epsilon=1e-8):

    # Define consistent bins across all distributions
    min_val = np.min([dist1.min(), dist2.min()])
    max_val = np.max([dist1.max(), dist2.max()])
    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Calculate histograms for each distribution
    histograms1 = np.histogram(dist1, bins=bins, density=True)[0] + epsilon
    histograms2 = np.histogram(dist2, bins=bins, density=True)[0] + epsilon

    # Calculate overlap matrix
    return np.sum(np.minimum(histograms1, histograms2)) / np.sum(np.maximum(histograms1, histograms2))


def calculate_kde_overlap_core(sample1, sample2, bw_method=None, x_min=None, x_max=None, num_points=1000):
    sample1 = sample1[~np.isnan(sample1) & ~np.isinf(sample1)] 
    sample2 = sample2[~np.isnan(sample2) & ~np.isinf(sample2)] 
    
    if len(sample1) == 0 or len(sample2) == 0:
        return np.nan
    
    # Create the KDE objects
    kde1 = stats.gaussian_kde(sample1, bw_method=bw_method)
    kde2 = stats.gaussian_kde(sample2, bw_method=bw_method)

    # Determine the range to evaluate the KDEs
    if x_min is None:
        x_min = min(np.min(sample1), np.min(sample2))
        x_min = x_min - 0.1 * abs(x_min)  # Add some padding

    if x_max is None:
        x_max = max(np.max(sample1), np.max(sample2))
        x_max = x_max + 0.1 * abs(x_max)  # Add some padding

    # Create the evaluation points
    x = np.linspace(x_min, x_max, num_points)

    # Evaluate the KDEs at these points
    kde1_values = kde1(x)
    kde2_values = kde2(x)

    # Calculate the step size for numerical integration
    step = (x_max - x_min) / (num_points - 1)

    # Calculate the overlap by numerical integration of min(kde1, kde2)
    overlap_values = np.minimum(kde1_values, kde2_values)
    overlap = np.sum(overlap_values) * step

    # Calculate total areas (should be close to 1 for proper PDFs)
    # kde1_area = np.sum(kde1_values) * step
    # kde2_area = np.sum(kde2_values) * step

    # overlap_percentage = (overlap / min(kde1_area, kde2_area)) * 100

    return overlap  # , overlap_percentage, kde1_area, kde2_area, kde1, kde2, x


# Calculate the overlap as a percentage of the smaller distribution

# Kullback-Leibler (KL) Divergence: Measures how one probability distribution differs from another. It's not symmetric and isn't technically a distance metric.
# Jensen-Shannon Divergence: A symmetrized version of KL divergence that has a square root that is a true metric.
# Total Variation Distance: The maximum difference between probabilities assigned to the same event by the two distributions.
# Wasserstein Distance (Earth Mover's Distance): Measures the minimum "work" required to transform one distribution into another, where work is defined as the amount of distribution times the distance it has to be moved.
# Hellinger Distance: Measures the similarity between two probability distributions. It's bounded between 0 and 1, and is symmetric.
# Bhattacharyya Distance: Related to the Hellinger distance, it measures the overlap between two distributions.
# Maximum Mean Discrepancy (MMD): A kernel-based approach that measures the difference between distributions in a reproducing kernel Hilbert space.
# Cross-Entropy: Often used in machine learning, especially for classification tasks.
# f-Divergences: A family of functions that includes KL divergence and Hellinger distance.
# Kolmogorov-Smirnov Statistic: The maximum absolute difference between two cumulative distribution functions.
# Overlap Coefficient: The sum of the minimum value of each corresponding point in the distributions.
# Mahalanobis Distance: Useful for measuring distance between points in a multivariate space, accounting for correlations.
# Area Under ROC Curve (AUC): When treating the distributions as scores for binary classification, AUC measures separability.

# # Example usage
# P = analytics_reshaped[i]
# Q = analytics_reshaped[j]
# vi = variation_of_information(P, Q)
# print(f"Variation of Information: {vi}")
