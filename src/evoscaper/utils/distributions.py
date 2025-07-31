from scipy import stats
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
import numpy as np


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


def calculate_kde_overlap(sample1, sample2, bw_method=None, x_min=None, x_max=None, num_points=1000):
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
