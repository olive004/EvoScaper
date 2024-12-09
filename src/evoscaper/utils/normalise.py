import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Dict, Any, Tuple


class DataNormalizer:
    """
    A comprehensive data normalization utility with reversible transformations
    Supports multiple normalization techniques with inverse transformations
    """

    @staticmethod
    def standardization(data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Standardize data to have zero mean and unit variance

        Args:
            data (jnp.ndarray): Input data array

        Returns:
            Tuple of:
            - Standardized data
            - Metadata for reversing the transformation
        """
        mean = jnp.mean(data, axis=0)
        std = jnp.std(data, axis=0)

        # Prevent division by zero
        std = jnp.where(std == 0, 1.0, std)

        standardized = (data - mean) / std

        return standardized, {
            'mean': mean,
            'std': std
        }

    @staticmethod
    def inverse_standardization(
        normalized_data: jnp.ndarray,
        metadata: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Reverse standardization transformation

        Args:
            normalized_data (jnp.ndarray): Standardized data
            metadata (dict): Metadata from standardization

        Returns:
            jnp.ndarray: Original scale data
        """
        return normalized_data * metadata['std'] + metadata['mean']

    @staticmethod
    def min_max_scaling(
        data: jnp.ndarray,
        feature_range: Tuple[float, float] = (0, 1)
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Scale features to a given range using min-max scaling

        Args:
            data (jnp.ndarray): Input data array
            feature_range (tuple): Desired output range, default (0, 1)

        Returns:
            Tuple of:
            - Scaled data
            - Metadata for reversing the transformation
        """
        min_val = jnp.min(data, axis=0)
        max_val = jnp.max(data, axis=0)

        # Prevent division by zero
        scale = max_val - min_val
        scale = jnp.where(scale == 0, 1.0, scale)

        # Map to desired feature range
        min_range, max_range = feature_range
        scaled = ((data - min_val) / scale) * \
            (max_range - min_range) + min_range

        return scaled, {
            'min_val': min_val,
            'scale': scale,
            'feature_range': feature_range
        }

    @staticmethod
    def inverse_min_max_scaling(
        normalized_data: jnp.ndarray,
        metadata: Dict[str, Any]
    ) -> jnp.ndarray:
        """
        Reverse min-max scaling transformation

        Args:
            normalized_data (jnp.ndarray): Scaled data
            metadata (dict): Metadata from min-max scaling

        Returns:
            jnp.ndarray: Original scale data
        """
        min_range, max_range = metadata['feature_range']

        # Reverse the feature range scaling
        unscaled = (normalized_data - min_range) / (max_range - min_range)

        # Reverse the min-max transformation
        return unscaled * metadata['scale'] + metadata['min_val']

    @staticmethod
    def robust_scaling(data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Scale features using median and interquartile range

        Args:
            data (jnp.ndarray): Input data array

        Returns:
            Tuple of:
            - Robustly scaled data
            - Metadata for reversing the transformation
        """
        median = jnp.median(data, axis=0)
        q1 = jnp.percentile(data, 25, axis=0)
        q3 = jnp.percentile(data, 75, axis=0)

        iqr = q3 - q1
        iqr = jnp.where(iqr == 0, 1.0, iqr)

        robust_scaled = (data - median) / iqr

        return robust_scaled, {
            'median': median,
            'iqr': iqr
        }

    @staticmethod
    def inverse_robust_scaling(
        normalized_data: jnp.ndarray,
        metadata: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Reverse robust scaling transformation

        Args:
            normalized_data (jnp.ndarray): Robustly scaled data
            metadata (dict): Metadata from robust scaling

        Returns:
            jnp.ndarray: Original scale data
        """
        return normalized_data * metadata['iqr'] + metadata['median']

    @staticmethod
    def create_normalization_layer(method='standardization'):
        """
        Create a Haiku module for normalization with reversal capability

        Args:
            method (str): Normalization method to use

        Returns:
            hk.Module: A normalization layer with forward and inverse methods
        """
        def normalize_and_track(x):
            if method == 'standardization':
                return DataNormalizer.standardization(x)
            elif method == 'min_max':
                return DataNormalizer.min_max_scaling(x)
            elif method == 'robust':
                return DataNormalizer.robust_scaling(x)
            else:
                raise ValueError(f"Unsupported normalization method: {method}")

        return hk.Sequential([
            hk.Lambda(normalize_and_track)
        ])

# Example usage demonstrating reversible normalization


def main():
    # Simulate training data
    np.random.seed(0)
    training_data = np.random.randn(1000, 5) * 10 + 5

    # Convert to JAX array
    jax_data = jnp.array(training_data)

    # Standardization with reversal
    print("Standardization Example:")
    standardized, std_metadata = DataNormalizer.standardization(jax_data)
    reconstructed_std = DataNormalizer.inverse_standardization(
        standardized, std_metadata)
    print("Original Data Mean:", jnp.mean(jax_data, axis=0))
    print("Reconstructed Data Mean:", jnp.mean(reconstructed_std, axis=0))
    print("Mean Difference:", jnp.mean(jnp.abs(jax_data - reconstructed_std)))

    # Min-Max Scaling with reversal
    print("\nMin-Max Scaling Example:")
    min_max_scaled, minmax_metadata = DataNormalizer.min_max_scaling(jax_data)
    reconstructed_minmax = DataNormalizer.inverse_min_max_scaling(
        min_max_scaled, minmax_metadata)
    print("Original Data Min:", jnp.min(jax_data, axis=0))
    print("Reconstructed Data Min:", jnp.min(reconstructed_minmax, axis=0))
    print("Mean Difference:", jnp.mean(
        jnp.abs(jax_data - reconstructed_minmax)))

    # Robust Scaling with reversal
    print("\nRobust Scaling Example:")
    robust_scaled, robust_metadata = DataNormalizer.robust_scaling(jax_data)
    reconstructed_robust = DataNormalizer.inverse_robust_scaling(
        robust_scaled, robust_metadata)
    print("Original Data Median:", jnp.median(jax_data, axis=0))
    print("Reconstructed Data Median:", jnp.median(reconstructed_robust, axis=0))
    print("Mean Difference:", jnp.mean(
        jnp.abs(jax_data - reconstructed_robust)))


if __name__ == '__main__':
    main()
