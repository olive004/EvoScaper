import jax.numpy as jnp
import jax
import haiku as hk
import numpy as np
from typing import Dict, Any, Tuple, Literal, Union, Optional
    

class DataNormalizer:
    """
    A comprehensive data normalization utility with reversible transformations
    Supports multiple normalization techniques with inverse transformations
    """

    def __init__(self, categorical_n_bins: int = 10, categorical_method: str = 'equal_width'):
        self.metadata: Dict[str, Any] = {}
        self.categorical_n_bins = categorical_n_bins
        self.categorical_method = categorical_method

    def standardise(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
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

        self.metadata.update({
            'mean': mean,
            'std': std
        })

        return standardized

    def inverse_standardise(
        self,
        normalized_data: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Reverse standardise transformation

        Args:
            normalized_data (jnp.ndarray): Standardized data
            metadata (dict): Metadata from standardise

        Returns:
            jnp.ndarray: Original scale data
        """
        return normalized_data * self.metadata['std'] + self.metadata['mean']

    def min_max_scaling(
        self,
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

        self.metadata.update({
            'min_val': min_val,
            'scale': scale,
            'feature_range': feature_range
        })
        return scaled

    def inverse_min_max_scaling(
        self,
        normalized_data: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Reverse min-max scaling transformation

        Args:
            normalized_data (jnp.ndarray): Scaled data
            metadata (dict): Metadata from min-max scaling

        Returns:
            jnp.ndarray: Original scale data
        """
        min_range, max_range = self.metadata['feature_range']

        # Reverse the feature range scaling
        unscaled = (normalized_data - min_range) / (max_range - min_range)

        # Reverse the min-max transformation
        return unscaled * self.metadata['scale'] + self.metadata['min_val']

    def robust_scaling(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
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

        self.metadata.update({
            'median': median,
            'iqr': iqr
        })

        return robust_scaled

    def inverse_robust_scaling(
        self,
        normalized_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Reverse robust scaling transformation

        Args:
            normalized_data (jnp.ndarray): Robustly scaled data
            metadata (dict): Metadata from robust scaling

        Returns:
            jnp.ndarray: Original scale data
        """
        return normalized_data * self.metadata['iqr'] + self.metadata['median']

    def make_categorical(self, data: jnp.ndarray, n_bins: int=10, method='equal_width') -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Convert continuous data to categorical data

        Args:
            data (jnp.ndarray): Input data array

        Returns:
            Tuple of:
            - Categorical data
            - Metadata for reversing the transformation
        """
        # unique_values = jnp.unique(data)
        # categories = jnp.arange(len(unique_values))
        # category_map = dict(zip(unique_values, categories))

        # categorical_data = jnp.vectorize(category_map.get)(data)

        # self.metadata.update({
        #     'category_map': category_map
        # })

        categorical_data, bin_edges = ContinuousToCategorical.bin_data(
            data,
            n_bins=n_bins,
            method=method
        )

        if 'category_map' not in self.metadata:
            category_map = dict(zip(np.arange(n_bins), [np.mean([bin_edges[i], bin_edges[i+1]]) for i in range(n_bins)]))
            self.metadata.update({
                'category_map': category_map
            })
        return categorical_data
    
    def inverse_categorical(self, data: jnp.array):
        """
        Convert categorical data back to continuous data

        Args:
            data (jnp.ndarray): Categorical data

        Returns:
            jnp.ndarray: Continuous data
        """
        return np.vectorize(self.metadata['category_map'].get)(data)
    
    def categorical_onehot(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Convert continuous data to one-hot encoded categorical data

        Args:
            data (jnp.ndarray): Input data array

        Returns:
            Tuple of:
            - One-hot encoded data
            - Metadata for reversing the transformation
        """
        onehot_data = jax.nn.one_hot(data, int(data.max() + 1)).squeeze()

        return onehot_data
    
    def inverse_onehot(self, onehot_data: jnp.ndarray) -> jnp.ndarray:
        """
        Convert one-hot encoded data back to categorical data

        Args:
            onehot_data (jnp.ndarray): One-hot encoded data

        Returns:
            jnp.ndarray: Categorical data
        """
        return jnp.argmax(onehot_data, axis=-1)

    def negative_scaling(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Scale features to negative values

        Args:
            data (jnp.ndarray): Input data array

        Returns:
            jnp.ndarray: Negative scaled data
        """
        return -data

    def inverse_negative_scaling(
        self,
        normalized_data: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Reverse negative scaling transformation

        Args:
            normalized_data (jnp.ndarray): Negative scaled data

        Returns:
            jnp.ndarray: Original scale data
        """
        return -normalized_data

    def log_scaling(self, data: jnp.ndarray, zero_log_replacement=-10.0) -> jnp.ndarray:
        """
        Scale features to log values

        Args:
            data (jnp.ndarray): Input data array

        Returns:
            jnp.ndarray: Log scaled data
        """
        return jnp.where(data != 0, jnp.log10(data), zero_log_replacement)

    def inverse_log_scaling(
        self,
        normalized_data: jnp.ndarray,
        zero_log_replacement=-10.0
    ) -> jnp.ndarray:
        """
        Reverse log scaling transformation

        Args:
            normalized_data (jnp.ndarray): Log scaled data

        Returns:
            jnp.ndarray: Original scale data
        """
        return jnp.where(normalized_data != zero_log_replacement, jnp.power(10, normalized_data), 0)

    def create_chain_preprocessor(self, methods: list):
        """
        Create a Haiku module for chaining multiple normalization methods

        Args:
            methods (list): List of normalization methods to chain

        Returns:
            hk.Module: A normalization layer with forward and inverse methods
        """
        def chain_preprocess(x):
            for method in methods:
                if method == 'standardise':
                    x = self.standardise(x)
                elif method == 'min_max':
                    x = self.min_max_scaling(x)
                elif method == 'robust_scaling':
                    x = self.robust_scaling(x)
                elif method == 'negative':
                    x = self.negative_scaling(x)
                elif method == 'log':
                    x = self.log_scaling(x)
                elif method == 'categorical':
                    x = self.make_categorical(x, n_bins=self.categorical_n_bins, method=self.categorical_method)
                elif method == 'categorical_onehot':
                    x = self.categorical_onehot(x)
                else:
                    raise ValueError(
                        f"Unsupported normalization method: {method}")
            return x
        return chain_preprocess

    def create_chain_preprocessor_inverse(self, methods: list):
        def chain_inverse_preprocess(x):
            for method in reversed(methods):
                if method == 'standardise':
                    x = self.inverse_standardise(x)
                elif method == 'min_max':
                    x = self.inverse_min_max_scaling(x)
                elif method == 'robust_scaling':
                    x = self.inverse_robust_scaling(x)
                elif method == 'negative':
                    x = self.inverse_negative_scaling(x)
                elif method == 'log':
                    x = self.inverse_log_scaling(x)
                elif method == 'categorical':
                    x = self.inverse_categorical(x)
                elif method == 'categorical_onehot':
                    x = self.inverse_onehot(x)
                else:
                    raise ValueError(
                        f"Unsupported normalization method: {method}")
            return x

        return chain_inverse_preprocess

    @staticmethod
    def create_normalization_layer(method='standardise'):
        """
        Create a Haiku module for normalization with reversal capability

        Args:
            method (str): Normalization method to use

        Returns:
            hk.Module: A normalization layer with forward and inverse methods
        """
        def normalize_and_track(x):
            if method == 'standardise':
                return DataNormalizer.standardise(x)
            elif method == 'min_max':
                return DataNormalizer.min_max_scaling(x)
            elif method == 'robust_scaling':
                return DataNormalizer.robust_scaling(x)
            else:
                raise ValueError(f"Unsupported normalization method: {method}")

        return hk.Sequential([
            hk.Lambda(normalize_and_track)
        ])


class ContinuousToCategorical:
    """
    Utility for converting continuous data into categorical bins
    Supports multiple binning strategies
    """

    @staticmethod
    def bin_data(
        data: Union[jnp.ndarray, np.ndarray],
        n_bins: int = 5,
        method: Literal['equal_width', 'equal_frequency',
                        'quantile', 'custom'] = 'equal_width',
        custom_breaks: Optional[jnp.ndarray] = None
    ) -> tuple:
        """
        Convert continuous data to categorical bins with guaranteed sample distribution

        Args:
            data (array-like): Input continuous data
            n_bins (int): Number of bins to create
            method (str): Binning strategy
            custom_breaks (array-like, optional): Custom bin boundaries

        Returns:
            tuple: 
            - Binned categorical data (integer labels)
            - Bin edges used for transformation
        """
        # Convert to JAX array if not already
        data = jnp.asarray(data)

        # Compute bin edges based on method
        if method == 'equal_width':
            # Initial equal-width binning
            min_val, max_val = jnp.min(data), jnp.max(data)
            bin_edges = jnp.linspace(min_val, max_val, n_bins + 1)
        elif method == 'equal_frequency':
            # Percentile-based binning
            bin_edges = jnp.percentile(
                data,
                jnp.linspace(0, 100, n_bins + 1)
            )
        elif method == 'quantile':
            # Quantile-based binning
            bin_edges = jnp.quantile(
                data,
                jnp.linspace(0, 1, n_bins + 1)
            )
        elif method == 'custom':
            if custom_breaks is None:
                raise ValueError(
                    "Custom breaks must be provided for 'custom' method")
            bin_edges = jnp.asarray(custom_breaks)
        else:
            raise ValueError(f"Unsupported binning method: {method}")

        # Ensure unique bin edges
        bin_edges = jnp.unique(bin_edges)

        # Corrected binning: use all bin edges except the last one for comparison
        binned_data = jnp.searchsorted(bin_edges[:-1], data) - 1

        return binned_data, bin_edges

    @staticmethod
    def inverse_bin(
        binned_data: jnp.ndarray,
        bin_edges: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Convert categorical bins back to continuous representative values

        Args:
            binned_data (array-like): Categorical bin labels
            bin_edges (array-like): Bin edges from original binning

        Returns:
            jnp.ndarray: Continuous representative values for each bin
        """
        # Compute bin midpoints
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Map bin labels to midpoints
        return bin_midpoints[binned_data]

    @staticmethod
    def bin_summary(
        original_data: Union[jnp.ndarray, np.ndarray],
        binned_data: jnp.ndarray,
        bin_edges: jnp.ndarray
    ) -> dict:
        """
        Provide summary statistics for each bin

        Args:
            original_data (array-like): Original continuous data
            binned_data (array-like): Categorical bin labels
            bin_edges (array-like): Bin edges

        Returns:
            dict: Summary statistics for each bin
        """
        # Convert to JAX arrays
        original_data = jnp.asarray(original_data)
        binned_data = jnp.asarray(binned_data)

        # Prepare summary dictionary
        summary = {}

        for bin_label in range(len(bin_edges) - 1):
            # Create mask for current bin
            bin_mask = binned_data == bin_label

            # Extract data for this bin
            bin_data = original_data[bin_mask]

            # Compute summary statistics
            summary[bin_label] = {
                'count': jnp.sum(bin_mask),
                'mean': jnp.mean(bin_data) if bin_data.size > 0 else None,
                'median': jnp.median(bin_data) if bin_data.size > 0 else None,
                'min': jnp.min(bin_data) if bin_data.size > 0 else None,
                'max': jnp.max(bin_data) if bin_data.size > 0 else None,
                'bin_range': (bin_edges[bin_label], bin_edges[bin_label + 1])
            }

        return summary


def make_chain_f(data_norm_settings: NormalizationSettings):
    """ Helper function """
    datanormaliser = DataNormalizer()
    methods_preprocessing = []
    if data_norm_settings.negative:
        methods_preprocessing.append('negative')
    if data_norm_settings.log:
        methods_preprocessing.append('log')
    if data_norm_settings.categorical:
        methods_preprocessing.append('categorical')
        datanormaliser.categorical_n_bins = data_norm_settings.categorical_n_bins
        datanormaliser.categorical_method = data_norm_settings.categorical_method
        if data_norm_settings.categorical_onehot:
            methods_preprocessing.append('categorical_onehot')
    if data_norm_settings.standardise:
        methods_preprocessing.append('standardise')
    if data_norm_settings.robust_scaling:
        methods_preprocessing.append('robust_scaling')
    if data_norm_settings.min_max:
        methods_preprocessing.append('min_max')
    return datanormaliser, methods_preprocessing


# Example usage demonstrating reversible normalization
def main_normalise():
    # Simulate training data
    np.random.seed(0)
    training_data = np.random.randn(1000, 5) * 10 + 5

    # Convert to JAX array
    jax_data = jnp.array(training_data)

    normer = DataNormalizer()

    # standardise with reversal
    print("standardise Example:")
    standardized = normer.standardise(jax_data)
    reconstructed_std = normer.inverse_standardise(
        standardized)
    print("Original Data Mean:", jnp.mean(jax_data, axis=0))
    print("Reconstructed Data Mean:", jnp.mean(reconstructed_std, axis=0))
    print("Mean Difference:", jnp.mean(jnp.abs(jax_data - reconstructed_std)))

    # Min-Max Scaling with reversal
    print("\nMin-Max Scaling Example:")
    min_max_scaled = normer.min_max_scaling(jax_data)
    reconstructed_minmax = normer.inverse_min_max_scaling(
        min_max_scaled)
    print("Original Data Min:", jnp.min(jax_data, axis=0))
    print("Reconstructed Data Min:", jnp.min(reconstructed_minmax, axis=0))
    print("Mean Difference:", jnp.mean(
        jnp.abs(jax_data - reconstructed_minmax)))

    # Robust Scaling with reversal
    print("\nRobust Scaling Example:")
    robust_scaled = normer.robust_scaling(jax_data)
    reconstructed_robust = normer.inverse_robust_scaling(
        robust_scaled)
    print("Original Data Median:", jnp.median(jax_data, axis=0))
    print("Reconstructed Data Median:", jnp.median(reconstructed_robust, axis=0))
    print("Mean Difference:", jnp.mean(
        jnp.abs(jax_data - reconstructed_robust)))


# Example usage demonstration categorical binning
def main_categorical():
    # Generate sample continuous data with potential binning challenges
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=10, scale=2, size=300),  # Cluster 1
        np.random.normal(loc=50, scale=5, size=400),  # Cluster 2
        np.random.normal(loc=90, scale=3, size=300)   # Cluster 3
    ])

    # Convert to categorical using different methods
    methods = ['equal_width', 'equal_frequency', 'quantile']

    for method in methods:
        print(f"\nBinning Method: {method}")

        # Bin the data
        binned_data, bin_edges = ContinuousToCategorical.bin_data(
            data,
            n_bins=5,
            method=method
        )

        # Get bin summary
        summary = ContinuousToCategorical.bin_summary(
            data, binned_data, bin_edges)

        # Print bin summary
        for bin_label, bin_info in summary.items():
            print(f"Bin {bin_label}:")
            print(f"  Range: {bin_info['bin_range']}")
            print(f"  Count: {bin_info['count']}")
            print(f"  Mean: {bin_info['mean']}")


if __name__ == '__main__':
    # main_normalise()
    main_categorical()
