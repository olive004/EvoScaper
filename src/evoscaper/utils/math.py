

from bisect import bisect_left
from numbers import Number
from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax


def arrayise(d):
    """ Make a nested dictionary into an array. Useful for loading
    previously saved weights back in from a json """
    for k, v in d.items():
        if type(v) == dict:
            for kk, vv in v.items():
                d[k][kk] = jnp.array(vv)
    return d


def bin_array(data, num_bins=10):
    original_shape = data.shape
    flattened_data = data.flatten()
    min_val = np.nanmin(flattened_data[~np.isinf(flattened_data)])
    max_val = np.nanmax(flattened_data[~np.isinf(flattened_data)])
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_means = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]
    bin_indices = np.clip(np.digitize(flattened_data, bin_edges) - 1, 0, num_bins - 1)
    binned_data = np.array([bin_means[idx] for idx in bin_indices])
    binned_data = binned_data.reshape(original_shape)
    bin_labels = [f"Bin {i}: [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}), Mean: {bin_means[i]:.2f}"
                 for i in range(num_bins)]
    return binned_data, bin_edges, bin_labels


def bin_to_nearest_edge(x: np.ndarray, n_bins):
    """ Bin the elements in x to the nearest lowest bin """
    edges = np.linspace(x.min(), x.max(), n_bins)
    return round_to_nearest_array(x, edges)


def calculate_conv_output(input_size: int, kernel_size: int, padding: int, stride: int):
    return int((input_size - kernel_size + 2 * padding) // stride + 1)


def convert_to_scientific_exponent(x, numerical_resolution: dict):
    exp_not = f'{x:.0e}'.split('e')
    resolution = numerical_resolution[int(exp_not[1])]
    base = int(10 / resolution)
    pre = custom_round(int(exp_not[0]), base=base)
    return int(exp_not[1]) + pre / 10


def convert_to_scientific_exponent_simple(x):
    return int(f'{x:.0e}'.split('e')[1])


def custom_round(x, base=5):
    return base * round(x/base)


def make_flat_triangle(matrices):
    """ Can be used with JAX vmap"""
    n = matrices.shape[-1]
    rows, cols = jnp.triu_indices(n)

    # Use advanced indexing that works with vmap
    return matrices[..., rows, cols]


def make_sequence_from_symmetrical(arr, side_length: int):
    """ For a symmetrical 2D matrix, return the upper triangle as a 1D array.
    For example, an array of shape [n, 3, 3] becomes [n, 6] """
    arr_flat = jax.vmap(
        lambda x: x[np.triu_indices(side_length)])(arr)
    return arr_flat


def make_symmetrical_matrix_from_sequence_nojax(arr, side_length: int):
    """ For a flat 1D array, make a symmetrical 2D matrix filling
    in the upper triangle with the 1D array. Not jax-friendly. """
    n = np.zeros((side_length, side_length))
    n[np.triu_indices(side_length)] = arr
    symmetric_matrix = n + n.T - np.diag(np.diag(n))
    return symmetric_matrix


def make_symmetrical_matrix_from_sequence(arr: jnp.ndarray, side_length: int) -> jnp.ndarray:
    """Base function to create a single symmetric matrix.
    This function operates on a single array and will be vmapped."""
    rows, cols = jnp.triu_indices(side_length)
    matrix = jnp.zeros((side_length, side_length))
    matrix = matrix.at[rows, cols].set(arr)
    symmetric_matrix = matrix + matrix.T
    diagonal_mask = jnp.eye(side_length, dtype=bool)
    symmetric_matrix = symmetric_matrix.at[diagonal_mask].multiply(0.5)
    return symmetric_matrix


def make_batch_symmetrical_matrices(arrs: jnp.ndarray, side_length: int) -> jnp.ndarray:
    """Vectorized version that handles batches of arrays.

    Args:
        arrs: Array of shape (batch_size, n_elements) where n_elements is the 
             number of elements needed for upper triangle
        side_length: The dimension of each output square matrix

    Returns:
        Array of shape (batch_size, side_length, side_length) containing 
        symmetric matrices
    """
    return jax.vmap(lambda x: make_symmetrical_matrix_from_sequence(x, side_length))(arrs)


def recombine_dec_exponent(base_num: Number, exponent: int) -> Number:
    return base_num * np.power(10.0, exponent)


def round_to_nearest_array(x, y):
    """ 
    x: array to be mapped over
    y: array with integers to round to """
    distances = np.abs(x[:, None] - y[None, :])
    nearest_indices = np.argmin(distances, axis=1)
    nearest_values = y[nearest_indices]
    return nearest_values


def scientific_exponent(value: Number) -> int:
    # GCG
    if value == 0:
        return 0

    exponent = int(np.floor(np.log10(np.abs(value))))
    return exponent


def scientific_notation(value: Number) -> Tuple[Number, int]:
    # Bard
    """Returns the numerical value and scientific exponent of a number.

    Args:
    number: The number.

    Returns:
    A tuple of two values: the numerical value and the scientific exponent.
    """

    exponent = scientific_exponent(value)
    numerical_value = value / (10 ** exponent)

    return numerical_value, exponent


def take_closest(listlike: list, num: Number) -> Number:
    """
    From https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(listlike, num)
    if pos == 0:
        return listlike[0]
    if pos == len(listlike):
        return listlike[-1]
    before = listlike[pos - 1]
    after = listlike[pos]
    if after - num < num - before:
        return after
    else:
        return before
