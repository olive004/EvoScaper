

from bisect import bisect_left
from numbers import Number
from typing import Tuple
import numpy as np
import jax.numpy as jnp


def arrayise(d):
    """ Make a nested dictionary into an array. Useful for loading
    previously saved weights back in from a json """
    for k, v in d.items():
        if type(v) == dict:
            for kk, vv in v.items():
                d[k][kk] = jnp.array(vv)
    return d


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


def make_symmetrical_matrix_from_sequence_nojax(arr, side_length: int):
    """ For a flat 1D array, make a symmetrical 2D matrix filling
    in the upper triangle with the 1D array. Not jax-friendly. """
    n = np.zeros((side_length, side_length))
    n[np.triu_indices(side_length)] = arr
    symmetric_matrix = n + n.T - np.diag(np.diag(n))
    return symmetric_matrix


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
