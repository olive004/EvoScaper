

from bisect import bisect_left
from numbers import Number
from typing import Tuple
import numpy as np


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


def recombine_dec_exponent(base_num: Number, exponent: int) -> Number:
    return base_num * np.power(10.0, exponent)


def calculate_conv_output(input_size: int, kernel_size: int, padding: int, stride: int):
    return int((input_size - kernel_size + 2 * padding) // stride + 1)


def convert_to_scientific_exponent(x): 
    return int(f'{x:.0e}'.split('e')[1])
