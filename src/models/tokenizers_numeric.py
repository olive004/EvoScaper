
from numbers import Number
from typing import List, Tuple, Dict
import numpy as np
import regex as re
import jax

from src.utils.math import take_closest, scientific_notation, recombine_dec_exponent
from src.models.tokenizers import StandardTokenizer


def compute_log_tokens(granularity_list: np.ndarray, max_exp: int, min_exp: int) -> Dict: 
    s = jax.tree_util.tree_map(lambda e: [recombine_dec_exponent(g, e) for g in granularity_list], np.arange(min_exp, max_exp))
    scientific_notation()
    {d: i for d, i in zip()}
    


class PropertyTokenizer(StandardTokenizer):
    """ From XLNet (https://github.com/IBM/regression-transformer/)
    Run a property tokenization. 
    For turning floating point numbers into tokens. """

    def __init__(self) -> None:
        """Constructs a PropertyTokenizer."""
        self.regex = re.compile(r"\s*(<\w+>)\s*?(\+|-)?(\d+)(\.)?(\d+)?\s*")

    def tokenize(self, text: str) -> List[str]:
        """Tokenization of a property.

        Args:
            text: text to tokenize.

        Returns:
            extracted tokens.
        """
        tokens = []
        matched = self.regex.match(text)
        if matched:
            property_name, sign, units, dot, decimals = matched.groups()
            tokens = [property_name]
            if sign:
                tokens += [f"_{sign}_"]
            tokens += [
                f"_{number}_{position}_" for position, number in enumerate(units[::-1])
            ][::-1]
            if dot:
                tokens += [f"_{dot}_"]
            if decimals:
                tokens += [
                    f"_{number}_-{position}_"
                    for position, number in enumerate(decimals, 1)
                ]
        return tokens
    
    def batch_tokenize(self, sequences: List[str]) -> List[Tuple[List[str], List[int]]]:
        return super().batch_tokenize(sequences)
    
    
class LogTokenizer(StandardTokenizer):
    """ For tokenising floating point numbers on the log scale """
    
    def __init__(self,
                 granularity: int,
                 max_exp: int,
                 min_exp: int) -> None:
        """_summary_

        Args:
            granularity (int): The level of discretisation of the log scale
                taken as granularity per decimal exponent. A granularity of 2 
                would therefore mean each decimal point range would be split 
                into 2 bins.
            max_exp (int): Maximum range for decimal exponents. A 
                max_exp of 6 would mean input numbers max out at 10^6.
            min_exp (int): Minimum range for decimal exponents. A 
                min_exp of -6 means numbers cannot be less than 10^(-6).
        """
        
        super().__init__()
        
        self.granularity = granularity
        self.max_exp = max_exp
        self.min_exp = min_exp
        self.granularity_list = np.arange(0, 10, 10/granularity, dtype=np.float32)

        standard_tokens = compute_log_tokens(self.granularity_list, max_exp, min_exp)
        
        StandardTokenizer.__init__(
            self,
            standard_tokens=standard_tokens
            # unk_token=unk_token,
            # pad_token=pad_token,
            # mask_token=mask_token,
            # class_token=class_token,
            # eos_token=eos_token,
            # bos_token=bos_token,
            # prepend_bos_token=prepend_bos_token,
            # prepend_cls_token=prepend_cls_token,
            # append_eos_token=append_eos_token,
            # tokens_to_ids=tokens_to_ids,
        )

    def tokenize(self, val: Number) -> List[str]:
        val = np.log(val)
        raw_base_val, exponent = scientific_notation(val)
        base_val = take_closest(self.granularity_list, raw_base_val)
        exponent = np.min([np.max([exponent, self.min_exp]), self.max_exp])
        
        return recombine_dec_exponent(base_val, exponent)

