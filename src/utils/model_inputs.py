

import numpy as np


def construct_binding_img(binding_idxs: list, seq_length: int):
    img = np.zeros((seq_length, seq_length))
    for i, j in binding_idxs:
        img[i, j] = 1
    return img


def construct_binding_img_complex(binding_idxs: list, seq_length: int, sequence_q, sequence_t, binding_value_map: dict):
    """ Warning: sequence q (query) would be the first strand by ordinal number, but would actually be
    the bottom strand in the IntaRNA simulator. Conversely, sequence t (target) is the sequence displayed
    on the top. Here is an example: 

    query: CGGCGGUCGAAGAAUUCCCG target: CACGGCCGUUAUAUCACGUG
                3      10
                |      |
            5'-CA        AUAU...GUG-3'
                CGGCCGUU
                ++++++++
                GCUGGCGG
    3'-GCC...AGAA        C-5'
                |      |
                9      2


    The binding indices in the column `binding_sites_idxs`
    are a tuple where the first number is the index along the top strand and the second (decreasing) 
    number in the tuple set is the index along the bottom strand. """
    
    img = np.zeros((seq_length, seq_length))

    for t, q in binding_idxs:
        img[q, t] = binding_value_map[tuple(
            sorted((sequence_q[q], sequence_t[t])))]
    return img