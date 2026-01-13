## InstaDeep interview question 
# #!/bin/python3

# import math
# import os
# import random
# import re
# import sys



# #
# # Complete the 'find_pdb_mapping' function below.
# #
# # The function accepts following arguments:
# #  1. (str) reference_sequence: the primary structure of the protein
# #  2. (List[Tuple[str, str]]) pdb_sequence: list of (PDB position, amino acid 1 letter code) tuples.
# #
# # PDB positions are represented as strings with the format "{chain_id}.{position}.{insertion_code}"
# # For example:
# # pos_1 = "A.1."
# # pos_2 = "A.2."
# # pos_3 = "A.2.A"
# #
# # The function is expected to return a list of Tuple[str, int] linking pdb positions to indexes (starting from 0)
# # in the reference sequence.
# #
# # For instance: [("A.1.", 0), ("A.2.", 1), ("A.2.A", 2)]
# #
# from typing import List, Tuple


# def find_pdb_mapping(reference_sequence, pdb_sequence) -> List[Tuple[str, int]]:
#     # Write your code here
#     ABC_LOOKUP = dict(zip(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), range(26)))
    
#     def get_pos_num(pos: str) -> int:
#         """ Get the numerical value in the position, eg
#         for pos = 'C.44.' return 44 """
#         return int(pos.split('.')[1])
        
#     def get_pos_char_idx(pos: str) -> int:
#         """ Get the index of the insertion code in the alphabet """
#         pos_split = pos.strip('.').split('.')
#         if len(pos_split) == 2:
#             return 0
#         elif len(pos_split) == 3:
#             return ABC_LOOKUP[pos_split[-1]]
#         else:
#             raise ValueError(f'Invalid PDB position {pos}')
    
#     def calc_pos_distance(pos1, pos2):
#         """ Calculate the distance between two consecutive PDB positions,
#         for example pos1 = 'A.10.B' and pos2 = 'A.12' (assuming no 
#         missing insertions) """

#         pos1_num = get_pos_num(pos1)
#         pos2_num = get_pos_num(pos2)
#         if pos1_num == pos2_num:
#             """ We are explicitly ignoring cases of missing insertion codes
#             between two positions with different numerical value """
#             return abs(get_pos_char_idx(pos2) - get_pos_char_idx(pos1))
#         return abs(pos2_num - pos1_num)
        
#     def extract_aa(sequences):
#         """ Extract just the amino acids from a list of (pos, aa) tuples """
#         return [aa for _pos, aa in sequences]
        
#     def assemble_pdb_groups(pdb_sequence) -> List[List[Tuple[str, str]]]:
#         """ Group consecutive pdb sequence entries into sub-lists to make 
#         their detection in the reference sequence easier.
#         Returns List of List of (pos, aa) tuples, for example: 
#         [
#             [('A.1.', 'M'), ('A.2.', 'S'), ('A.3.', 'L')],
#             [('A.5.', 'G'), ('A.6.', 'A')]
#         ]
#         """
#         idx_l = 0
#         pdb_grouped = []
#         prev_pos = pdb_sequence[0][0]
#         for idx_r in range(1, len(pdb_sequence)):
#             curr_pos, aa_pdb = pdb_sequence[idx_r]
#             distance = calc_pos_distance(curr_pos, prev_pos)
#             is_last = (idx_r == len(pdb_sequence) - 1)
#             if (distance > 1):
#                 pdb_grouped.append(pdb_sequence[idx_l:idx_r])
#                 idx_l = idx_r
#             if is_last:
#                 pdb_grouped.append(pdb_sequence[idx_l:idx_r+1])
#             prev_pos = curr_pos
#         assert ''.join([''.join(extract_aa(pdb_subgroup)) for pdb_subgroup in pdb_grouped]) == ''.join(extract_aa(pdb_sequence)), f'PDB sub-groups assembled incorrectly: {pdb_grouped} turned into {"".join(["".join(extract_aa(pdb_subgroup)) for pdb_subgroup in pdb_grouped])} != {"".join(extract_aa(pdb_sequence))}'
#         return pdb_grouped
    
#     # Initialise an empty list for the mapping
#     pdb_mapping = [None] * len(pdb_sequence)
        
#     # Group consecutive pdb sequence entries into tuples
#     pdb_grouped = assemble_pdb_groups(pdb_sequence)
        
#     # Map each group to the reference sequence
#     idx_pdb = idx_ref = 0 
#     for grouping in pdb_grouped:
#         substr = ''.join(extract_aa(grouping))
#         try:
#             idx_ref_next = reference_sequence.index(substr)
#             if idx_ref_next < idx_ref:
#                 idx_ref_next = reference_sequence[idx_ref:].index(substr) + idx_ref
#             idx_ref = idx_ref_next
#         except ValueError:
#             raise ValueError(f'PDB sub-sequence {grouping} not found, sub-groups assembled incorrectly')
            
#         for i in range(len(grouping)):
#             pdb_mapping[idx_pdb] = (grouping[i][0], i + idx_ref)
#             idx_pdb += 1
            
#     return pdb_mapping