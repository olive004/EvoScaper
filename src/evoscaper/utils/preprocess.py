import numpy as np
import pandas as pd
import jax


from synbio_morpher.utils.misc.numerical import count_monotonic_group_lengths
from synbio_morpher.utils.misc.string_handling import string_to_tuple_list, convert_liststr_to_list
from synbio_morpher.utils.misc.type_handling import get_nth_elements
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols

SEQ_LENGTH = 20


def drop_duplicates_keep_first_n(df, column, n):
    """ GCG """
    indices = df[df.duplicated(subset=column, keep=False)].groupby(
        column).head(n).index
    all_duplicates_indices = df[df.duplicated(subset=column, keep=False)].index
    to_drop = list(set(all_duplicates_indices) - set(indices))
    df2 = df.drop(to_drop)
    return df2


def proc_info(info: pd.DataFrame, include_log: bool = True):
    info['num_interacting_all'] = info['num_interacting'] + \
        info['num_self_interacting']
    info['sp_distance'] = 0
    info.loc[(info['sensitivity_wrt_species-6'] <= 1) & (info['precision_wrt_species-6'] <= 10), 'sp_distance'] = np.sqrt(
        np.power(1-info['sensitivity_wrt_species-6'], 2) + np.power(10 - info['precision_wrt_species-6'], 2))
    info.loc[(info['sensitivity_wrt_species-6'] <= 1) & (info['precision_wrt_species-6']
                                                         > 10), 'sp_distance'] = np.absolute(info['sensitivity_wrt_species-6'] - 1)
    info.loc[(info['sensitivity_wrt_species-6'] > 1) & (info['precision_wrt_species-6']
                                                        <= 10), 'sp_distance'] = np.absolute(info['precision_wrt_species-6'] - 10)

    if type(info['mutation_type'].iloc[0]) == str:
        info['mutation_type'] = convert_liststr_to_list(
            info['mutation_type'].str)
    if type(info['mutation_positions'].iloc[0]) == str:
        info['mutation_positions'] = convert_liststr_to_list(
            info['mutation_positions'].str)

    #  Binding sites

    num_group_cols = [e.replace('energies', 'binding_sites_groups')
                      for e in get_true_interaction_cols(info, 'energies')]
    num_bs_cols = [e.replace('energies', 'binding_sites_count')
                   for e in get_true_interaction_cols(info, 'energies')]
    bs_idxs_cols = [e.replace('energies', 'binding_sites_idxs')
                    for e in get_true_interaction_cols(info, 'energies')]
    bs_range_cols = [e.replace('energies', 'binding_site_group_range')
                     for e in get_true_interaction_cols(info, 'energies')]

    for b, g, bs, bsi, r in zip(get_true_interaction_cols(info, 'binding_sites'), num_group_cols, num_bs_cols, bs_idxs_cols, bs_range_cols):
        # fbs = [string_to_tuple_list(bb) for bb in info[b]]
        fbs = jax.tree_util.tree_map(
            lambda bb: string_to_tuple_list(bb), info[b].to_list())
        first = get_nth_elements(fbs, empty_replacement=[], n=0)
        info[bs] = list(map(count_monotonic_group_lengths, first))
        info[bsi] = fbs
        info[g] = info[bs].apply(len)
        # info[r] = [[(bb[0], bb[-1]) for bb in b] for b in info[bsi]]
        info[r] = list(
            map(lambda b: list(map(lambda x: (x[0], x[-1]), b)), info[bsi]))

    # Mutation number ratiometric change

    numerical_cols = [c for c in info.columns if (type(info[(info['mutation_num'] > 0) & (info['eqconstants_0-0'] > 1)][c].iloc[0]) != str) and (
        type(info[c].iloc[0]) != list) and c not in get_true_interaction_cols(info, 'binding_sites')]
    key_cols = ['circuit_name', 'interacting',
                'mutation_name', 'name', 'sample_name']

    if include_log:
        grouped = info.groupby(['circuit_name', 'sample_name'], as_index=False)
        mutation_log = grouped[numerical_cols].apply(
            lambda x: np.log(x / x.loc[x['mutation_num'] == 0].squeeze()))
        mutation_log = mutation_log.reset_index()
        mutation_log[numerical_cols] = info[numerical_cols]
        info[[c + '_logm' for c in numerical_cols]
             ] = mutation_log[numerical_cols]

    return info, num_group_cols, num_bs_cols, numerical_cols, key_cols, bs_range_cols


def txt_to_csv(input_file: str, output_file: str):
    """ GCG """

    import csv

    # Open the input and output files
    with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write the header row to the CSV file
        header = next(txt_file).strip().split('\t')
        csv_writer.writerow(header)

        # Process the rest of the lines in the text file
        for line in txt_file:
            # Split the line by tab character
            data = line.strip().split('\t')
            # Write the data to the CSV file
            csv_writer.writerow(data)

    print("Conversion complete. CSV file saved as", output_file)
    

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


def make_xcols(data, x_type, include_diffs=False, remove_symmetrical=True):
    x_cols = list(get_true_interaction_cols(data, x_type, remove_symmetrical=remove_symmetrical))
    if include_diffs:
        x_cols = x_cols + \
            [[f'{i}_diffs' for i in get_true_interaction_cols(
                data, x_type, remove_symmetrical=True)]]
    return x_cols