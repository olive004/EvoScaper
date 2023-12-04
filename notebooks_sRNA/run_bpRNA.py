

import os
import pandas as pd
import re
import numpy as np 
import subprocess


BPRNA_STRUCTURE_TYPES = ["S", "H", "B", "I", "M", "X", "E", "PK", "PKBP", "NCBP", "segment"]

def process_st(fn_st):
    
    # Match all digits in the string and replace them with an empty string
    pattern1 = r'[0-9]'
    pattern2 = "[^0-9]"
    
    with open(fn_st, 'r') as f:
        reads = [l.split(' ')[:2] for l in f.readlines()[7:]]
        motifs = [l[0] for l in reads]
        motifs = list(map(lambda x: [re.sub(pattern1, '', x.replace('.', '')), [int(re.sub(pattern2, '', xx)) for xx in x.split('.')]], motifs))
        
        motif_lengths = [l[1].split('..') for l in reads]
        motif_lengths = list(map(lambda x: int(x[1]) - int(x[0]) if len(x) > 1 else 1, motif_lengths))
        
    d = {}
    for (a, b), ml in zip(motifs, motif_lengths):
        d.setdefault(a, {}).setdefault('motif_groups', []).append(b)
        d.setdefault(a, {}).setdefault('motif_length', []).append(ml)
        
    return d
    
    
def aggregate_motifs(sim_data, dir_st='./data/08_comparison_sequences/2023_11_22_225107/st'):
    structures = pd.DataFrame()
    for id1 in sim_data:
        for id2 in sim_data[id1]:
            fn_st = os.path.join(dir_st, id1, id1 + '_' + id2 + '.st')
            if not os.path.isfile(fn_st):
                continue
            structures_d = {}
            d = process_st(fn_st)
            structures_d[('sRNA', '')] = id1
            structures_d[('Target', '')] = id2
            for s in BPRNA_STRUCTURE_TYPES:
                if d.get(s):
                    structures_d[('Mean Length', s)] = np.mean(d[s]['motif_length'])
                    structures_d[('Num in seq', s)] = len(d[s]['motif_groups'])
                else:
                    structures_d[('Mean Length', s)] = [0]
                    structures_d[('Num in seq', s)] = [0]
            structures = pd.concat([structures, pd.DataFrame.from_dict(structures_d)])
            
    structures = structures.reset_index().drop(columns='index')
    return structures[structures.columns[:2].to_list() + sorted(structures.columns[2:])]
    
    
def write_dbn(outname, outdir, id_name, seq, db):
    fn = os.path.join(outdir, outname + '.dbn')
    with open(fn, 'w') as f:
        f.writelines('>' + id_name + '\n' +
                    seq + '\n' +
                    db + '\n')
    return fn


def execute_perl_script(*args):
    """ Bard """
    script_path = './bpRNA.pl'
    try:
        subprocess.run(["perl", script_path, *args])
    except Exception as e:
        print(f"Error executing Perl script: {e}")


def run_bpRNA(sim_data, data, data_writer):
    for k1 in sim_data:
        data_writer.subdivide_writing('st')
        data_writer.subdivide_writing(k1, safe_dir_change=False)
        data_writer.unsubdivide()
        data_writer.subdivide_writing('dbn')
        data_writer.subdivide_writing(k1, safe_dir_change=False)
        
        for k2 in sim_data[k1]:
            # bplist = sim_data[k1][k2]['bpList']
            # make_db(bplist, seq_len=len(db))
            db = sim_data[k1][k2]['hybridDPfull'].replace('&', '')
            seq = data[data['Symbol'] == k1]['Sequence'].iloc[0] + \
                data[data['Symbol'] == k2]['Sequence'].iloc[0]
            fn = write_dbn(k1 + '_' + k2, data_writer.write_dir, id_name='arcZ', seq=seq, db=db)
            try:
                execute_perl_script(fn, fn.replace('.dbn', '').replace('dbn', 'st'))
            except:
                print('Could not write', k1, k2)
    data_writer.unsubdivide()
