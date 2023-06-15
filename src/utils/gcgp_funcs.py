from Bio import SeqIO
from typing import Union, Dict, List


def load_seq_from_FASTA(filename, as_type="list") -> Union[Dict,List]:
    assert filename.split('.')[-1] == 'fasta', f'Make sure the input file name {filename} is a FASTA file'
    fasta_records = SeqIO.parse(open(filename),'fasta')
    if as_type=="dict":
        sequences = {}
        for fasta_record in fasta_records:
            sequences[fasta_record.id] = str(fasta_record.seq)
        return sequences
    elif as_type=="list":
        sequences = []
        for fasta_record in fasta_records:
            sequences.append(str(fasta_record.seq))
        return sequences
    else:
        raise ValueError(f"Desired type {as_type} not supported.")
