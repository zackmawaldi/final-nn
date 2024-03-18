# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    
    labels_set = set( labels )

    if len(labels_set) != 2:
        raise ValueError(f'Entered labels have more that two labels.')

    labels_count = {}

    for label in labels_set:
        labels_count[label] = labels.count( label )

    A, B = list( labels_set )

    # check for equality... means it's already balanced
    if labels_count[A] == labels_count[B]:
        return seqs, labels


    larger_label = A if labels_count[A] > labels_count[B] else B
    smaller_label = A if labels_count[A] < labels_count[B] else B

    only_larger_label = [seq for seq, label in zip(seqs, labels) if label == larger_label]
    only_smaller_label = [seq for seq, label in zip(seqs, labels) if label == smaller_label]

    
    upsampled_smaller_label = only_smaller_label.copy()
    # upsample smaller label
    for _ in range( labels_count[larger_label] - labels_count[smaller_label] ):
        upsampled_smaller_label.append( random.choice(only_smaller_label) )


    balanced_seq = []
    
    balanced_seq.extend(only_larger_label)
    balanced_seq.extend(upsampled_smaller_label)

    balanced_label = [larger_label] * labels_count[larger_label] + [smaller_label] * labels_count[larger_label] 

    # shuffle, just in case

    zipped = list( zip(balanced_seq, balanced_label) )
    random.shuffle(zipped)

    seqs, labels = zip(*zipped)

    return list(seqs), list(labels)


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    

    one_hot_dict = {'A': [1, 0, 0, 0],
                    'T': [0, 1, 0, 0],
                    'C': [0, 0, 1, 0],
                    'G': [0, 0, 0, 1]}
    
    one_hot_flat = []

    for seq in seq_arr:
        encoded_seq = []

        for base in seq:
            encoded_seq.extend(one_hot_dict[base])
        
        one_hot_flat.extend(encoded_seq)

    return np.array(one_hot_flat)
            

