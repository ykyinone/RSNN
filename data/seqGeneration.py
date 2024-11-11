import numpy as np
import pandas as pd


def generate_non_markov_sequence(seqnum, dim, ratio, filename, inverse=False):
    sequence = np.zeros((seqnum, dim), dtype=np.int32)
    
    initial_vector = np.random.randint(2, size=dim)
    sequence[0, :] = initial_vector
    
    for i in range(1, seqnum):
        n = int(i * ratio) 
        if n == 0:
            n = 1 
        
        reference_vectors = sequence[:i, :][-n:, :]
        
        masks = np.zeros((n, dim), dtype=np.int32)
        for k in range(n):
            mask = np.ones(dim, dtype=np.int32)
            if inverse:
                mask[:int(dim * (k / n))] = 0
            else:
                mask[:int(dim * (1 - k / n))] = 0 
            masks[k, :] = mask
        
        masked_vectors = reference_vectors & masks
        new_vector = np.bitwise_or.reduce(masked_vectors, axis=0)
        sequence[i, :] = new_vector
    
    df = pd.DataFrame(sequence)
    df.to_csv(filename, index=False)
    
    return sequence


def generate_shifted_sequence(seqnum, dim, segratio=2):
    assert segratio>1, "segratio must be more than 1" 

    sequence = np.zeros((seqnum, dim), dtype=np.int32)
    
    initial_vector = np.zeros(dim, dtype=np.int32)
    initial_vector[:dim // segratio] = 1
    
    sequence[0, :] = initial_vector
    
    for i in range(1, seqnum):
        shifted_vector = np.roll(initial_vector, i)
        sequence[i, :] = shifted_vector
    
    return sequence


