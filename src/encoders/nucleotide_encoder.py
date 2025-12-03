"""Nucleotide Property Encoder for RNA sequences."""
import numpy as np
from collections import Counter


class NucleotidePropertyEncoder:
    """Encodes nucleotide properties including one-hot, purine/pyrimidine, and entropy."""
    
    def __init__(self):
        self.base_to_onehot = {
            'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1], 'T': [0, 0, 0, 1]
        }
        self.purine = {'A': 1, 'G': 1, 'C': 0, 'U': 0, 'T': 0}

    def _calculate_entropy(self, window):
        """Calculate Shannon entropy for a sequence window."""
        counts = Counter(window)
        total = len(window)
        return -sum((c / total) * np.log2(c / total) for c in counts.values())

    def encode(self, sequence):
        """
        Encode sequence with nucleotide properties.
        
        Args:
            sequence: RNA/DNA sequence string
            
        Returns:
            numpy array of shape (seq_len, 6) containing:
            - One-hot encoding (4 dims)
            - Purine/pyrimidine classification (1 dim)
            - Shannon entropy (1 dim)
        """
        seq = sequence.upper().replace('T', 'U')
        seq_len = len(seq)
        features = np.zeros((seq_len, 6))

        for i, base in enumerate(seq):
            # One-hot encoding (4 dims)
            features[i, 0:4] = self.base_to_onehot.get(base, [0.25, 0.25, 0.25, 0.25])
            
            # Purine/pyrimidine classification (1 dim)
            features[i, 4] = self.purine.get(base, 0.5)
            
            # Shannon entropy (1 dim)
            start = max(0, i - 3)
            end = min(seq_len, i + 4)
            features[i, 5] = self._calculate_entropy(seq[start:end])

        return features
