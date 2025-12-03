"""Density Encoder for RNA sequences."""
import numpy as np


class DensityEncoder:
    """Calculates local nucleotide density features around each position."""
    
    def __init__(self, window_size=5):
        """
        Initialize density encoder.
        
        Args:
            window_size: Size of the sliding window for density calculation
        """
        self.window_size = window_size

    def encode(self, sequence):
        """
        Encode sequence with local nucleotide density features.
        
        Args:
            sequence: RNA/DNA sequence string
            
        Returns:
            numpy array of shape (seq_len, 4) containing density of A, U, C, G
        """
        sequence = sequence.upper().replace('T', 'U')
        seq_len = len(sequence)
        density_features = np.zeros((seq_len, 4))

        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            window = sequence[start:end]

            density_features[i, 0] = window.count('A') / len(window)
            density_features[i, 1] = window.count('U') / len(window)
            density_features[i, 2] = window.count('C') / len(window)
            density_features[i, 3] = window.count('G') / len(window)

        return density_features
