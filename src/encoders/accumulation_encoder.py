"""Base Accumulation Frequency Encoder for RNA sequences."""
import numpy as np


class BaseAccumulationFrequencyEncoder:
    """
    Encodes each base using:
    1. Specific vector representation: A(1,1,1), C(0,1,0), G(1,0,0), U(0,0,1)
    2. Cumulative frequency: count of base in first i positions divided by i
    
    Each base is represented by a 4-dim vector: [specific_vector(3) + cumulative_freq(1)]
    """
    
    def __init__(self):
        # Base-specific vector representations
        self.base_vectors = {
            'A': [1, 1, 1, 0],  # Last dim reserved for cumulative frequency
            'C': [0, 1, 0, 0],
            'G': [1, 0, 0, 0],
            'U': [0, 0, 1, 0],
            'T': [0, 0, 1, 0]   # Treat T same as U
        }
        
    def encode(self, sequence):
        """
        Encode sequence with base accumulation frequency features.
        
        Args:
            sequence: RNA/DNA sequence string
            
        Returns:
            numpy array of shape (seq_len, 4)
        """
        sequence = sequence.upper().replace('T', 'U')
        seq_len = len(sequence)
        features = np.zeros((seq_len, 4))
        
        # Counter for each base occurrence
        base_counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
        
        for i, base in enumerate(sequence):
            if base not in self.base_vectors:
                # Use average values for invalid bases
                vector = [0.25, 0.5, 0.25, 0]
            else:
                # Increment base count
                base_counts[base] += 1
                
                # Get first 3 dims of base vector
                vector = self.base_vectors[base].copy()
                
                # Calculate cumulative frequency
                vector[3] = base_counts[base] / (i + 1)
                
            features[i] = vector
            
        return features
