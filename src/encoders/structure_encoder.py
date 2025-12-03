"""DNA/RNA Structure Encoder based on dinucleotide properties."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class DNAStructureEncoder:
    """Encodes dinucleotide structural properties (twist, stacking energy, bendability)."""
    
    def __init__(self):
        # Dinucleotide structural parameters: [twist, stacking_energy, bendability]
        self.structural_params = {
            'AA': [38.9, -12, 3.07], 'AU': [33.81, -10.6, 2.6],
            'AC': [31.12, -11.8, 2.97], 'AG': [32.15, -11.5, 2.31],
            'UA': [33.28, -11.2, 6.74], 'UU': [38.9, -12, 3.07],
            'UC': [41.31, -11.4, 2.51], 'UG': [41.41, -12.3, 3.58],
            'CA': [41.41, -12.3, 3.58], 'CU': [32.15, -11.5, 2.31],
            'CC': [34.96, -9.5, 2.16], 'CG': [32.91, -13.1, 2.81],
            'GA': [41.31, -11.4, 2.51], 'GU': [31.12, -11.8, 2.97],
            'GG': [34.96, -9.5, 2.16], 'GC': [38.5, -13.2, 3.06]
        }
        self.scaler = StandardScaler().fit(
            np.array(list(self.structural_params.values()))
        )
    
    def encode(self, sequence):
        """
        Encode sequence with dinucleotide structural features.
        
        Args:
            sequence: RNA/DNA sequence string
            
        Returns:
            numpy array of shape (seq_len, 3) containing normalized structural features
        """
        sequence = sequence.upper().replace('T', 'U')
        features = []
        
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i + 2]
            features.append(self.structural_params.get(dinuc, [0, 0, 0]))
        features.append([0, 0, 0])  # Padding for last position
        
        features = np.array(features)
        return self.scaler.transform(features)
