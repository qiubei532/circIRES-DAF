"""Fusion Dataset for RNA sequence and graph structure."""
import torch
import numpy as np
import RNA
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from ..encoders import (
    NucleotidePropertyEncoder, 
    DensityEncoder, 
    DNAStructureEncoder,
    BaseAccumulationFrequencyEncoder
)


class FusionDataset(Dataset):
    """Dataset combining sequence features and graph structure for RNA."""
    
    def __init__(self, fasta_file, label_file):
        """
        Initialize fusion dataset.
        
        Args:
            fasta_file: Path to FASTA sequence file
            label_file: Path to label file (NumPy format)
        """
        # Read sequences
        self.sequences = []
        with open(fasta_file, 'r') as f:
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        current_seq = current_seq.upper().replace('T', 'U')
                        self.sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                current_seq = current_seq.upper().replace('T', 'U')
                self.sequences.append(current_seq)
                
        # Read labels
        self.labels = np.load(label_file)
        
        # Validate data consistency
        assert len(self.sequences) == len(self.labels), \
            "Number of sequences and labels must match"
        
        # Initialize feature encoders
        self.prop_encoder = NucleotidePropertyEncoder()
        self.density_encoder = DensityEncoder()
        self.struct_encoder = DNAStructureEncoder()
        self.accum_freq_encoder = BaseAccumulationFrequencyEncoder()
        
        # Pre-compute all features and graph representations
        print("Pre-computing sequence and graph features...")
        self.seq_features = []
        self.graphs = []
        self.node_features = []
        
        for seq in tqdm(self.sequences):
            # Compute sequence features
            prop = self.prop_encoder.encode(seq)
            density = self.density_encoder.encode(seq)
            accum_freq = self.accum_freq_encoder.encode(seq)
            
            # Store sequence features
            self.seq_features.append({
                'prop': torch.tensor(prop, dtype=torch.float),
                'density': torch.tensor(density, dtype=torch.float),
                'accum_freq': torch.tensor(accum_freq, dtype=torch.float)
            })
            
            # Create graph structure and node features
            edge_index, node_feat = self._create_rna_graph(seq)
            self.graphs.append(edge_index)
            self.node_features.append(node_feat)
    
    def __len__(self):
        return len(self.sequences)
    
    def _create_rna_graph(self, sequence):
        """Create RNA graph structure and node features."""
        # Get RNA secondary structure
        dot_bracket, _ = RNA.fold(sequence)
        
        # Create one-hot encoded node features (4 dims)
        base_to_idx = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
        one_hot = np.zeros((len(sequence), 4))
        for i, base in enumerate(sequence):
            if base in base_to_idx:
                one_hot[i, base_to_idx[base]] = 1
            else:
                one_hot[i] = [0.25, 0.25, 0.25, 0.25]
        
        # Get dinucleotide structural features (3 dims)
        struct_features = self.struct_encoder.encode(sequence)
        
        # Combine features (7 dims: 4 one-hot + 3 structural)
        node_features = np.hstack((one_hot, struct_features))
        
        # Create edge index list
        edge_index = []
        stack = []
        
        # Add backbone edges (sequence continuity)
        for i in range(len(sequence) - 1):
            edge_index.extend([[i, i + 1], [i + 1, i]])
        
        # Add base pairing edges (secondary structure)
        for i, char in enumerate(dot_bracket):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                edge_index.extend([[i, j], [j, i]])
        
        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        return edge_index, node_features
    
    def __getitem__(self, idx):
        """Get a data sample."""
        seq_features = self.seq_features[idx]
        edge_index = self.graphs[idx]
        node_features = self.node_features[idx]
        graph = Data(x=node_features, edge_index=edge_index)
        
        return {
            'seq_features': seq_features,
            'graph': graph,
            'label': torch.FloatTensor([self.labels[idx]])
        }


def fusion_collate_fn(batch):
    """Collate function for batching sequence features and graph structures."""
    # Extract sequence features
    prop_features = torch.stack([d['seq_features']['prop'] for d in batch])
    density_features = torch.stack([d['seq_features']['density'] for d in batch])
    accum_freq_features = torch.stack([d['seq_features']['accum_freq'] for d in batch])
    
    seq_features = {
        'prop': prop_features,
        'density': density_features,
        'accum_freq': accum_freq_features
    }
    
    # Extract labels
    labels = torch.cat([d['label'] for d in batch])
    
    # Batch graph data using PyG's Batch class
    graphs = [d['graph'] for d in batch]
    batched_graphs = Batch.from_data_list(graphs)
    
    return {
        'seq_features': seq_features,
        'graph': batched_graphs,
        'label': labels
    }
