"""Graph Neural Network Processor for RNA structure."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool


class GraphProcessor(nn.Module):
    """GNN module for processing RNA graph structures."""
    
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=3):
        super().__init__()
        
        # Use GraphSAGE convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim * 2))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 2))
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass through GNN layers.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment for nodes
            
        Returns:
            pooled: Graph-level features
            x: Node-level features
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Use sum pooling for graph-level representation
        pooled = global_add_pool(x, batch)
        return pooled, x
