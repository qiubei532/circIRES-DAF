"""Attention Mechanisms for feature weighting."""
import torch
import torch.nn as nn


class LightweightAttention(nn.Module):
    """Lightweight attention mechanism for feature weighting."""
    
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply attention weighting.
        
        Args:
            x: Input features [batch_size, feature_dim]
            
        Returns:
            weighted_x: Attention-weighted features
            weights: Attention weights
        """
        weights = self.attention(x)
        return x * weights, weights
