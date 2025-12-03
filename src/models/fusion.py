"""Fusion Strategies for combining sequence and graph features."""
import torch
import torch.nn as nn


class DualAttenuationFusion(nn.Module):
    """Dual attenuation fusion - identifies and suppresses noisy features."""
    
    def __init__(self, seq_dim, graph_dim):
        super().__init__()
        
        # Feature quality assessment
        self.seq_quality = nn.Sequential(
            nn.Linear(seq_dim, seq_dim // 4),
            nn.LayerNorm(seq_dim // 4),
            nn.ReLU(),
            nn.Linear(seq_dim // 4, seq_dim),
            nn.Sigmoid()
        )
        
        self.graph_quality = nn.Sequential(
            nn.Linear(graph_dim, graph_dim // 4),
            nn.LayerNorm(graph_dim // 4),
            nn.ReLU(),
            nn.Linear(graph_dim // 4, graph_dim),
            nn.Sigmoid()
        )
        
        # Global representation importance
        self.global_importance = nn.Sequential(
            nn.Linear(seq_dim + graph_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, seq_feat, graph_feat):
        """
        Apply dual attenuation fusion.
        
        Args:
            seq_feat: Sequence features [batch, seq_dim]
            graph_feat: Graph features [batch, graph_dim]
            
        Returns:
            fused: Fused features
            global_weights: Modal importance weights
        """
        # 1. Local feature attenuation - suppress noisy features
        seq_quality_scores = self.seq_quality(seq_feat)
        graph_quality_scores = self.graph_quality(graph_feat)
        
        refined_seq = seq_feat * seq_quality_scores
        refined_graph = graph_feat * graph_quality_scores
        
        # 2. Global modal importance
        combined = torch.cat([refined_seq, refined_graph], dim=1)
        global_weights = self.global_importance(combined)
        
        # 3. Dual weighting
        weighted_seq = refined_seq * global_weights[:, 0:1]
        weighted_graph = refined_graph * global_weights[:, 1:2]
        
        # 4. Fusion with residual connection
        fused = torch.cat([weighted_seq, weighted_graph], dim=1)
        
        return fused, global_weights
