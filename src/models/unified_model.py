"""Unified Fusion Model combining sequence and graph features."""
import torch
import torch.nn as nn

from .seq_processor import DynamicMultiChannelSeqProcessor
from .graph_processor import GraphProcessor
from .attention import LightweightAttention
from .fusion import DualAttenuationFusion


class UnifiedFusionModel(nn.Module):
    """Unified sequence-graph fusion model for RNA analysis."""
    
    def __init__(self, seq_hidden_dim=32, graph_hidden_dim=32, dropout=0.4,
                 use_seq=True, use_graph=True, use_prop=True, 
                 use_density=True, use_accum_freq=True,
                 fusion_type='simple'):
        super().__init__()
        
        # Feature configuration
        self.use_seq = use_seq
        self.use_graph = use_graph
        self.use_prop = use_prop if use_seq else False
        self.use_density = use_density if use_seq else False
        self.use_accum_freq = use_accum_freq if use_seq else False
        self.fusion_type = fusion_type
        
        # Calculate feature dimensions
        self.seq_dim = 0
        if self.use_seq:
            if self.use_prop:
                self.seq_dim += seq_hidden_dim * 2
            if self.use_density:
                self.seq_dim += seq_hidden_dim
            if self.use_accum_freq:
                self.seq_dim += seq_hidden_dim
                
        self.graph_dim = graph_hidden_dim * 2 if self.use_graph else 0
        
        # Sequence feature processing
        if self.use_seq:
            self.seq_processor = DynamicMultiChannelSeqProcessor(
                hidden_dim=seq_hidden_dim,
                use_prop=self.use_prop,
                use_density=self.use_density,
                use_accum_freq=self.use_accum_freq
            )
            if self.seq_dim > 0:
                self.seq_attention = LightweightAttention(self.seq_dim)
                self.seq_calibration = nn.Sequential(
                    nn.Linear(self.seq_dim, self.seq_dim),
                    nn.LayerNorm(self.seq_dim),
                    nn.ReLU()
                )
        
        # Graph structure processing
        if self.use_graph:
            self.graph_processor = GraphProcessor(
                input_dim=7, 
                hidden_dim=graph_hidden_dim, 
                num_layers=2
            )
            self.graph_attention = LightweightAttention(self.graph_dim)
            self.graph_calibration = nn.Sequential(
                nn.Linear(self.graph_dim, self.graph_dim),
                nn.LayerNorm(self.graph_dim),
                nn.ReLU()
            )
        
        # Fusion module
        if self.use_seq and self.use_graph:
            if fusion_type == 'dual_attenuation':
                self.fusion_module = DualAttenuationFusion(self.seq_dim, self.graph_dim)
        else:
            self.fusion_module = None
            
        # Output dimension
        self.output_dim = self.seq_dim + self.graph_dim
        if self.output_dim == 0:
            raise ValueError("At least one of sequence or graph features must be enabled")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.LayerNorm(self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim // 2, self.output_dim // 4),
            nn.LayerNorm(self.output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim // 4, 1)
        )
        
        # Feature importance
        importance_dim = 0
        if self.use_seq:
            if self.use_prop:
                importance_dim += 1
            if self.use_density:
                importance_dim += 1
            if self.use_accum_freq:
                importance_dim += 1
        if self.use_graph:
            importance_dim += 1
            
        if importance_dim > 0:
            self.feature_importance = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.output_dim // 2, importance_dim),
                nn.Softmax(dim=1)
            )
            
    def forward(self, data):
        seq_features = None
        graph_features = None
        seq_weights = None
        graph_weights = None
    
        # 1. Process sequence features
        if self.use_seq:
            seq_features = self.seq_processor(data['seq_features'])
            seq_features = seq_features.mean(dim=1)  # Global average pooling
            seq_features, seq_weights = self.seq_attention(seq_features)
            seq_features = self.seq_calibration(seq_features)
    
        # 2. Process graph features
        if self.use_graph:
            graph_data = data['graph']
            node_features = graph_data.x
            edge_index = graph_data.edge_index
            batch = graph_data.batch
        
            graph_features, _ = self.graph_processor(node_features, edge_index, batch)
            graph_features, graph_weights = self.graph_attention(graph_features)
            graph_features = self.graph_calibration(graph_features)
    
        # 3. Feature fusion
        feature_importance = {
            'sequence': {'property': 0.0, 'density': 0.0, 'accum_freq': 0.0},
            'graph': 0.0,
            'path': {'sequence': 0.0, 'graph': 0.0}
        }

        if self.use_seq and self.use_graph:
            if hasattr(self, 'fusion_module') and self.fusion_module is not None:
                combined_features, fusion_weights = self.fusion_module(seq_features, graph_features)
                feature_importance['path']['sequence'] = fusion_weights[:, 0].mean().item()
                feature_importance['path']['graph'] = fusion_weights[:, 1].mean().item()
                path_weights = fusion_weights
            else:
                combined_features = torch.cat([seq_features, graph_features], dim=1)
                path_weights = torch.ones(seq_features.size(0), 2).to(seq_features.device) * 0.5
                feature_importance['path']['sequence'] = 0.5
                feature_importance['path']['graph'] = 0.5
        elif self.use_seq:
            combined_features = seq_features
            path_weights = torch.tensor([[1.0, 0.0]]).expand(seq_features.size(0), -1).to(seq_features.device)
            feature_importance['path']['sequence'] = 1.0
        else:
            combined_features = graph_features
            path_weights = torch.tensor([[0.0, 1.0]]).expand(graph_features.size(0), -1).to(graph_features.device)
            feature_importance['path']['graph'] = 1.0

        # 4. Feature importance calculation
        if hasattr(self, 'feature_importance'):
            importance = self.feature_importance(combined_features)
            idx = 0
    
            if self.use_seq:
                if self.use_prop:
                    feature_importance['sequence']['property'] = importance[:, idx].mean().item()
                    idx += 1
                if self.use_density:
                    feature_importance['sequence']['density'] = importance[:, idx].mean().item()
                    idx += 1
                if self.use_accum_freq:
                    feature_importance['sequence']['accum_freq'] = importance[:, idx].mean().item()
                    idx += 1
    
            if self.use_graph:
                feature_importance['graph'] = importance[:, idx].mean().item()

        # 5. Prediction
        output = self.classifier(combined_features)

        attention_weights = {
            'seq': seq_weights,
            'graph': graph_weights,
            'path': path_weights
        }

        return output, feature_importance, attention_weights


def create_unified_model(config):
    """Create a unified fusion model from configuration."""
    return UnifiedFusionModel(
        seq_hidden_dim=32,
        graph_hidden_dim=64,
        dropout=0.3,
        use_seq=config['use_seq'],
        use_graph=config['use_graph'],
        use_prop=config.get('use_prop', True),
        use_density=config.get('use_density', False),
        use_accum_freq=config.get('use_accum_freq', True),
        fusion_type=config.get('fusion_type', 'simple')
    )
