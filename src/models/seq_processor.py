"""Sequence Feature Processor with dynamic channel support."""
import torch
import torch.nn as nn


class DynamicMultiChannelSeqProcessor(nn.Module):
    """Multi-channel processor for sequence features with dynamic channel enabling."""
    
    def __init__(self, hidden_dim=64, use_prop=True, use_density=True, use_accum_freq=True):
        super().__init__()
        
        # Record enabled channels
        self.use_prop = use_prop
        self.use_density = use_density
        self.use_accum_freq = use_accum_freq
        self.out_feature_dim = 0
        
        # Property feature processing
        if use_prop:
            self.prop_conv = nn.Sequential(
                nn.Conv1d(6, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU()
            )
            self.out_feature_dim += hidden_dim * 2

        # Density feature processing
        if use_density:
            self.density_conv = nn.Sequential(
                nn.Conv1d(4, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.out_feature_dim += hidden_dim

        # Accumulation frequency feature processing
        if use_accum_freq:
            self.accum_freq_conv = nn.Sequential(
                nn.Conv1d(4, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.out_feature_dim += hidden_dim
            
    def forward(self, seq_features):
        feature_outputs = []
        
        # Process property features
        if self.use_prop and 'prop' in seq_features:
            prop_out = self.prop_conv(seq_features['prop'].transpose(1, 2))
            feature_outputs.append(prop_out)
        
        # Process density features
        if self.use_density and 'density' in seq_features:
            density_out = self.density_conv(seq_features['density'].transpose(1, 2))
            feature_outputs.append(density_out)

        # Process accumulation frequency features
        if self.use_accum_freq and 'accum_freq' in seq_features:
            accum_freq_out = self.accum_freq_conv(seq_features['accum_freq'].transpose(1, 2))
            feature_outputs.append(accum_freq_out)
        
        # Ensure at least one feature is enabled
        if not feature_outputs:
            raise ValueError("At least one sequence feature channel must be enabled")
            
        # Concatenate all enabled features
        if len(feature_outputs) > 1:
            combined = torch.cat(feature_outputs, dim=1)
        else:
            combined = feature_outputs[0]
            
        return combined.transpose(1, 2)  # [batch_size, seq_len, features]
