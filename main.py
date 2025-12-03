"""
RNA Sequence-Graph Fusion Model
Main entry point for training and evaluation.
"""
import os
import argparse
import numpy as np
import torch

from src.data import FusionDataset
from src.models import create_unified_model
from src.training import evaluate_single_config
from src.utils import visualize_fusion_results, set_seed
import warnings
warnings.filterwarnings('ignore')


# Model configurations for ablation study
ABLATION_CONFIGS = {
    # Full model with dual attenuation fusion
    'dual_attenuation': {
        'use_seq': True, 'use_graph': True, 
        'use_prop': True, 'use_density': False, 'use_accum_freq': True,
        'fusion_type': 'dual_attenuation'
    },
    # Single modality models
    'seq_only': {
        'use_seq': True, 'use_graph': False, 
        'use_prop': True, 'use_density': False, 'use_accum_freq': True,
        'fusion_type': 'none'
    },
    'graph_only': {
        'use_seq': False, 'use_graph': True, 
        'use_prop': True, 'use_density': True, 'use_accum_freq': True,
        'fusion_type': 'none'
    },
    # Sequence channel ablation
    'seq_no_prop': {
        'use_seq': True, 'use_graph': True, 
        'use_prop': False, 'use_density': False, 'use_accum_freq': True,
        'fusion_type': 'dual_attenuation'
    },
    'seq_no_accum_freq': {
        'use_seq': True, 'use_graph': True, 
        'use_prop': True, 'use_density': False, 'use_accum_freq': False,
        'fusion_type': 'dual_attenuation'
    },
    # Single channel models
    'prop_only': {
        'use_seq': True, 'use_graph': False, 
        'use_prop': True, 'use_density': False, 'use_accum_freq': False,
        'fusion_type': 'none'
    },
    'accum_freq_only': {
        'use_seq': True, 'use_graph': False, 
        'use_prop': False, 'use_density': False, 'use_accum_freq': True,
        'fusion_type': 'none'
    },
}


def run_ablation_study(train_path, train_label_path, test_path, test_label_path, 
                       output_dir='results', configs=None):
    """
    Run ablation study with specified configurations.
    
    Args:
        train_path: Path to training FASTA file
        train_label_path: Path to training labels
        test_path: Path to test FASTA file
        test_label_path: Path to test labels
        output_dir: Output directory for results
        configs: Dictionary of configurations to evaluate (default: all)
    """
    print("Starting Fusion Model Ablation Study...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    
    # Set random seed
    set_seed(42)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FusionDataset(train_path, train_label_path)
    test_dataset = FusionDataset(test_path, test_label_path)
    
    # Device selection
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Use specified configs or all
    if configs is None:
        configs = ABLATION_CONFIGS
    
    # Store all results
    all_results = {}
    
    # Evaluate each configuration
    for config_name, config in configs.items():
        cv_results, test_metrics, importance = evaluate_single_config(
            config_name, config, train_dataset, test_dataset, device
        )
        
        all_results[config_name] = {
            'cv_results': cv_results,
            'test_metrics': test_metrics,
            'feature_importance': importance
        }
    
    # Save all results
    np.save(f'{output_dir}/all_results.npy', all_results)
    
    # Visualize results
    visualize_fusion_results(all_results, f'{output_dir}/figures')
    
    return all_results


def train_single_model(train_path, train_label_path, test_path, test_label_path,
                       config_name='dual_attenuation', output_dir='results'):
    """
    Train a single model configuration.
    
    Args:
        train_path: Path to training FASTA file
        train_label_path: Path to training labels
        test_path: Path to test FASTA file
        test_label_path: Path to test labels
        config_name: Name of configuration to use
        output_dir: Output directory for results
    """
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ABLATION_CONFIGS.keys())}")
    
    configs = {config_name: ABLATION_CONFIGS[config_name]}
    return run_ablation_study(
        train_path, train_label_path, test_path, test_label_path,
        output_dir, configs
    )


def main():
    parser = argparse.ArgumentParser(description='RNA Sequence-Graph Fusion Model')
    parser.add_argument('--train_fasta', type=str, required=True, help='Training FASTA file')
    parser.add_argument('--train_labels', type=str, required=True, help='Training labels (npy)')
    parser.add_argument('--test_fasta', type=str, required=True, help='Test FASTA file')
    parser.add_argument('--test_labels', type=str, required=True, help='Test labels (npy)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--config', type=str, default=None, 
                        help='Specific config to run (default: run all)')
    parser.add_argument('--ablation', action='store_true', help='Run full ablation study')
    
    args = parser.parse_args()
    
    if args.ablation or args.config is None:
        run_ablation_study(
            args.train_fasta, args.train_labels,
            args.test_fasta, args.test_labels,
            args.output_dir
        )
    else:
        train_single_model(
            args.train_fasta, args.train_labels,
            args.test_fasta, args.test_labels,
            args.config, args.output_dir
        )


if __name__ == '__main__':
    main()
