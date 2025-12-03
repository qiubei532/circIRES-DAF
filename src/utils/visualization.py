"""Visualization utilities for model results."""
import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_fusion_results(all_results, output_dir='results/figures'):
    """
    Visualize fusion model ablation experiment results.
    
    Args:
        all_results: Dictionary containing results for all configurations
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configs = list(all_results.keys())
    
    # Extract metrics
    cv_aucs = [all_results[c]['cv_results']['mean_auc'] for c in configs]
    cv_auc_stds = [all_results[c]['cv_results']['std_auc'] for c in configs]
    test_aucs = [all_results[c]['test_metrics']['auc'] for c in configs]
    test_auprs = [all_results[c]['test_metrics']['aupr'] for c in configs]
    test_f1s = [all_results[c]['test_metrics']['f1'] for c in configs]
    
    # 1. AUC Comparison
    plt.figure(figsize=(14, 8))
    x = np.arange(len(configs))
    width = 0.35
    
    plt.bar(x - width/2, cv_aucs, width, yerr=cv_auc_stds,
            label='Cross-Validation AUC', color='#3498db', alpha=0.8, capsize=5)
    plt.bar(x + width/2, test_aucs, width,
            label='Test AUC', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Model Configuration', fontsize=14)
    plt.ylabel('AUC Score', fontsize=14)
    plt.title('Impact of Feature Fusion on Model Performance', fontsize=16)
    plt.xticks(x, configs, rotation=45, ha='right', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (cv_auc, test_auc) in enumerate(zip(cv_aucs, test_aucs)):
        plt.text(i - width/2, cv_auc + 0.01, f'{cv_auc:.3f}', ha='center', fontsize=10)
        plt.text(i + width/2, test_auc + 0.01, f'{test_auc:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Test Metrics Comparison
    plt.figure(figsize=(14, 8))
    width = 0.25
    
    plt.bar(x - width, test_aucs, width, label='AUC', color='#3498db')
    plt.bar(x, test_auprs, width, label='AUPR', color='#e74c3c')
    plt.bar(x + width, test_f1s, width, label='F1 Score', color='#2ecc71')
    
    plt.xlabel('Model Configuration', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Comparison of Test Set Metrics', fontsize=16)
    plt.xticks(x, configs, rotation=45, ha='right', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance (if available)
    model_key = 'dual_attenuation'
    if model_key in all_results:
        importance = all_results[model_key]['feature_importance']
        
        if 'sequence' in importance and 'graph' in importance:
            categories = ['Seq-Property', 'Seq-Density', 'Seq-AccumFreq', 'Graph']
            values = [
                importance['sequence'].get('property', 0),
                importance['sequence'].get('density', 0),
                importance['sequence'].get('accum_freq', 0),
                importance.get('graph', 0)
            ]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, values, color=['#3498db', '#e74c3c', '#9b59b6', '#2ecc71'])
            
            plt.title('Feature Importance in Fusion Model', fontsize=16)
            plt.ylabel('Relative Importance', fontsize=14)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Figures saved to {output_dir}/")
