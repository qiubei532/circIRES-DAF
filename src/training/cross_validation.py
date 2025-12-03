"""Cross-validation functions for model evaluation."""
import random
import copy
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from ..models import create_unified_model
from ..data import fusion_collate_fn
from .trainer import train_fusion_model
from .evaluator import evaluate_fusion_model


def evaluate_single_config(config_name, config, train_dataset, test_dataset, device):
    """
    Evaluate a single model configuration with cross-validation.
    
    Args:
        config_name: Name of the configuration
        config: Model configuration dictionary
        train_dataset: Training dataset
        test_dataset: Test dataset
        device: Device to use
        
    Returns:
        cv_results: Cross-validation results
        test_metrics: Test set metrics
        importance: Feature importance
    """
    print(f"\n{'-'*50}")
    print(f"Evaluating configuration: {config_name}")
    print(f"Config details: {config}")
    print(f"{'-'*50}")
    
    # Reset random seeds
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    
    # Initialize results storage
    cv_results = {
        'fold_aucs': [], 'fold_auprs': [], 'fold_accuracies': [],
        'fold_precisions': [], 'fold_recalls': [], 'fold_f1_scores': [],
        'fold_sensitivities': [], 'fold_specificities': [], 'fold_mccs': [],
        'best_epochs': [], 'best_models': []
    }
    
    # K-fold cross-validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    splits = list(kfold.split(range(len(train_dataset))))
    
    best_fold_auc = 0.0
    best_fold_idx = 0

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{k_folds}:")
        
        # Reset seeds for each fold
        np.random.seed(SEED + fold)
        torch.manual_seed(SEED + fold)
        torch.cuda.manual_seed_all(SEED + fold)
        random.seed(SEED + fold)
        
        # Create data loaders
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        g = torch.Generator()
        g.manual_seed(SEED + fold)
        
        train_loader = DataLoader(
            train_subset, batch_size=32, shuffle=True,
            num_workers=4, generator=g, collate_fn=fusion_collate_fn
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=32, shuffle=False,
            num_workers=4, collate_fn=fusion_collate_fn
        )
        
        # Create and train model
        model = create_unified_model(config)
        model, _, best_epoch = train_fusion_model(
            model, train_loader, val_loader, 
            num_epochs=50, device=device
        )
        cv_results['best_epochs'].append(best_epoch)
        
        # Evaluate on validation set
        val_metrics, _, _, _ = evaluate_fusion_model(model, val_loader, device)
        
        # Record results
        cv_results['fold_aucs'].append(val_metrics['auc'])
        cv_results['fold_auprs'].append(val_metrics['aupr'])
        cv_results['fold_accuracies'].append(val_metrics['accuracy'])
        cv_results['fold_precisions'].append(val_metrics['precision'])
        cv_results['fold_recalls'].append(val_metrics['recall'])
        cv_results['fold_f1_scores'].append(val_metrics['f1'])
        cv_results['fold_sensitivities'].append(val_metrics['sensitivity'])
        cv_results['fold_specificities'].append(val_metrics['specificity'])
        cv_results['fold_mccs'].append(val_metrics['mcc'])
        cv_results['best_models'].append(copy.deepcopy(model.state_dict()))
        
        if val_metrics['auc'] > best_fold_auc:
            best_fold_auc = val_metrics['auc']
            best_fold_idx = fold

        print(f"Fold {fold + 1} Validation AUC: {val_metrics['auc']:.4f}")
    
    # Calculate mean CV metrics
    cv_results['mean_best_epoch'] = int(np.mean(cv_results['best_epochs']))
    cv_results['std_best_epoch'] = np.std(cv_results['best_epochs'])
    cv_results['mean_auc'] = np.mean(cv_results['fold_aucs'])
    cv_results['std_auc'] = np.std(cv_results['fold_aucs'])
    cv_results['mean_aupr'] = np.mean(cv_results['fold_auprs'])
    cv_results['std_aupr'] = np.std(cv_results['fold_auprs'])
    cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
    cv_results['mean_precision'] = np.mean(cv_results['fold_precisions'])
    cv_results['mean_recall'] = np.mean(cv_results['fold_recalls'])
    cv_results['mean_f1'] = np.mean(cv_results['fold_f1_scores'])
    cv_results['mean_sensitivity'] = np.mean(cv_results['fold_sensitivities'])
    cv_results['mean_specificity'] = np.mean(cv_results['fold_specificities'])
    cv_results['mean_mcc'] = np.mean(cv_results['fold_mccs'])
    cv_results['std_sensitivity'] = np.std(cv_results['fold_sensitivities'])
    cv_results['std_specificity'] = np.std(cv_results['fold_specificities'])
    cv_results['std_mcc'] = np.std(cv_results['fold_mccs'])
    
    print(f"\nCross-Validation Summary for {config_name}:")
    print(f"Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
    print(f"Mean AUPR: {cv_results['mean_aupr']:.4f}")
    print(f"Mean F1: {cv_results['mean_f1']:.4f}")
    print(f"Mean Sensitivity: {cv_results['mean_sensitivity']:.4f}")
    print(f"Mean Specificity: {cv_results['mean_specificity']:.4f}")
    print(f"Mean MCC: {cv_results['mean_mcc']:.4f}")
    
    # Evaluate on test set using best fold model
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    
    train_g = torch.Generator()
    train_g.manual_seed(SEED)
    
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=fusion_collate_fn
    )
    
    final_model = create_unified_model(config)
    final_model.load_state_dict(cv_results['best_models'][best_fold_idx])
    
    print(f"\nEvaluating test set with best fold {best_fold_idx + 1} model...")
    test_metrics, importance, test_labels, test_probs = evaluate_fusion_model(
        final_model, test_loader, device
    )
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_prob': test_probs
    })
    predictions_df.to_excel(f'results/{config_name}_test_predictions.xlsx', index=False)
    
    # Save model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': config,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'feature_importance': importance
    }, f'results/{config_name}_model.pth')
    
    return cv_results, test_metrics, importance
