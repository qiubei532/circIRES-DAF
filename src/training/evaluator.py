"""Evaluation functions for fusion model."""
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, 
    matthews_corrcoef, confusion_matrix
)


def evaluate_fusion_model(model, data_loader, device=None):
    """
    Evaluate fusion model performance.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        
    Returns:
        metrics: Dictionary of evaluation metrics
        avg_importance: Average feature importance
        all_labels: True labels
        all_probs: Predicted probabilities
    """
    if device is None:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    all_importance = {
        'sequence': {'property': [], 'density': [], 'accum_freq': []},
        'graph': [],
        'path': {'sequence': [], 'graph': []}
    }
    
    with torch.no_grad():
        for batch in data_loader:
            batch['graph'] = batch['graph'].to(device)
            batch['seq_features'] = {k: v.to(device) for k, v in batch['seq_features'].items()}
            batch['label'] = batch['label'].to(device)
            
            output, importance, _ = model(batch)
            probs = torch.sigmoid(output.squeeze())
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch['label'].squeeze().cpu().numpy())
            
            # Collect feature importance
            all_importance['sequence']['property'].append(importance['sequence']['property'])
            all_importance['sequence']['density'].append(importance['sequence']['density'])
            all_importance['sequence']['accum_freq'].append(importance['sequence']['accum_freq'])
            all_importance['graph'].append(importance['graph'])
            all_importance['path']['sequence'].append(importance['path']['sequence'])
            all_importance['path']['graph'].append(importance['path']['graph'])
    
    # Calculate average feature importance
    avg_importance = {
        'sequence': {
            'property': np.mean(all_importance['sequence']['property']),
            'density': np.mean(all_importance['sequence']['density']),
            'accum_freq': np.mean(all_importance['sequence']['accum_freq'])
        },
        'graph': np.mean(all_importance['graph']),
        'path': {
            'sequence': np.mean(all_importance['path']['sequence']),
            'graph': np.mean(all_importance['path']['graph'])
        }
    }
    
    # Calculate evaluation metrics
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    all_labels = np.array(all_labels)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn)  # Sensitivity (SN) = Recall = TPR
    specificity = tn / (tn + fp)  # Specificity (SP) = TNR
    mcc = matthews_corrcoef(all_labels, all_preds)

    metrics = {
        'auc': roc_auc_score(all_labels, all_probs),
        'aupr': average_precision_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc
    }
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"AUPR: {metrics['aupr']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    print(f"Sequence - Property: {avg_importance['sequence']['property']:.4f}")
    print(f"Sequence - Density: {avg_importance['sequence']['density']:.4f}")
    print(f"Sequence - Accum. Freq.: {avg_importance['sequence']['accum_freq']:.4f}")
    print(f"Graph Structure: {avg_importance['graph']:.4f}")
    print(f"Path - Sequence: {avg_importance['path']['sequence']:.4f}")
    print(f"Path - Graph: {avg_importance['path']['graph']:.4f}")
    
    return metrics, avg_importance, all_labels, all_probs
