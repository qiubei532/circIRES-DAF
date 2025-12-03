"""Training functions for fusion model."""
import copy
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_fusion_model(model, train_loader, val_loader=None, num_epochs=50, device=None):
    """
    Train the fusion model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        model: Trained model
        history: Training history
        best_epoch: Best epoch number
    """
    if device is None:
        device = torch.device('cpu')
    
    model = model.to(device)
    
    # Initialize optimizer with fixed seed
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=0.0001
    )
    
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    # Training history
    history = {'train_loss': [], 'train_auc': []}
    if val_loader:
        history['val_loss'] = []
        history['val_auc'] = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            # Prepare data
            batch['graph'] = batch['graph'].to(device)
            batch['seq_features'] = {k: v.to(device) for k, v in batch['seq_features'].items()}
            batch['label'] = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output, _, _ = model(batch)
            
            # Compute loss
            loss = criterion(output.squeeze(), batch['label'].squeeze())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            probs = torch.sigmoid(output.squeeze()).detach()
            all_train_preds.extend(probs.cpu().numpy())
            all_train_labels.extend(batch['label'].squeeze().cpu().numpy())
        
        # Calculate training metrics
        avg_train_loss = total_loss / len(train_loader)
        train_auc = roc_auc_score(all_train_labels, all_train_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['train_auc'].append(train_auc)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch['graph'] = batch['graph'].to(device)
                    batch['seq_features'] = {k: v.to(device) for k, v in batch['seq_features'].items()}
                    batch['label'] = batch['label'].to(device)
                    
                    output, _, _ = model(batch)
                    loss = criterion(output.squeeze(), batch['label'].squeeze())
                    
                    val_loss += loss.item()
                    probs = torch.sigmoid(output.squeeze())
                    all_val_preds.extend(probs.cpu().numpy())
                    all_val_labels.extend(batch['label'].squeeze().cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_auc = roc_auc_score(all_val_labels, all_val_preds)
            
            history['val_loss'].append(avg_val_loss)
            history['val_auc'].append(val_auc)
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
        else:
            scheduler.step(avg_train_loss)
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
        
        # Print epoch summary
        print(f'\nEpoch {epoch + 1}/{num_epochs} Summary:')
        print(f'Training Loss: {avg_train_loss:.4f}, AUC: {train_auc:.4f}')
        if val_loader:
            print(f'Validation Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}')
        print('-' * 50)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_epoch
