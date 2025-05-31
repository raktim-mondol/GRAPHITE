"""
Training utilities for MIL-based histopathology image classification.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, optimizer, criterion, device, epoch_num=None, total_epochs=None):
    """
    Training loop for one epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (callable): Loss function
        device (torch.device): Device to train on
        epoch_num (int, optional): Current epoch number for logging
        total_epochs (int, optional): Total number of epochs for logging
        
    Returns:        tuple: (avg_loss, metrics_dict) - average loss and metrics for the epoch
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    progress_desc = "Training" if epoch_num is None else f"Epoch {epoch_num}/{total_epochs} [Train]"
    pbar = tqdm(train_loader, desc=progress_desc, leave=False)
    for batch_idx, (patches, labels, _) in enumerate(pbar):
        # With custom_collate, patches is now a list of tensors with variable number of patches
        # Move each patch tensor to device
        patches = [p.to(device) for p in patches]
        labels = labels.to(device)
        
        # Forward pass - handle each patient's patches separately
        batch_logits = []
        for i, patient_patches in enumerate(patches):
            # Add a batch dimension for single patient
            patient_patches = patient_patches.unsqueeze(0)  # [1, num_patches, C, H, W]
            _, _, logits_i, _ = model(patient_patches)
            batch_logits.append(logits_i)
        
        # Stack all logits into a single batch tensor
        logits = torch.cat(batch_logits, dim=0)
        
        # Compute loss
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            loss = criterion(logits, labels.float())
        else:
            loss = criterion(logits, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Convert logits to predictions
        if logits.shape == labels.shape:  # Binary case with BCEWithLogitsLoss
            preds = (torch.sigmoid(logits) > 0.5).int()
            probs = torch.sigmoid(logits)
        else:  # Multi-class case with CrossEntropyLoss
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        if probs.ndim > 1 and probs.shape[1] > 1:
            all_probs.extend(probs[:, 1].detach().cpu().numpy())  # For multi-class, use prob of positive class
        else:
            all_probs.extend(probs.detach().cpu().numpy())
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
    
    # Compute metrics
    epoch_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # In case of single class prediction
        logger.warning("Could not compute AUC for training - possibly single class prediction")
        auc = 0.0
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc
    }
    
    return epoch_loss, metrics


def validate(model, val_loader, criterion, device, epoch_num=None, total_epochs=None):
    """
    Validation loop
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (callable): Loss function
        device (torch.device): Device to use for validation
        epoch_num (int): Current epoch number for logging
        total_epochs (int): Total number of epochs for logging
        
    Returns:
        tuple: (avg_loss, metrics_dict) - average loss and metrics for validation
    """
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_probs = []
    progress_desc = "Validation" if epoch_num is None else f"Epoch {epoch_num}/{total_epochs} [Val]"
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=progress_desc, leave=False)
        for patches, labels, _ in pbar:
            # With custom_collate, patches is now a list of tensors with variable number of patches
            # Move each patch tensor to device
            patches = [p.to(device) for p in patches]
            labels = labels.to(device)
            
            # Forward pass - handle each patient's patches separately
            batch_logits = []
            for i, patient_patches in enumerate(patches):
                # Add a batch dimension for single patient
                patient_patches = patient_patches.unsqueeze(0)  # [1, num_patches, C, H, W]
                _, _, logits_i, _ = model(patient_patches)
                batch_logits.append(logits_i)
            
            # Stack all logits into a single batch tensor
            logits = torch.cat(batch_logits, dim=0)
            
            # Compute loss
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(logits, labels.float())
            else:
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            # Convert logits to predictions
            if logits.shape == labels.shape:  # Binary case with BCEWithLogitsLoss
                preds = (torch.sigmoid(logits) > 0.5).int()
                probs = torch.sigmoid(logits)
            else:  # Multi-class case with CrossEntropyLoss
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            if probs.ndim > 1 and probs.shape[1] > 1:
                all_probs.extend(probs[:, 1].detach().cpu().numpy())  # For multi-class, use prob of positive class
            else:
                all_probs.extend(probs.detach().cpu().numpy())
    
    # Compute metrics
    epoch_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # In case of single class prediction
        logger.warning("Could not compute AUC for validation - possibly single class prediction")
        auc = 0.0
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc
    }
    
    return epoch_loss, metrics


def evaluate(model, test_loader, device, epoch=None, num_epochs=None):
    """
    Evaluation loop for test data
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device for evaluation
        epoch (int, optional): Current epoch number for logging
        num_epochs (int, optional): Total number of epochs for logging
        
    Returns:
        tuple: (accuracy, f1, auc) - Metrics for the test set
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    progress_desc = "Test" if epoch is None else f"Epoch {epoch+1}/{num_epochs} [Test]"
    pbar = tqdm(test_loader, desc=progress_desc, leave=False)
    with torch.no_grad():
        for patches, labels, _ in pbar:
            # With custom_collate, patches is now a list of tensors with variable number of patches
            # Move each patch tensor to device
            patches = [p.to(device) for p in patches]
            labels = labels.to(device)
            
            # Forward pass - handle each patient's patches separately
            batch_logits = []
            for i, patient_patches in enumerate(patches):
                # Add a batch dimension for single patient
                patient_patches = patient_patches.unsqueeze(0)  # [1, num_patches, C, H, W]
                _, _, logits_i, _ = model(patient_patches)
                batch_logits.append(logits_i)
            
            # Stack all logits into a single batch tensor
            logits = torch.cat(batch_logits, dim=0)
            
            # Get probabilities and predictions
            if logits.shape == labels.shape:  # Binary case
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
            else:  # Multi-class case
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
            # Update metrics
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Store probabilities for AUC calculation
            if probs.ndim > 1 and probs.shape[1] > 1:
                all_probs.extend(probs[:, 1].detach().cpu().numpy())  # For positive class in binary classification
            else:
                all_probs.extend(probs.detach().cpu().numpy())
                
            # Update progress bar
            pbar.set_postfix({'Acc': f'{correct/total:.4f}'})
    
    # Calculate final metrics
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        logger.warning("Could not compute AUC - possibly single class prediction")
        auc = 0.0
        
    return accuracy, f1, auc


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, 
          scheduler=None, early_stopping_patience=10, model_save_path='best_model.pth',
          metrics_to_monitor='auc'):
    """
    Full training loop with validation and early stopping
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (callable): Loss function
        optimizer (Optimizer): Optimizer
        device (torch.device): Device to use for training
        num_epochs (int): Number of epochs to train for
        scheduler (lr_scheduler): Learning rate scheduler
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping
        model_save_path (str): Path to save the model to
        metrics_to_monitor (str): Metric to monitor for early stopping and model saving
        
    Returns:
        tuple: (model, history) - trained model and training history
    """
    # Move model to device
    model.to(device)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_metrics': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Initialize early stopping variables
    best_metric_value = float('-inf') if metrics_to_monitor != 'loss' else float('inf')
    no_improvement_count = 0
    is_better = lambda x, y: x > y if metrics_to_monitor != 'loss' else x < y
    best_epoch = -1
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Monitoring {metrics_to_monitor} for early stopping with patience {early_stopping_patience}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch+1, num_epochs)
        
        # Validation phase
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch+1, num_epochs)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_metrics'].append(train_metrics)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        logger.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                current_metric = val_metrics[metrics_to_monitor] if metrics_to_monitor != 'loss' else val_loss
                scheduler.step(current_metric)
            else:
                scheduler.step()
        
        # Check if this is the best model so far
        current_metric = val_metrics[metrics_to_monitor] if metrics_to_monitor != 'loss' else val_loss
        
        if is_better(current_metric, best_metric_value):
            best_metric_value = current_metric
            no_improvement_count = 0
            best_epoch = epoch
            
            # Save the model
            save_model(model, model_save_path, {                'epoch': epoch+1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            logger.info(f"New best model saved! Best {metrics_to_monitor}: {best_metric_value:.4f}")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} epochs. Best {metrics_to_monitor}: {best_metric_value:.4f}")
        
        # Early stopping
        if no_improvement_count >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s. Best {metrics_to_monitor}: {best_metric_value:.4f} at epoch {best_epoch+1}")
    
    # Load the best model
    if os.path.exists(model_save_path):
        model = load_model(model, model_save_path)
    
    return model, history


def save_model(model, path, metrics=None):
    """
    Save a model and optionally its metrics
    
    Args:
        model (nn.Module): Model to save
        path (str): Path to save the model to
        metrics (dict, optional): Dictionary of metrics to save with the model
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save the model
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    save_dict = {'model_state_dict': state_dict}
    
    # Add metrics if provided
    if metrics is not None:
        save_dict['metrics'] = metrics
    
    torch.save(save_dict, path)


def load_model(model, path):
    """
    Load a model from a path
    
    Args:
        model (nn.Module): Model to load weights into
        path (str): Path to the saved model
        
    Returns:
        nn.Module: The model with loaded weights
    """
    checkpoint = torch.load(path)
    
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def test_model_on_dataset(model, test_loader, criterion, device):
    """
    Test a model on a dataset and return metrics
    
    Args:
        model (nn.Module): Model to test
        test_loader (DataLoader): Test data loader
        criterion (callable): Loss function
        device (torch.device): Device to use for testing
        
    Returns:
        dict: Dictionary of metrics
    """
    model.eval()
    
    # Get validation metrics
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    logger.info(f"Test results - Loss: {test_loss:.4f}, "
                f"Accuracy: {test_metrics['accuracy']:.4f}, "
                f"F1: {test_metrics['f1']:.4f}, "
                f"AUC: {test_metrics['auc']:.4f}")
    
    return test_metrics


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history (dict): Dictionary containing training and validation metrics
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    train_acc = [metrics['accuracy'] for metrics in history['train_metrics']]
    val_acc = [metrics['accuracy'] for metrics in history['val_metrics']]
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 score
    plt.subplot(2, 2, 3)
    train_f1 = [metrics['f1'] for metrics in history['train_metrics']]
    val_f1 = [metrics['f1'] for metrics in history['val_metrics']]
    plt.plot(train_f1, label='Train')
    plt.plot(val_f1, label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(2, 2, 4)
    train_auc = [metrics['auc'] for metrics in history['train_metrics']]
    val_auc = [metrics['auc'] for metrics in history['val_metrics']]
    plt.plot(train_auc, label='Train')
    plt.plot(val_auc, label='Validation')
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        Namespace: Parsed arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a MIL-based histopathology image classifier")
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to the training data directory')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to the validation data directory')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to the test data directory')
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--max_patches', type=int, default=100,
                        help='Maximum number of patches per image')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, choices=['step', 'plateau'], default='plateau',
                        help='Learning rate scheduler type')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--model_save_path', type=str, default='best_model.pth',
                        help='Path to save the best model')
    parser.add_argument('--metrics', type=str, default='auc',
                        help='Metrics to monitor for early stopping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for data loader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging training progress')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    
    args = parser.parse_args()
    
    # Print the arguments
    logger.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    return args
