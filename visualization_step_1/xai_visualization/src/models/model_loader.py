"""
Model loading utilities.

This module provides utilities for loading and initializing trained models.
"""

import torch
from .mil_model import MILHistopathModel


def load_model(model_path, device, config):
    """
    Load a trained MIL model.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        config: Configuration object containing model parameters
        
    Returns:
        Loaded model in evaluation mode
    """
    # Model parameters
    feat_dim = getattr(config, 'feat_dim', 512)  # Default for ResNet18
    proj_dim = getattr(config, 'proj_dim', 128)
    num_classes = getattr(config, 'num_classes', 2)
    
    # Initialize model
    model = MILHistopathModel(
        num_classes=num_classes, 
        feat_dim=feat_dim, 
        proj_dim=proj_dim
    )
    
    # Load state dict with weights_only=False to handle PyTorch 2.6 security change
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model 