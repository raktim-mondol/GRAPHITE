"""
Model loading utilities for consistent MIL model handling.

This module provides utilities for loading and initializing trained models
in a way that's consistent with the step_1_part_2 implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.mil_model import MILHistopathModel


def load_mil_model(model_path, device, num_classes=2, feat_dim=512, proj_dim=128):
    """
    Load a trained MIL model with consistent parameters.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        num_classes: Number of output classes (default: 2 for binary classification)
        feat_dim: Feature dimension from the backbone (default: 512 for ResNet18)
        proj_dim: Projection dimension (default: 128)
        
    Returns:
        Loaded model in evaluation mode
    """
    # Initialize model with consistent parameters
    model = MILHistopathModel(
        num_classes=num_classes, 
        feat_dim=feat_dim, 
        proj_dim=proj_dim
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def get_model_transform(model):
    """
    Get the appropriate data transform for the model.
    
    Args:
        model: The MIL model
        
    Returns:
        Transform function for preprocessing images
    """
    data_config = timm.data.resolve_model_data_config(model.feature_extractor)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform


def get_target_layer_for_cam(model):
    """
    Get the target layer for CAM visualization.
    
    Args:
        model: The MIL model
        
    Returns:
        Target layer for CAM methods
    """
    # For ResNet-based models, use the last layer of layer4
    if hasattr(model.feature_extractor, 'layer4'):
        return model.feature_extractor.layer4[-1]
    elif hasattr(model.feature_extractor, 'stages'):
        # For other architectures like EfficientNet
        return model.feature_extractor.stages[-1]
    else:
        # Fallback: try to find the last convolutional layer
        layers = []
        for module in model.feature_extractor.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                layers.append(module)
        if layers:
            return layers[-1]
        else:
            raise ValueError("Could not find suitable target layer for CAM")


class MILModelWrapper(nn.Module):
    """
    Wrapper for MIL model to make it compatible with CAM methods.
    
    This wrapper is consistent with the step_1_part_2 implementation.
    """
    
    def __init__(self, model):
        super(MILModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        # CAM methods expect direct input, add batch dimension for MIL model
        x = x.unsqueeze(1)
        _, _, logits, _ = self.model(x)
        return logits 