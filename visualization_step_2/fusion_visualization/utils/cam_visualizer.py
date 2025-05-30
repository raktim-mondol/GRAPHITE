"""
CAM-based visualization methods adapted from step_1_part_2.

This module implements various Class Activation Mapping (CAM) methods for explaining
model predictions on histopathology images, integrated with the existing visualization system.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import timm
import logging

# Import pytorch-grad-cam methods
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, 
    AblationCAM, XGradCAM, EigenCAM, FullGrad
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .model_utils import MILModelWrapper, get_target_layer_for_cam, get_model_transform


class CAMVisualizer:
    """CAM-based visualization methods integrated with existing system."""
    
    def __init__(self, model, device, cam_method='fullgrad', target_class=1):
        """
        Initialize CAM visualizer.
        
        Args:
            model: The trained MIL model
            device: Computation device
            cam_method: CAM method name ('gradcam', 'hirescam', 'fullgrad', etc.)
            target_class: Target class index for CAM (default: 1 for cancer)
        """
        self.model = model
        self.device = device
        self.cam_method = cam_method.lower()
        self.target_class = target_class
        
        # Setup data transforms - consistent with step_1_part_2
        self.transform = get_model_transform(model)
        
        # Wrap model for CAM compatibility - consistent with step_1_part_2
        self.wrapped_model = MILModelWrapper(model)
        
        # Get target layer for CAM - consistent with step_1_part_2
        self.target_layer = get_target_layer_for_cam(model)
        
        # Initialize CAM method
        self.cam = self._get_cam_method()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _get_cam_method(self):
        """Initialize the appropriate CAM method."""
        cam_methods = {
            'gradcam': GradCAM,
            'hirescam': HiResCAM,
            'scorecam': ScoreCAM,
            'gradcampp': GradCAMPlusPlus,
            'gradcam++': GradCAMPlusPlus,  # Alternative name
            'ablationcam': AblationCAM,
            'xgradcam': XGradCAM,
            'eigencam': EigenCAM,
            'fullgrad': FullGrad
        }
        
        if self.cam_method not in cam_methods:
            available_methods = ', '.join(cam_methods.keys())
            raise ValueError(f"Unknown CAM method: {self.cam_method}. Available methods: {available_methods}")
        
        cam_class = cam_methods[self.cam_method]
        return cam_class(model=self.wrapped_model, target_layers=[self.target_layer])
    
    def process_patch(self, patch_img, patch_coords=None):
        """
        Process a single patch through the CAM method.
        
        Args:
            patch_img: PIL Image or numpy array of the patch
            patch_coords: Optional coordinates for debugging
            
        Returns:
            CAM activation map as numpy array
        """
        try:
            # Convert to PIL if numpy array
            if isinstance(patch_img, np.ndarray):
                patch_pil = Image.fromarray(patch_img)
            else:
                patch_pil = patch_img
            
            # Transform patch - consistent with step_1_part_2
            patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)
            
            # Generate CAM - consistent with step_1_part_2
            grayscale_cam = self.cam(
                input_tensor=patch_tensor, 
                targets=[ClassifierOutputTarget(self.target_class)]
            )
            
            # Handle CAM output format - consistent with step_1_part_2
            if isinstance(grayscale_cam, tuple):
                grayscale_cam = grayscale_cam[0]
            else:
                grayscale_cam = grayscale_cam[0]  # First element in batch
            
            if len(grayscale_cam.shape) > 2:
                grayscale_cam = np.mean(grayscale_cam, axis=0)
            
            return grayscale_cam
            
        except Exception as e:
            coords_str = f" at {patch_coords}" if patch_coords else ""
            self.logger.warning(f"Error processing patch{coords_str}: {str(e)}")
            return None
    
    def process_patches_batch(self, patches_data, original_shape):
        """
        Process multiple patches and create a combined CAM map.
        
        Args:
            patches_data: List of tuples (patch_img, x, y, patch_size)
            original_shape: (H, W) of the original image
            
        Returns:
            Combined CAM map as numpy array
        """
        H, W = original_shape
        cam_map = np.zeros((H, W))
        processed_count = 0
        
        for patch_img, x, y, patch_size in patches_data:
            cam_result = self.process_patch(patch_img, (x, y))
            
            if cam_result is not None:
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                cam_map[y:y_end, x:x_end] = cam_result
                processed_count += 1
        
        self.logger.info(f"Processed {processed_count}/{len(patches_data)} patches with {self.cam_method.upper()}")
        return cam_map
    
    def get_method_name(self):
        """Get the display name for the CAM method."""
        method_names = {
            'gradcam': 'Grad-CAM',
            'hirescam': 'HiRes-CAM',
            'scorecam': 'Score-CAM', 
            'gradcampp': 'Grad-CAM++',
            'gradcam++': 'Grad-CAM++',
            'ablationcam': 'Ablation-CAM',
            'xgradcam': 'XGrad-CAM',
            'eigencam': 'Eigen-CAM',
            'fullgrad': 'FullGrad'
        }
        return method_names.get(self.cam_method, self.cam_method.upper())


def create_cam_visualizer(model, device, cam_method='fullgrad', target_class=1):
    """
    Factory function to create a CAM visualizer.
    
    Args:
        model: The trained MIL model
        device: Computation device
        cam_method: CAM method name (default: 'fullgrad')
        target_class: Target class index (default: 1 for cancer)
        
    Returns:
        CAMVisualizer instance
    """
    return CAMVisualizer(model, device, cam_method, target_class) 