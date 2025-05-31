"""
CAM-based visualization methods.

This module implements various Class Activation Mapping (CAM) methods for explaining
model predictions on histopathology images.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import timm

# Import pytorch-grad-cam methods
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, 
    AblationCAM, XGradCAM, EigenCAM, FullGrad
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .base_visualizer import BaseVisualizer
from ..data.slide_reader import CustomSlide
from ..data.color_normalizer import MacenkoColorNormalization, NormalizationError


class ModelWrapper(nn.Module):
    """Wrapper for MIL model to make it compatible with CAM methods."""
    
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        # CAM methods expect direct input, add batch dimension for MIL model
        x = x.unsqueeze(1)
        _, _, logits, _ = self.model(x)
        return logits


class CAMVisualizer(BaseVisualizer):
    """CAM-based visualization methods."""
    
    def __init__(self, model, device, cam_method, config):
        """
        Initialize CAM visualizer.
        
        Args:
            model: The trained model
            device: Computation device
            cam_method: CAM method name ('gradcam', 'hirescam', etc.)
            config: Configuration object
        """
        super().__init__(model, device, config)
        self.cam_method = cam_method
        self.color_normalizer = MacenkoColorNormalization()
        
        # Setup data transforms
        data_config = timm.data.resolve_model_data_config(model.feature_extractor)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        # Wrap model for CAM compatibility
        self.wrapped_model = ModelWrapper(model)
        
        # Get target layer for CAM
        self.target_layer = self._get_target_layer()
        
        # Initialize CAM method
        self.cam = self._get_cam_method()
    
    def _get_target_layer(self):
        """Get the target layer for CAM visualization."""
        # For ResNet-based models, use the last layer of layer4
        if hasattr(self.model.feature_extractor, 'layer4'):
            return self.model.feature_extractor.layer4[-1]
        elif hasattr(self.model.feature_extractor, 'stages'):
            # For other architectures like EfficientNet
            return self.model.feature_extractor.stages[-1]
        else:
            # Fallback: try to find the last convolutional layer
            layers = []
            for module in self.model.feature_extractor.modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    layers.append(module)
            if layers:
                return layers[-1]
            else:
                raise ValueError("Could not find suitable target layer for CAM")
    
    def _get_cam_method(self):
        """Initialize the appropriate CAM method."""
        cam_methods = {
            'gradcam': GradCAM,
            'hirescam': HiResCAM,
            'scorecam': ScoreCAM,
            'gradcampp': GradCAMPlusPlus,
            'ablationcam': AblationCAM,
            'xgradcam': XGradCAM,
            'eigencam': EigenCAM,
            'fullgrad': FullGrad
        }
        
        if self.cam_method not in cam_methods:
            raise ValueError(f"Unknown CAM method: {self.cam_method}")
        
        cam_class = cam_methods[self.cam_method]
        return cam_class(model=self.wrapped_model, target_layers=[self.target_layer])
    
    def _create_tissue_mask(self, wsi_image):
        """Create tissue mask to focus on tissue regions."""
        # Convert to BGR (OpenCV format)
        wsi_bgr = cv2.cvtColor(wsi_image, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(wsi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply binary threshold
        _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.logger.warning("No contours found. Using full image.")
            return np.ones_like(gray)
        
        # Create mask from largest contour (assumed to be tissue)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        return mask
    
    def generate_heatmap(self, wsi_path: str, output_path: str) -> np.ndarray:
        """
        Generate CAM heatmap for a WSI image.
        
        Args:
            wsi_path: Path to the WSI image
            output_path: Path to save the heatmap visualization
            
        Returns:
            Generated heatmap as numpy array
        """
        slide = CustomSlide(wsi_path)
        patch_size = self.config.patch_size
        stride = self.config.stride
        target_class_idx = self.config.target_class
        
        all_cams = []
        all_coords = []
        skipped_coords = []
        
        height, width = slide.dimensions
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                try:
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                    
                    try:
                        # Apply color normalization if enabled
                        if self.config.use_color_normalization:
                            normalized_patch = self.color_normalizer(patch)
                        else:
                            normalized_patch = patch
                        
                        # Transform patch
                        patch_tensor = self.transform(normalized_patch).unsqueeze(0).to(self.device)
                        
                        # Generate CAM
                        grayscale_cam = self.cam(
                            input_tensor=patch_tensor, 
                            targets=[ClassifierOutputTarget(target_class_idx)]
                        )
                        
                        all_cams.append(grayscale_cam[0])
                        all_coords.append((x, y))
                        
                    except NormalizationError:
                        self.logger.warning(f"Normalization failed for patch at ({x}, {y})")
                        skipped_coords.append((x, y))
                        
                except ValueError as e:
                    self.logger.warning(f"Skipping patch at ({x}, {y}): {str(e)}")
                    skipped_coords.append((x, y))
        
        # Create heatmap from CAM values
        heatmap_shape = (height // patch_size, width // patch_size)
        heatmap = np.zeros(heatmap_shape)
        
        for (x, y), cam in zip(all_coords, all_cams):
            heatmap[y // patch_size, x // patch_size] = np.mean(cam)
        
        # Normalize heatmap
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Upsample and smooth heatmap
        heatmap_upsampled = np.repeat(np.repeat(heatmap, patch_size, axis=0), patch_size, axis=1)
        heatmap_smooth = gaussian_filter(heatmap_upsampled, sigma=patch_size//4)
        heatmap_smooth = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min())
        
        # Get WSI image for visualization
        wsi_thumb = slide.image.compute().squeeze(0)
        
        # Create tissue mask
        tissue_mask = self._create_tissue_mask(wsi_thumb)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap_smooth, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Apply tissue mask to heatmap
        masked_heatmap = np.where(tissue_mask == 255, heatmap_resized, np.nan)
        
        # Create visualization
        plt.figure(figsize=(20, 20))
        plt.imshow(wsi_thumb)
        plt.imshow(masked_heatmap, alpha=0.5, cmap='jet', interpolation='nearest')
        plt.axis('off')
        plt.title(f"WSI with {self.cam_method.upper()} Heatmap (Tissue Only)")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        total_patches = len(all_coords) + len(skipped_coords)
        self.logger.info(f"Total patches: {total_patches}")
        self.logger.info(f"Processed patches: {len(all_coords)}")
        self.logger.info(f"Skipped patches: {len(skipped_coords)}")
        
        return masked_heatmap 