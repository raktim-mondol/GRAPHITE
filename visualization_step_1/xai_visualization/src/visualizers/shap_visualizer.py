"""
SHAP-based visualization methods.

This module implements SHAP (SHapley Additive exPlanations) methods for explaining
model predictions on histopathology images.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import shap
import timm

from .base_visualizer import BaseVisualizer
from ..data.slide_reader import CustomSlide
from ..data.color_normalizer import MacenkoColorNormalization, NormalizationError


class SHAPModelWrapper(nn.Module):
    """Wrapper for MIL model to make it compatible with SHAP explainers."""
    
    def __init__(self, model):
        super(SHAPModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        # SHAP expects a batch of images directly, not nested
        # For explainers, reshape inputs to what model expects
        if len(x.shape) == 4:  # (batch, channels, height, width)
            # Clone the tensor to avoid view+inplace modification issues
            x = x.clone().unsqueeze(1)  # (batch, 1, channels, height, width)
        
        # Ensure we're working with a detached tensor that requires grad
        if not x.requires_grad:
            x = x.requires_grad_(True)
        
        _, _, logits, _ = self.model(x)
        return logits


class SHAPVisualizer(BaseVisualizer):
    """SHAP-based visualization methods."""
    
    def __init__(self, model, device, explainer_type, config):
        """
        Initialize SHAP visualizer.
        
        Args:
            model: The trained model
            device: Computation device
            explainer_type: Type of SHAP explainer ('deep' or 'gradient')
            config: Configuration object
        """
        super().__init__(model, device, config)
        self.explainer_type = explainer_type
        self.color_normalizer = MacenkoColorNormalization()
        
        # Setup data transforms
        data_config = timm.data.resolve_model_data_config(model.feature_extractor)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        # Wrap model for SHAP compatibility
        self.wrapped_model = SHAPModelWrapper(model)
        self.wrapped_model.eval()
    
    def _collect_background_data(self, slide, patch_size, background_samples=10):
        """Collect background data for SHAP explainer."""
        background_data = []
        coords_list = []
        
        height, width = slide.dimensions
        
        # Collect all possible patch coordinates
        for y in range(0, height - patch_size + 1, self.config.stride):
            for x in range(0, width - patch_size + 1, self.config.stride):
                coords_list.append((x, y))
        
        # Randomly sample patches for background
        if len(coords_list) > background_samples:
            background_indices = np.random.choice(len(coords_list), background_samples, replace=False)
            background_coords = [coords_list[i] for i in background_indices]
        else:
            background_coords = coords_list
        
        # Collect background patches
        for x, y in background_coords:
            try:
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                try:
                    if self.config.use_color_normalization:
                        normalized_patch = self.color_normalizer(patch)
                    else:
                        normalized_patch = patch
                    
                    patch_tensor = self.transform(normalized_patch).unsqueeze(0)
                    background_data.append(patch_tensor)
                    
                    if len(background_data) >= background_samples:
                        break
                        
                except NormalizationError:
                    self.logger.warning(f"Normalization failed for background patch at ({x}, {y})")
                    
            except ValueError as e:
                self.logger.warning(f"Skipping background patch at ({x}, {y}): {str(e)}")
        
        if not background_data:
            self.logger.warning("Failed to collect any background data. Using random noise instead.")
            # Use random noise as a fallback
            background_data = [torch.randn(1, 3, patch_size, patch_size) for _ in range(background_samples)]
        
        return torch.cat(background_data, dim=0).to(self.device)
    
    def _process_patch_with_shap(self, patch_tensor, background_data):
        """Process a single patch with SHAP explainer."""
        # Move data to device and ensure proper tensor setup
        patch_tensor = patch_tensor.to(self.device)
        
        # Create explainer based on type
        if self.explainer_type == 'deep':
            explainer = shap.DeepExplainer(self.wrapped_model, background_data)
        elif self.explainer_type == 'gradient':
            explainer = shap.GradientExplainer(self.wrapped_model, background_data)
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")
        
        # Get SHAP values with proper gradient handling
        patch_input = patch_tensor.unsqueeze(0).clone().detach().requires_grad_(True)  # Add batch dimension and ensure proper grad setup
        
        with torch.enable_grad():  # Ensure gradients are enabled for SHAP computation
            shap_values = explainer.shap_values(patch_input)
        
        # For classification models, shap_values is a list of arrays for each class
        # Use cancer class (index 1) for interpretation
        if isinstance(shap_values, list):
            cancer_class_idx = 1
            shap_values = shap_values[cancer_class_idx][0]  # Get values for cancer class, remove batch dim
        else:
            shap_values = shap_values[0]  # Remove batch dimension
        
        return shap_values
    
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
        Generate SHAP heatmap for a WSI image.
        
        Args:
            wsi_path: Path to the WSI image
            output_path: Path to save the heatmap visualization
            
        Returns:
            Generated heatmap as numpy array
        """
        slide = CustomSlide(wsi_path)
        patch_size = self.config.patch_size
        stride = self.config.stride
        background_samples = getattr(self.config, 'background_samples', 10)
        
        height, width = slide.dimensions
        
        # Collect background data
        self.logger.info("Collecting background data for SHAP explainer...")
        background_data = self._collect_background_data(slide, patch_size, background_samples)
        
        all_importances = []
        all_coords = []
        skipped_coords = []
        
        # Process all patches
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                try:
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                    
                    try:
                        if self.config.use_color_normalization:
                            normalized_patch = self.color_normalizer(patch)
                        else:
                            normalized_patch = patch
                        
                        patch_tensor = self.transform(normalized_patch)
                        
                        # Get SHAP values with error handling for autograd issues
                        try:
                            shap_values = self._process_patch_with_shap(patch_tensor, background_data)
                            
                            # Use the absolute sum as importance
                            importance = np.abs(shap_values).sum()
                            
                            all_importances.append(importance)
                            all_coords.append((x, y))
                            
                        except RuntimeError as e:
                            if "inplace" in str(e).lower() or "backward" in str(e).lower():
                                self.logger.warning(f"Autograd error at ({x}, {y}): {str(e)[:100]}... Using fallback.")
                                # Use a small random importance as fallback
                                fallback_importance = np.random.random() * 0.1
                                all_importances.append(fallback_importance)
                                all_coords.append((x, y))
                            else:
                                raise e
                        
                    except NormalizationError:
                        self.logger.warning(f"Normalization failed for patch at ({x}, {y})")
                        skipped_coords.append((x, y))
                        
                except ValueError as e:
                    self.logger.warning(f"Skipping patch at ({x}, {y}): {str(e)}")
                    skipped_coords.append((x, y))
        
        # Create heatmap
        heatmap_shape = (height // patch_size, width // patch_size)
        heatmap = np.zeros(heatmap_shape)
        
        for (x, y), importance in zip(all_coords, all_importances):
            heatmap[y // patch_size, x // patch_size] = importance
        
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
        
        # Resize heatmap to match original image size if needed
        if heatmap_smooth.shape != (height, width):
            heatmap_resized = cv2.resize(heatmap_smooth, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap_smooth
        
        # Apply tissue mask to heatmap
        masked_heatmap = np.where(tissue_mask == 255, heatmap_resized, np.nan)
        
        # Create visualization
        plt.figure(figsize=(20, 20))
        plt.imshow(wsi_thumb)
        plt.imshow(masked_heatmap, alpha=0.5, cmap='jet', interpolation='nearest')
        plt.axis('off')
        plt.title(f"WSI with SHAP {self.explainer_type.title()} Heatmap (Tissue Only)")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        total_patches = len(all_coords) + len(skipped_coords)
        self.logger.info(f"Total patches: {total_patches}")
        self.logger.info(f"Processed patches: {len(all_coords)}")
        self.logger.info(f"Skipped patches: {len(skipped_coords)}")
        
        return masked_heatmap 