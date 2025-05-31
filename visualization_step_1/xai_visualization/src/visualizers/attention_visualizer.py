"""
Attention-based visualization methods.

This module implements attention-based visualization for Multiple Instance Learning (MIL)
models on histopathology images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import timm

from .base_visualizer import BaseVisualizer
from ..data.slide_reader import CustomSlide
from ..data.color_normalizer import MacenkoColorNormalization, NormalizationError


class AttentionVisualizer(BaseVisualizer):
    """Attention-based visualization for MIL models."""
    
    def __init__(self, model, device, config):
        """
        Initialize attention visualizer.
        
        Args:
            model: The trained MIL model with attention mechanism
            device: Computation device
            config: Configuration object
        """
        super().__init__(model, device, config)
        self.color_normalizer = MacenkoColorNormalization()
        
        # Setup data transforms
        data_config = timm.data.resolve_model_data_config(model.feature_extractor)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
    
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
        
        # Create mask from all contours (tissue regions)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        
        return mask
    
    def _process_wsi_and_get_attention(self, wsi_path):
        """Process WSI and extract attention weights."""
        slide = CustomSlide(wsi_path)
        patch_size = self.config.patch_size
        stride = self.config.stride
        
        all_patches = []
        all_coords = []
        skipped_coords = []
        
        height, width = slide.dimensions
        
        # Extract patches
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
                        
                        patch_tensor = self.transform(normalized_patch).unsqueeze(0)
                        all_patches.append(patch_tensor)
                        all_coords.append((x, y))
                        
                    except NormalizationError:
                        self.logger.warning(f"Normalization failed for patch at ({x}, {y})")
                        skipped_coords.append((x, y))
                        
                except ValueError as e:
                    self.logger.warning(f"Skipping patch at ({x}, {y}): {str(e)}")
                    skipped_coords.append((x, y))
        
        if not all_patches:
            raise ValueError("No valid patches were extracted from the image.")
        
        # Concatenate all patches and process through model
        all_patches = torch.cat(all_patches, dim=0).to(self.device)
        
        with torch.no_grad():
            # Get attention weights from the model
            _, _, _, attention = self.model(all_patches.unsqueeze(0))
        
        attention_scores = attention.squeeze().cpu().numpy()
        
        # Normalize attention scores
        if attention_scores.max() > attention_scores.min():
            attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
        
        return attention_scores, all_coords, skipped_coords, (width, height)
    
    def generate_heatmap(self, wsi_path: str, output_path: str) -> np.ndarray:
        """
        Generate attention heatmap for a WSI image.
        
        Args:
            wsi_path: Path to the WSI image
            output_path: Path to save the heatmap visualization
            
        Returns:
            Generated heatmap as numpy array
        """
        # Get attention scores
        attention_scores, all_coords, skipped_coords, wsi_dimensions = self._process_wsi_and_get_attention(wsi_path)
        
        width, height = wsi_dimensions
        patch_size = self.config.patch_size
        
        # Create initial heatmap
        heatmap = np.zeros((height, width))
        for (x, y), score in zip(all_coords, attention_scores):
            heatmap[y:y+patch_size, x:x+patch_size] = score
        
        # Smooth heatmap
        heatmap_smooth = gaussian_filter(heatmap, sigma=patch_size//4)
        
        # Normalize smoothed heatmap
        if heatmap_smooth.max() > heatmap_smooth.min():
            heatmap_smooth = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min())
        
        # Load WSI for visualization
        slide = CustomSlide(wsi_path)
        wsi_thumb = slide.image.compute().squeeze(0)
        
        # Create tissue mask
        tissue_mask = self._create_tissue_mask(wsi_thumb)
        
        # Apply tissue mask to heatmap
        masked_heatmap = np.where(tissue_mask > 0, heatmap_smooth, 0)
        
        # Create visualization
        plt.figure(figsize=(20, 20))
        plt.imshow(wsi_thumb)
        
        # Create a colormap for the heatmap
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(masked_heatmap)
        
        # Set alpha channel based on tissue presence
        heatmap_colored[..., 3] = np.where(masked_heatmap > 0, 0.5, 0)  # 50% opacity where tissue is present
        
        plt.imshow(heatmap_colored)
        plt.axis('off')
        plt.title("WSI with Attention Heatmap (Tissue Only)")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        total_patches = len(all_coords) + len(skipped_coords)
        self.logger.info(f"Total patches: {total_patches}")
        self.logger.info(f"Processed patches: {len(all_coords)}")
        self.logger.info(f"Skipped patches: {len(skipped_coords)}")
        
        return masked_heatmap 