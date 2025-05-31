"""
LIME-based visualization methods.

This module implements LIME (Local Interpretable Model-agnostic Explanations) 
for explaining model predictions on histopathology images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
import timm
from lime import lime_image

from .base_visualizer import BaseVisualizer
from ..data.slide_reader import CustomSlide
from ..data.color_normalizer import MacenkoColorNormalization, NormalizationError


class LimeModelWrapper:
    """Wrapper for the MIL model to make it compatible with LIME explainer."""
    
    def __init__(self, model, device, transform):
        self.model = model
        self.device = device
        self.transform = transform
        self.model.eval()
        
    def predict(self, images):
        """
        Predict function for LIME.
        
        Args:
            images: Numpy array of images with shape (n_samples, height, width, channels)
            
        Returns:
            Prediction probabilities
        """
        # LIME uses numpy arrays with shape (n_samples, height, width, channels)
        # Need to convert to PyTorch's (n_samples, channels, height, width) format
        batch = []
        for img in images:
            # Convert from (H,W,C) to PIL Image and then apply transforms
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_tensor = self.transform(img_pil)
            batch.append(img_tensor)
        
        batch_tensor = torch.stack(batch, dim=0).to(self.device)
        
        # Predict
        with torch.no_grad():
            # Add an extra dimension to simulate batch of patches
            batch_tensor = batch_tensor.unsqueeze(1)
            _, _, logits, _ = self.model(batch_tensor)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        return probs


class LIMEVisualizer(BaseVisualizer):
    """LIME-based visualization methods."""
    
    def __init__(self, model, device, config):
        """
        Initialize LIME visualizer.
        
        Args:
            model: The trained model
            device: Computation device
            config: Configuration object
        """
        super().__init__(model, device, config)
        self.color_normalizer = MacenkoColorNormalization()
        
        # Setup data transforms
        data_config = timm.data.resolve_model_data_config(model.feature_extractor)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
        
        # Create model wrapper for LIME
        self.lime_wrapper = LimeModelWrapper(model, device, self.transform)
    
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
        Generate LIME heatmap for a WSI image.
        
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
        
        # LIME parameters
        num_samples = getattr(self.config, 'lime_num_samples', 100)
        batch_size = getattr(self.config, 'lime_batch_size', 10)
        num_features = getattr(self.config, 'lime_num_features', 10)
        
        all_explanations = []
        all_coords = []
        skipped_coords = []
        
        height, width = slide.dimensions
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                try:
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                    
                    try:
                        # Use the original patch for LIME (no color normalization for LIME input)
                        patch_np = np.array(patch)
                        
                        # Get LIME explanation
                        explanation = self.explainer.explain_instance(
                            patch_np,
                            self.lime_wrapper.predict,
                            top_labels=2,  # Explain both classes
                            hide_color=0,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            segmentation_fn=None  # Use default quickshift segmentation
                        )
                        
                        # Get the explanation for the target class
                        # Higher positive values mean features supporting the class
                        exp_map = explanation.get_image_and_mask(
                            target_class_idx,
                            positive_only=True,
                            num_features=num_features,
                            hide_rest=False
                        )[1]  # Get just the mask
                        
                        # Compute importance as the mean of exp_map
                        importance = np.mean(exp_map)
                        
                        all_explanations.append(importance)
                        all_coords.append((x, y))
                        
                    except NormalizationError:
                        self.logger.warning(f"Normalization failed for patch at ({x}, {y})")
                        skipped_coords.append((x, y))
                        
                except ValueError as e:
                    self.logger.warning(f"Skipping patch at ({x}, {y}): {str(e)}")
                    skipped_coords.append((x, y))
        
        # Create heatmap
        heatmap_shape = (height // patch_size, width // patch_size)
        heatmap = np.zeros(heatmap_shape)
        
        for (x, y), exp_val in zip(all_coords, all_explanations):
            heatmap[y // patch_size, x // patch_size] = exp_val
        
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
        plt.title("WSI with LIME Heatmap (Tissue Only)")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        total_patches = len(all_coords) + len(skipped_coords)
        self.logger.info(f"Total patches: {total_patches}")
        self.logger.info(f"Processed patches: {len(all_coords)}")
        self.logger.info(f"Skipped patches: {len(skipped_coords)}")
        
        return masked_heatmap 