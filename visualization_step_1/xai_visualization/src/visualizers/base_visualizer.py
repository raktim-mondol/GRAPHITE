"""
Base visualizer class for XAI methods.

This module defines the abstract base class that all visualization methods should inherit from.
"""

from abc import ABC, abstractmethod
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image

from ..utils.metrics import calculate_metrics
from ..utils.logger import get_logger


class BaseVisualizer(ABC):
    """Abstract base class for all visualization methods."""
    
    def __init__(self, model, device, config):
        """
        Initialize the base visualizer.
        
        Args:
            model: The trained model to visualize
            device: Computation device (CPU/GPU)
            config: Configuration object
        """
        self.model = model
        self.device = device
        self.config = config
        self.logger = get_logger()
        
        # Set model to evaluation mode
        self.model.eval()
    
    @abstractmethod
    def generate_heatmap(self, wsi_path: str, output_path: str) -> np.ndarray:
        """
        Generate heatmap for a single WSI image.
        
        Args:
            wsi_path: Path to the WSI image
            output_path: Path to save the heatmap visualization
            
        Returns:
            Generated heatmap as numpy array
        """
        pass
    
    def evaluate_heatmap(self, heatmap: np.ndarray, mask_path: str) -> Tuple[List[Dict], Dict]:
        """
        Evaluate heatmap against ground truth mask.
        
        Args:
            heatmap: Generated heatmap
            mask_path: Path to ground truth mask
            
        Returns:
            Tuple of (results_list, best_result)
        """
        # Load and preprocess mask
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = mask > 0  # Convert to binary
        
        # Ensure heatmap and mask have the same dimensions
        if heatmap.shape != mask.shape:
            self.logger.warning("Reshaping mask due to size mismatch with heatmap")
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_img = mask_img.resize((heatmap.shape[1], heatmap.shape[0]), Image.NEAREST)
            mask = np.array(mask_img).astype(bool)
        
        # Evaluate at different thresholds
        thresholds = np.arange(0, 1.01, 0.1)
        results = []
        
        for threshold in thresholds:
            pred_mask = heatmap > threshold
            metrics = calculate_metrics(pred_mask, mask)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        # Find best threshold based on F1 score
        best_result = max(results, key=lambda x: x['f1_score'])
        
        self.logger.info(f"Best threshold: {best_result['threshold']:.2f}")
        self.logger.info(f"Best F1 score: {best_result['f1_score']:.4f}")
        self.logger.info(f"Precision: {best_result['precision']:.4f}")
        self.logger.info(f"Recall: {best_result['recall']:.4f}")
        self.logger.info(f"IoU: {best_result['iou']:.4f}")
        
        return results, best_result
    
    def process_single_image(self, wsi_path: str, mask_path: str, 
                           heatmap_path: str, csv_path: str) -> Dict:
        """
        Process a single WSI image and generate results.
        
        Args:
            wsi_path: Path to WSI image
            mask_path: Path to ground truth mask
            heatmap_path: Path to save heatmap visualization
            csv_path: Path to save evaluation results
            
        Returns:
            Dictionary containing best results
        """
        self.logger.info(f"Processing {Path(wsi_path).name}...")
        
        # Generate heatmap
        heatmap = self.generate_heatmap(wsi_path, heatmap_path)
        
        # Evaluate against ground truth
        results, best_result = self.evaluate_heatmap(heatmap, mask_path)
        
        # Save results to CSV
        df = pd.DataFrame(results)
        columns = ['threshold'] + [col for col in df.columns if col != 'threshold']
        df = df[columns]
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Results saved to {csv_path}")
        self.logger.info(f"Best threshold: {best_result['threshold']:.2f}")
        self.logger.info(f"Best F1 score: {best_result['f1_score']:.4f}")
        
        return best_result
    
    def process_images(self, wsi_folder: str, mask_folder: str, 
                      heatmap_folder: str, results_folder: str):
        """
        Process all images in the specified folders.
        
        Args:
            wsi_folder: Folder containing WSI images
            mask_folder: Folder containing ground truth masks
            heatmap_folder: Folder to save heatmap visualizations
            results_folder: Folder to save evaluation results
        """
        # Ensure output folders exist
        os.makedirs(heatmap_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)
        
        # Get all image files in the WSI folder
        wsi_files = [f for f in os.listdir(wsi_folder) if f.endswith('.png')]
        
        if not wsi_files:
            self.logger.warning(f"No PNG files found in {wsi_folder}")
            return
        
        self.logger.info(f"Found {len(wsi_files)} images to process")
        
        all_results = []
        
        for wsi_file in wsi_files:
            # Construct full paths
            wsi_path = os.path.join(wsi_folder, wsi_file)
            mask_file = wsi_file  # Assuming mask has the same filename
            mask_path = os.path.join(mask_folder, mask_file)
            
            # Skip if mask doesn't exist
            if not os.path.exists(mask_path):
                self.logger.warning(f"Mask not found for {wsi_file}, skipping.")
                continue
            
            # Construct output paths
            base_name = Path(wsi_file).stem
            method_name = self.__class__.__name__.lower().replace('visualizer', '')
            heatmap_path = os.path.join(heatmap_folder, f"{base_name}_{method_name}_heatmap.png")
            csv_path = os.path.join(results_folder, f"{base_name}_{method_name}_results.csv")
            
            try:
                # Process single image
                best_result = self.process_single_image(wsi_path, mask_path, heatmap_path, csv_path)
                best_result['image_name'] = wsi_file
                best_result['method'] = method_name
                all_results.append(best_result)
                
            except Exception as e:
                self.logger.error(f"Error processing {wsi_file}: {str(e)}")
                continue
            
            self.logger.info("--------------------")
        
        # Save summary results
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_path = os.path.join(results_folder, f"{method_name}_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Summary results saved to {summary_path}")
            
            # Print overall statistics
            avg_f1 = summary_df['f1_score'].mean()
            avg_iou = summary_df['iou'].mean()
            self.logger.info(f"Average F1 score: {avg_f1:.4f}")
            self.logger.info(f"Average IoU: {avg_iou:.4f}")
        
        self.logger.info("Processing completed!") 