#!/usr/bin/env python3
"""
Example usage of the XAI Visualization Tool.

This script demonstrates how to use the tool programmatically
instead of through the command line interface.
"""

import os
import sys
import torch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_loader import load_model
from utils.config import Config
from utils.logger import setup_logger
from visualizers import CAMVisualizer, SHAPVisualizer, LIMEVisualizer, AttentionVisualizer


def main():
    """Example usage of the XAI visualization tool."""
    
    # Setup logging
    logger = setup_logger(verbose=True)
    logger.info("Starting XAI visualization example")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = Config()
    config.set_seed(78)
    
    # Example paths (update these to your actual paths)
    model_path = "./models/best_fine_tuned_model_for_resnet18_cancervsnormal_v4.pth"
    wsi_folder = "./data/wsi"
    mask_folder = "./data/masks"
    output_folder = "./results"
    
    # Check if paths exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(wsi_folder):
        logger.error(f"WSI folder not found: {wsi_folder}")
        return
    
    # Load model
    logger.info("Loading model...")
    model = load_model(model_path, device, config)
    
    # Example 1: GradCAM visualization
    logger.info("Running GradCAM visualization...")
    gradcam_visualizer = CAMVisualizer(model, device, 'gradcam', config)
    gradcam_visualizer.process_images(
        wsi_folder=wsi_folder,
        mask_folder=mask_folder,
        heatmap_folder=os.path.join(output_folder, 'gradcam', 'heatmaps'),
        results_folder=os.path.join(output_folder, 'gradcam', 'results')
    )
    
    # Example 2: SHAP visualization
    logger.info("Running SHAP visualization...")
    shap_visualizer = SHAPVisualizer(model, device, 'deep', config)
    shap_visualizer.process_images(
        wsi_folder=wsi_folder,
        mask_folder=mask_folder,
        heatmap_folder=os.path.join(output_folder, 'shap', 'heatmaps'),
        results_folder=os.path.join(output_folder, 'shap', 'results')
    )
    
    # Example 3: LIME visualization
    logger.info("Running LIME visualization...")
    lime_visualizer = LIMEVisualizer(model, device, config)
    lime_visualizer.process_images(
        wsi_folder=wsi_folder,
        mask_folder=mask_folder,
        heatmap_folder=os.path.join(output_folder, 'lime', 'heatmaps'),
        results_folder=os.path.join(output_folder, 'lime', 'results')
    )
    
    # Example 4: Attention visualization
    logger.info("Running Attention visualization...")
    attention_visualizer = AttentionVisualizer(model, device, config)
    attention_visualizer.process_images(
        wsi_folder=wsi_folder,
        mask_folder=mask_folder,
        heatmap_folder=os.path.join(output_folder, 'attention', 'heatmaps'),
        results_folder=os.path.join(output_folder, 'attention', 'results')
    )
    
    logger.info("All visualizations completed!")


def single_image_example():
    """Example of processing a single image."""
    
    # Setup
    logger = setup_logger(verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    
    # Paths
    model_path = "./models/best_fine_tuned_model_for_resnet18_cancervsnormal_v4.pth"
    wsi_path = "./data/wsi/sample_image.png"
    mask_path = "./data/masks/sample_image.png"
    output_path = "./results/sample_gradcam.png"
    
    # Load model
    model = load_model(model_path, device, config)
    
    # Create visualizer
    visualizer = CAMVisualizer(model, device, 'gradcam', config)
    
    # Generate heatmap for single image
    heatmap = visualizer.generate_heatmap(wsi_path, output_path)
    
    # Evaluate against ground truth
    results, best_result = visualizer.evaluate_heatmap(heatmap, mask_path)
    
    logger.info(f"Best F1 score: {best_result['f1_score']:.4f}")
    logger.info(f"Best IoU: {best_result['iou']:.4f}")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run single image example
    # single_image_example() 