#!/usr/bin/env python3
"""
XAI Visualization Tool for Histopathology Images

This tool provides various explainability methods for analyzing histopathology images:
1. CAM-based methods (GradCAM, HiResCAM, ScoreCAM, etc.)
2. Model-agnostic methods (SHAP, LIME)
3. MIL attention-based visualization

Usage:
    python main.py --method gradcam --wsi_folder /path/to/wsi --mask_folder /path/to/masks --output_folder /path/to/output
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Import visualization modules
from src.visualizers.cam_visualizer import CAMVisualizer
from src.visualizers.shap_visualizer import SHAPVisualizer
from src.visualizers.lime_visualizer import LIMEVisualizer
from src.visualizers.attention_visualizer import AttentionVisualizer
from src.models.model_loader import load_model
from src.utils.config import Config
from src.utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XAI Visualization Tool for Histopathology Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Methods:
  CAM-based:
    - gradcam: Gradient-weighted Class Activation Mapping
    - hirescam: High Resolution Class Activation Mapping
    - scorecam: Score-weighted Class Activation Mapping
    - gradcampp: GradCAM++
    - ablationcam: Ablation CAM
    - xgradcam: XGradCAM
    - eigencam: EigenCAM
    - fullgrad: FullGrad
    
  Model-agnostic:
    - shap_deep: SHAP with Deep Explainer
    - shap_gradient: SHAP with Gradient Explainer
    - lime: LIME (Local Interpretable Model-agnostic Explanations)
    
  MIL attention:
    - attention: MIL attention-based visualization

Examples:
  python main.py --method gradcam --wsi_folder ./data/wsi --mask_folder ./data/masks --output_folder ./results
  python main.py --method shap_deep --wsi_folder ./data/wsi --mask_folder ./data/masks --output_folder ./results --model_path ./models/best_model.pth
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--method', 
        type=str, 
        required=True,
        choices=[
            'gradcam', 'hirescam', 'scorecam', 'gradcampp', 'ablationcam', 
            'xgradcam', 'eigencam', 'fullgrad',
            'shap_deep', 'shap_gradient', 'lime', 'attention'
        ],
        help='Visualization method to use'
    )
    
    parser.add_argument(
        '--wsi_folder', 
        type=str, 
        required=True,
        help='Path to folder containing WSI images'
    )
    
    parser.add_argument(
        '--mask_folder', 
        type=str, 
        required=True,
        help='Path to folder containing ground truth masks'
    )
    
    parser.add_argument(
        '--output_folder', 
        type=str, 
        required=True,
        help='Path to output folder for results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='./models/best_fine_tuned_model_for_resnet18_cancervsnormal_v4.pth',
        help='Path to the trained model'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='./config/default.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--patch_size', 
        type=int, 
        default=224,
        help='Size of patches to extract from WSI'
    )
    
    parser.add_argument(
        '--stride', 
        type=int, 
        default=224,
        help='Stride between patches'
    )
    
    parser.add_argument(
        '--target_class', 
        type=int, 
        default=1,
        help='Target class index for visualization (1 for cancer, 0 for normal)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for computation'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=78,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup computation device."""
    if device_arg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device

def validate_paths(args):
    """Validate input and output paths."""
    if not os.path.exists(args.wsi_folder):
        raise FileNotFoundError(f"WSI folder not found: {args.wsi_folder}")
    
    if not os.path.exists(args.mask_folder):
        raise FileNotFoundError(f"Mask folder not found: {args.mask_folder}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Create output directories
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'heatmaps'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'results'), exist_ok=True)

def get_visualizer(method, model, device, config):
    """Get the appropriate visualizer based on the method."""
    cam_methods = ['gradcam', 'hirescam', 'scorecam', 'gradcampp', 'ablationcam', 'xgradcam', 'eigencam', 'fullgrad']
    
    if method in cam_methods:
        return CAMVisualizer(model, device, method, config)
    elif method in ['shap_deep', 'shap_gradient']:
        explainer_type = method.split('_')[1]  # 'deep' or 'gradient'
        return SHAPVisualizer(model, device, explainer_type, config)
    elif method == 'lime':
        return LIMEVisualizer(model, device, config)
    elif method == 'attention':
        return AttentionVisualizer(model, device, config)
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger(verbose=args.verbose)
    logger.info(f"Starting XAI visualization with method: {args.method}")
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Setup device
        device = setup_device(args.device)
        
        # Load configuration
        config = Config(args.config if os.path.exists(args.config) else None)
        config.update_from_args(args)
        
        # Set random seed
        config.set_seed(args.seed)
        
        # Load model
        logger.info("Loading model...")
        model = load_model(args.model_path, device, config)
        
        # Get visualizer
        logger.info(f"Initializing {args.method} visualizer...")
        visualizer = get_visualizer(args.method, model, device, config)
        
        # Process images
        logger.info("Starting image processing...")
        visualizer.process_images(
            wsi_folder=args.wsi_folder,
            mask_folder=args.mask_folder,
            heatmap_folder=os.path.join(args.output_folder, 'heatmaps'),
            results_folder=os.path.join(args.output_folder, 'results')
        )
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 