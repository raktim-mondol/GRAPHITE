"""
Main training script for the MIL-based histopathology image classification model.
"""

import os
import logging
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from src.models.mil_classifier import MILHistopathModel
from src.data.datasets import setup_dataloaders
from src.training.train import train, plot_training_history, test_model_on_dataset
from src.utils.color_normalization import MacenkoColorNormalization, NormalizationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mil_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MIL_Training")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train MIL model for histopathology classification")
    
    # Data parameters
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Root directory containing patient folders (defaults to PBS_JOBFS)')
    parser.add_argument('--cancer_labels_path', type=str, 
                        default='../../dataset/cancer.txt',
                        help='Path to cancer patient labels file')
    parser.add_argument('--normal_labels_path', type=str,
                        default='../../dataset/normal.txt',
                        help='Path to normal patient labels file')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--max_patches', type=int, default=100,
                        help='Maximum number of patches to use per patient')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Proportion of dataset to use for testing (default: 0.3 for 70-30 split)')
    parser.add_argument('--random_state', type=int, default=78,
                        help='Random seed for train/test split')
    parser.add_argument('--use_balanced_sampler', action='store_true',
                        help='Use balanced batch sampling for training')
    # Model parameters
    parser.add_argument('--model_name', type=str, default='hf-hub:1aurent/resnet18.tiatoolbox-kather100k',
                        help='Name of the backbone model (supported by timm)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='Feature dimension of the backbone model')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='Projection dimension for patch features')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--metrics_to_monitor', type=str, default='auc',
                        choices=['accuracy', 'f1', 'auc', 'loss'],
                        help='Metric to monitor for early stopping')
    
    # Color normalization
    parser.add_argument('--use_color_normalization', action='store_true',
                        help='Apply Macenko color normalization to input images')
    
    # Output parameters
    parser.add_argument('--model_save_path', type=str, default='./output/best_model.pth',
                        help='Path to save the best model')
    parser.add_argument('--history_plot_path', type=str, default='./output/training_history.png',
                        help='Path to save the training history plot')                        
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    random.seed(args.random_state)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup data directory from PBS_JOBFS environment variable
    if args.root_dir is None:
        if "PBS_JOBFS" in os.environ:
            TMPDIR = os.environ["PBS_JOBFS"]
            args.root_dir = os.path.join(TMPDIR, "tma_core_normalized_with_normal")
            logger.info(f"Using PBS_JOBFS data directory: {args.root_dir}")
        else:
            logger.error("PBS_JOBFS environment variable not found and no root_dir specified")
            exit(1)
    
    if not os.path.exists(args.root_dir):
        logger.error(f"Data directory does not exist: {args.root_dir}")
        exit(1)

    # Setup color normalization if requested
    color_normalizer = None
    if args.use_color_normalization:
        try:
            logger.info("Using Macenko color normalization")
            color_normalizer = MacenkoColorNormalization()
        except Exception as e:
            logger.error(f"Failed to initialize color normalization: {e}")
            logger.info("Continuing without color normalization")
    
    # Load cancer and normal patient IDs
    try:
        cancer = pd.read_csv(args.cancer_labels_path, index_col='patient_id').index.tolist()
        normal = pd.read_csv(args.normal_labels_path, index_col='patient_id').index.tolist()
          # Combine into a DataFrame
        all_data = pd.concat([
            pd.DataFrame({'label': 1}, index=cancer),
            pd.DataFrame({'label': 0}, index=normal)
        ])
        
        patient_ids = all_data.index.tolist()
        labels = all_data
        
        logger.info(f"Loaded {len(cancer)} cancer patients and {len(normal)} normal patients")
        
    except FileNotFoundError as e:
        logger.error(f"Error loading patient data: {e}")
        logger.info("Please update the data paths in the configuration")
        exit(1)
    
    # Setup data loaders
    train_loader, test_loader = setup_dataloaders(
        root_dir=args.root_dir,
        patient_ids=patient_ids,
        labels=labels,
        batch_size=args.batch_size,
        max_patches=args.max_patches,
        test_size=args.test_size,
        random_state=args.random_state,
        use_balanced_sampler=args.use_balanced_sampler,
        color_normalization=color_normalizer
    )
    
    # Create model
    model = MILHistopathModel(
        num_classes=args.num_classes,        feat_dim=args.feat_dim,
        proj_dim=args.proj_dim,
        model_name=args.model_name
    )
    # Setup loss function, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max' if args.metrics_to_monitor != 'loss' else 'min',
        factor=0.5,
        patience=5
    )
    
    # Train model
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set for validation during training
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping_patience,
        model_save_path=args.model_save_path,
        metrics_to_monitor=args.metrics_to_monitor
    )
    
    # Plot training history
    plot_training_history(history, save_path=args.history_plot_path)
    
    # Test model on test set
    test_metrics = test_model_on_dataset(model, test_loader, criterion, device)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
