#!/usr/bin/env python3
"""
GRAPHITE Training: Graph-based Self-Supervised Learning for Histopathology
A hierarchical Graph Attention Network for multi-scale self-supervised learning

Usage:
    python train.py --config config/config.yaml --data_dir /path/to/data
    python train.py --data_dir /path/to/data --epochs 100 --batch_size 4
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.imports import *
from utils.config_manager import ConfigManager
from data.dataset import HierGATSSLDataset
from models.hiergat import HierGATSSL
from training.trainer import HierGATSSLTrainer
from training.losses import HierarchicalInfoMaxLoss
from torch_geometric.loader import DataLoader as PyGDataLoader

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GRAPHITE: Hierarchical GAT SSL Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to directory containing training images')
    
    # Optional configuration
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='output/hiergat_ssl',
                       help='Directory to save trained models and training plots')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for GAT layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_gat_layers', type=int, default=3,
                       help='Number of GAT layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Loss parameters
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for InfoMax loss')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for InfoMax loss')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Weight for Scale-wise loss')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='Temperature for Scale-wise loss')
    
    # Training options
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=78,
                       help='Random seed for reproducibility')
    
    # System options
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def setup_device(device_arg):
    """Setup and return the appropriate device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    return device

def worker_init_fn(worker_id):
    """Initialize worker with proper random seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GRAPHITE: Self-Supervised Training")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60 + "\n")
    
    # Create dataset
    print("Creating dataset...")
    dataset = HierGATSSLDataset(image_dir=args.data_dir)
    print(f"Dataset size: {len(dataset)} images")
    
    # Create data loader
    train_loader = PyGDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        follow_batch=['x', 'edge_index'],
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create model
    print("Initializing model...")
    model = HierGATSSL(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        num_levels=3,  # Fixed for hierarchical levels
        dropout=args.dropout
    )
    
    # Create loss function
    criterion = HierarchicalInfoMaxLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Handle resume training
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Create trainer
    trainer = HierGATSSLTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.output_dir,
        num_epochs=args.epochs,
        patience=args.patience,
        start_epoch=start_epoch
    )
    
    # Start training
    print("Starting training...")
    try:
        trainer.train(train_loader)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(
            trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0,
            {'total_loss': float('inf')},
            is_best=False,
            filename='interrupted_checkpoint.pt'
        )
        print("Checkpoint saved before exit")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 