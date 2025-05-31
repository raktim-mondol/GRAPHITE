#!/usr/bin/env python3
"""
GRAPHITE Demo: Quick demonstration of self-supervised training
This script shows how to use GRAPHITE for graph-based self-supervised learning
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.imports import *

def demo_training():
    """Demonstrate training setup"""
    print("="*60)
    print("GRAPHITE Training Demo")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(78)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    from models.hiergat import HierGATSSL
    model = HierGATSSL(
        input_dim=128,
        hidden_dim=128,
        num_gat_layers=3,
        num_heads=4,
        num_levels=3,
        dropout=0.1
    )
    
    # Create loss function
    from training.losses import HierarchicalInfoMaxLoss
    criterion = HierarchicalInfoMaxLoss(
        temperature=0.07,
        alpha=0.5,
        beta=0.5,
        tau=0.1
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"✓ Loss function: HierarchicalInfoMaxLoss")
    print(f"✓ Optimizer: Adam (lr=0.001)")
    
    print("\nTo start training with your data:")
    print("python train.py --data_dir /path/to/your/images")
    
    return model, criterion, optimizer

def demo_model_saving():
    """Demonstrate model saving setup"""
    print("\n" + "="*60)
    print("GRAPHITE Model Saving")
    print("="*60)
    
    print("✓ Models are automatically saved during training:")
    print("  - Best model: output/hiergat_ssl/best_model.pt")
    print("  - Regular checkpoints: output/hiergat_ssl/checkpoint_epoch_X.pt")
    print("  - Training logs: output/hiergat_ssl/training.log")
    print("  - Loss plots: output/hiergat_ssl/training_progress_epoch_X.png")
    
    print("\n✓ Model can be loaded later for:")
    print("  - Resume training with --resume flag")
    print("  - Transfer learning to new datasets")
    print("  - Feature extraction from trained representations")

def demo_data_requirements():
    """Show data requirements"""
    print("\n" + "="*60)
    print("Data Requirements")
    print("="*60)
    
    print("Your data should be organized as:")
    print("""
data/
├── train_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── test_images/
    ├── test1.png
    ├── test2.png
    └── ...
    """)
    
    print("Supported formats: PNG, JPG, JPEG, TIFF, TIF")
    print("Recommended image size: 224x224 pixels or larger")
    print("Minimum dataset size: 10+ images for demo, 100+ for real training")

def main():
    """Main demo function"""
    print("🔬 GRAPHITE: Graph-based Self-Supervised Learning for Histopathology")
    print("Demo script showing core functionality\n")
    
    # Demo training setup
    try:
        model, criterion, optimizer = demo_training()
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
        return False
    
    # Demo model saving
    try:
        demo_model_saving()
    except Exception as e:
        print(f"❌ Model saving demo failed: {e}")
        return False
    
    # Show data requirements
    demo_data_requirements()
    
    print("\n" + "="*60)
    print("🎉 Demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your histopathology images")
    print("2. Run self-supervised training: python train.py --data_dir /path/to/your/data")
    print("3. Monitor training progress in output/ directory")
    print("4. Use saved model for your downstream tasks")
    print("\n📊 Training outputs:")
    print("- Trained model saved in output/ directory")
    print("- Training progress plots (loss vs epoch)")
    print("- Training logs and statistics")
    print("\n🎯 Focus: Self-supervised learning without manual annotations")
    print("\nFor help: python train.py --help")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 