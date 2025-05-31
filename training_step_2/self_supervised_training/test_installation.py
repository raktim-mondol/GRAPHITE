#!/usr/bin/env python3
"""
GRAPHITE Installation Test
Test script to verify that all dependencies are properly installed
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Geometric: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import timm
        print(f"✓ TIMM: {timm.__version__}")
    except ImportError as e:
        print(f"✗ TIMM: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML: {yaml.__version__}")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    return True

def test_project_imports():
    """Test project-specific imports"""
    print("\nTesting project imports...")
    
    # Test basic utils first
    try:
        import utils.imports
        print("✓ Utils imports")
    except ImportError as e:
        print(f"✗ Utils imports: {e}")
        if "torch_scatter" in str(e):
            print("  ⚠ PyTorch Geometric dependencies missing")
            print("  💡 Try: conda install pyg -c pyg")
        return False
    
    # Test PyTorch Geometric specifically
    try:
        import torch_geometric
        print("✓ PyTorch Geometric")
    except ImportError as e:
        print(f"✗ PyTorch Geometric: {e}")
        print("  💡 Installation help:")
        print("     conda install pyg -c pyg")
        print("     OR pip install torch-geometric")
        return False
    
    # Test model imports
    try:
        from models.hiergat import HierGATSSL
        print("✓ HierGAT model")
    except ImportError as e:
        print(f"✗ HierGAT model: {e}")
        return False
    
    try:
        from training.losses import HierarchicalInfoMaxLoss
        print("✓ Loss functions")
    except ImportError as e:
        print(f"✗ Loss functions: {e}")
        return False
    
    try:
        from training.trainer import HierGATSSLTrainer
        print("✓ Trainer")
    except ImportError as e:
        print(f"✗ Trainer: {e}")
        return False
    
    try:
        from data.dataset import HierGATSSLDataset
        print("✓ Dataset")
    except ImportError as e:
        print(f"✗ Dataset: {e}")
        return False
    
    return True

def test_device():
    """Test device availability"""
    print("\nTesting device availability...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    print(f"✓ PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    return True

def test_model_creation():
    """Test basic model creation"""
    print("\nTesting model creation...")
    
    try:
        from models.hiergat import HierGATSSL
        from training.losses import HierarchicalInfoMaxLoss
        
        # Create model
        model = HierGATSSL(
            input_dim=128,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            num_levels=3,
            dropout=0.1
        )
        print("✓ Model creation successful")
        
        # Create loss function
        criterion = HierarchicalInfoMaxLoss(
            temperature=0.07,
            alpha=0.5,
            beta=0.5,
            tau=0.1
        )
        print("✓ Loss function creation successful")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("GRAPHITE Installation Test")
    print("="*60)
    
    tests = [
        ("Dependencies", test_imports),
        ("Project Imports", test_project_imports),
        ("Device", test_device),
        ("Model Creation", test_model_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! GRAPHITE is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your data in the required format")
        print("2. Run training: python train.py --data_dir /path/to/data")
        print("3. Use trained model for your downstream tasks")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Install PyTorch Geometric: see README for instructions")
        print("3. Check CUDA installation if using GPU")
    
    print("="*60)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 