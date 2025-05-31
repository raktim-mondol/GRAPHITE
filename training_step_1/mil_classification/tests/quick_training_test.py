"""
Quick training test for the MIL model.
This script runs a short training session to verify that the model can train properly.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mil_quick_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MIL_Quick_Training")

def main():
    """Run a quick training test"""
    logger.info("Starting quick training test (2 epochs, 50 patches max)")
    
    # Import here to avoid loading everything when this module is imported
    import torch
    from src.models.mil_classifier import MILHistopathModel
    import subprocess
    
    # First verify that the model can be instantiated and do a forward pass
    try:
        model = MILHistopathModel(num_classes=2)
        x = torch.randn(2, 5, 3, 224, 224)
        patch_proj, patient_proj, logits, att_weights = model(x)
        logger.info("Model forward pass successful!")
    except Exception as e:
        logger.error(f"Model forward pass failed: {e}")
        return False
    
    # Run a quick training session
    try:
        # Import the train module
        from train import main as train_main
        
        # Set up arguments for quick training
        sys.argv = [
            'train.py',
            '--max_patches', '50',
            '--num_epochs', '2',
            '--batch_size', '4'
        ]
        
        # Run training
        train_main()
        logger.info("Quick training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Quick training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 