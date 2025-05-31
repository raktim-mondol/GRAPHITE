#!/usr/bin/env python3
"""
Test script to debug dataset processing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.dataset import HierGATSSLDataset
import torch

def test_dataset():
    print("Testing dataset creation...")
    try:
        dataset = HierGATSSLDataset('dataset/core_images')
        print(f"Dataset created with {len(dataset)} images")
        
        print("Testing first sample...")
        sample = dataset[0]
        print("Sample processed successfully!")
        print(f"Sample type: {type(sample)}")
        
        if hasattr(sample, 'x'):
            print(f"Node features shape: {sample.x.shape}")
        else:
            print("No x attribute")
            
        if hasattr(sample, 'edge_index'):
            print(f"Edge index shape: {sample.edge_index.shape}")
        else:
            print("No edge_index attribute")
            
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dataset() 