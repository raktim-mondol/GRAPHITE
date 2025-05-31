"""
Tests for the MIL classifier model.
"""

import unittest
import torch
from src.models.mil_classifier import MILHistopathModel

class TestMILModel(unittest.TestCase):
    def setUp(self):
        self.model = MILHistopathModel(num_classes=2)
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsInstance(self.model, MILHistopathModel)
        
    def test_model_forward_pass(self):
        """Test forward pass with dummy input"""
        batch_size = 2
        num_patches = 5
        patch_size = (3, 224, 224)
        
        # Create dummy input
        x = torch.randn(batch_size, num_patches, *patch_size)
        
        # Run forward pass
        patch_proj, patient_proj, logits, attention_weights = self.model(x)
        
        # Check output shapes
        self.assertEqual(patch_proj.shape, (batch_size, num_patches, 128))
        self.assertEqual(patient_proj.shape, (batch_size, 256))
        self.assertEqual(logits.shape, (batch_size,))
        self.assertEqual(attention_weights.shape, (batch_size, num_patches))

if __name__ == "__main__":
    unittest.main()
