"""
Tests for the training functions.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.mil_classifier import MILHistopathModel
from src.training.train import train_epoch, validate

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create random patches and labels
        num_patches = 5
        patches = torch.randn(num_patches, 3, 224, 224)
        label = torch.randint(0, 2, (1,)).item()
        # Add a mock patient_id to match data loader format
        patient_id = f"patient_{idx}"
        return patches, label, patient_id

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MILHistopathModel(num_classes=2).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create mock dataset and dataloader
        self.train_dataset = MockDataset(10)
        self.val_dataset = MockDataset(5)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=2)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=2)
    
    def test_train_epoch(self):
        """Test train_epoch function"""
        loss, metrics = train_epoch(
            model=self.model,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device
        )
        
        self.assertIsNotNone(loss)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc', metrics)
    
    def test_validate(self):
        """Test validate function"""
        loss, metrics = validate(
            model=self.model,
            val_loader=self.val_loader,
            criterion=self.criterion,
            device=self.device
        )
        
        self.assertIsNotNone(loss)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc', metrics)

if __name__ == "__main__":
    unittest.main()
