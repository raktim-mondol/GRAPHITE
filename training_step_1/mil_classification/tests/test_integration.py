"""
Integration tests for the MIL model pipeline.
These tests verify the end-to-end functionality of the model.
"""

import unittest
import os
import torch
import torch.nn as nn
import pandas as pd
import timm
from pathlib import Path

from src.models.mil_classifier import MILHistopathModel
from src.data.datasets import PatientDataset, custom_collate
from src.training.train import test_model_on_dataset

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "./output/best_model.pth"
        self.root_dir = "./dataset/testing_dataset"
        self.labels_dir = "./dataset/testing_dataset_patient_id"
        
        # Skip tests if data or model doesn't exist
        self.skip_test = not (os.path.exists(self.root_dir) and 
                             os.path.exists(self.labels_dir) and
                             os.path.exists(self.model_path))
    
    def test_end_to_end_inference(self):
        """Test end-to-end model inference on a small dataset"""
        if self.skip_test:
            self.skipTest("Test data or model not found")
        
        # Load model
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model = MILHistopathModel()
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
        except Exception as e:
            self.fail(f"Failed to load model: {e}")
        
        # Load a small subset of test data
        try:
            cancer = pd.read_csv(os.path.join(self.labels_dir, 'cancer.txt'), 
                                index_col='patient_id').index.tolist()[:2]  # Just 2 patients
            normal = pd.read_csv(os.path.join(self.labels_dir, 'normal.txt'), 
                                index_col='patient_id').index.tolist()[:2]  # Just 2 patients
            
            # Combine into a DataFrame
            all_data = pd.concat([
                pd.DataFrame({'label': 1}, index=cancer),
                pd.DataFrame({'label': 0}, index=normal)
            ])
            
            patient_ids = all_data.index.tolist()
            labels = all_data
        except Exception as e:
            self.fail(f"Failed to load test data: {e}")
        
        # Create dataset and dataloader
        temp_model = timm.create_model("resnet18", pretrained=True)
        data_config = timm.data.resolve_model_data_config(temp_model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        
        test_dataset = PatientDataset(
            root_dir=self.root_dir,
            patient_ids=patient_ids,
            labels=labels,
            model_transform=transform,
            max_patches=50  # Limit patches for quick testing
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=custom_collate
        )
        
        # Run inference
        criterion = nn.BCEWithLogitsLoss()
        test_metrics = test_model_on_dataset(model, test_loader, criterion, self.device)
        
        # Check that metrics are valid
        self.assertIn('accuracy', test_metrics)
        self.assertIn('f1', test_metrics)
        self.assertIn('auc', test_metrics)
        self.assertTrue(0 <= test_metrics['accuracy'] <= 1)
        self.assertTrue(0 <= test_metrics['f1'] <= 1)
        self.assertTrue(0 <= test_metrics['auc'] <= 1)

if __name__ == "__main__":
    unittest.main() 