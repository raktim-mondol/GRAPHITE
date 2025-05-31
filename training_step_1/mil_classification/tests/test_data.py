"""
Tests for the PatientDataset and related data utilities.
"""

import unittest
import os
import pandas as pd
import torch
from src.data.datasets import PatientDataset, BalancedBatchSampler, setup_dataloaders

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Setup mock data for testing
        self.root_dir = "./dataset/training_dataset"
        
        # Check if the directory exists before proceeding
        self.skip_test = not os.path.exists(self.root_dir)
        if self.skip_test:
            return
            
        # Create test data
        cancer_patients = pd.read_csv(os.path.join("./dataset/training_dataset_patient_id", "cancer.txt"), 
                                     index_col="patient_id").index.tolist()
        normal_patients = pd.read_csv(os.path.join("./dataset/training_dataset_patient_id", "normal.txt"), 
                                     index_col="patient_id").index.tolist()
        
        # Sample a small number of patients for testing
        self.patient_ids = cancer_patients[:2] + normal_patients[:2]
        self.labels = pd.concat([
            pd.DataFrame({"label": 1}, index=cancer_patients[:2]),
            pd.DataFrame({"label": 0}, index=normal_patients[:2])
        ])
    
    def test_patient_dataset_creation(self):
        """Test that PatientDataset can be created"""
        if self.skip_test:
            self.skipTest("Test data directory not found")
            
        dataset = PatientDataset(
            root_dir=self.root_dir,
            patient_ids=self.patient_ids,
            labels=self.labels,
            max_patches=10
        )
        
        self.assertEqual(len(dataset), len(self.patient_ids))
        
    def test_dataloader_setup(self):
        """Test setup_dataloaders function"""
        if self.skip_test:
            self.skipTest("Test data directory not found")
            
        train_loader, val_loader = setup_dataloaders(
            root_dir=self.root_dir,
            patient_ids=self.patient_ids,
            labels=self.labels,
            batch_size=2,
            max_patches=10,
            test_size=0.5
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

if __name__ == "__main__":
    unittest.main()
