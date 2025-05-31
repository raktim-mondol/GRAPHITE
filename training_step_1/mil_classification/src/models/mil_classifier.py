"""
MIL Classifier Model Architecture.
Implements the Multiple Instance Learning model architecture for histopathology image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging

logger = logging.getLogger(__name__)

class MILHistopathModel(nn.Module):
    """
    Multiple Instance Learning model for histopathology image classification
    using an attention-based approach to aggregate patch-level features.
    
    Architecture:
    1. ResNet18 feature extractor with pretrained weights
    2. Patch-level projection
    3. Attention mechanism for patch aggregation
    4. Patient-level projection
    5. Classifier head
    """
    def __init__(self, num_classes=2, feat_dim=512, proj_dim=128, model_name="hf-hub:1aurent/resnet18.tiatoolbox-kather100k"):
        super(MILHistopathModel, self).__init__()
        
        # 1. ResNet18 feature extractor with pretrained weights
        self.feature_extractor = timm.create_model(model_name, pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        
        # 2. Patch-level projection head
        self.patch_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # 3. Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 4. Patient-level projection head
        self.patient_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(feat_dim // 2, num_classes if num_classes > 2 else 1)
        )
        
        # Layer normalization for patient features
        self.patient_layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        """
        Forward pass of the MIL model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, 3, H, W)
            
        Returns:
            tuple: patch_projections, patient_projections, logits, attention_weights
        """
        batch_size, num_patches = x.shape[:2]
        
        # Reshape and pass through feature extractor
        x = x.view(-1, *x.shape[2:])  # Reshape to (batch_size * num_patches, 3, H, W)
        patch_features = self.feature_extractor(x)  # Shape: (batch_size * num_patches, feat_dim)
        patch_features = patch_features.view(batch_size, num_patches, -1)  # Reshape back
        
        # Project patch features
        patch_projections = self.patch_projector(patch_features)  # Shape: (batch_size, num_patches, proj_dim)
        
        # Compute attention weights
        attention_weights = self.attention(patch_features).squeeze(-1)  # Shape: (batch_size, num_patches)
        attention_weights = F.softmax(attention_weights, dim=1)  # Shape: (batch_size, num_patches)
        
        # Compute patient-level features using attention
        patient_features = torch.sum(attention_weights.unsqueeze(-1) * patch_features, dim=1)  # Shape: (batch_size, feat_dim)
        
        # Apply layer normalization to patient features
        patient_features = self.patient_layer_norm(patient_features)
        
        # Project patient features
        patient_projections = self.patient_projector(patient_features)  # Shape: (batch_size, proj_dim)
        
        # Classify using patient features (not projections, matching ref_models.py)
        logits = self.classifier(patient_features)  # Shape: (batch_size, num_classes or 1)
        
        if logits.shape[1] == 1:
            # For binary classification with BCEWithLogitsLoss
            logits = logits.squeeze(-1)
        
        return patch_projections, patient_projections, logits, attention_weights
