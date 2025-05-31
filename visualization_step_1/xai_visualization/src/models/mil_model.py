"""
Multiple Instance Learning (MIL) model for histopathology.

This module contains the MIL model architecture used for histopathology image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MILHistopathModel(nn.Module):
    """
    Multiple Instance Learning model for histopathology image classification.
    
    This model uses a pre-trained feature extractor (ResNet) followed by attention-based
    aggregation for patient-level prediction from patch-level features.
    """
    
    def __init__(self, num_classes=2, feat_dim=512, proj_dim=128):
        """
        Initialize the MIL model.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            feat_dim: Feature dimension from the backbone (default: 512 for ResNet18)
            proj_dim: Projection dimension (default: 128)
        """
        super(MILHistopathModel, self).__init__()
        
        # Load the pre-trained ResNet18 model
        model_name = "hf-hub:1aurent/resnet18.tiatoolbox-kather100k"
        # Alternative models:
        # model_name = "hf-hub:1aurent/resnet50.tiatoolbox-kather100k"
        # model_name = "hf-hub:1aurent/resnet50.tcga_brca_simclr"
        # model_name = "resnet50.a1_in1k"   # using imagenet
        
        self.feature_extractor = timm.create_model(model_name, pretrained=True)
        
        # Remove the last layer of the feature extractor
        self.feature_extractor.fc = nn.Identity()
        
        # Patch-level projection head
        self.patch_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Patient-level projection head
        self.patient_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(feat_dim // 2, num_classes)
        )
        
        # Layer normalization for patient features
        self.patient_layer_norm = nn.LayerNorm(feat_dim)
        
    def forward(self, x):
        """
        Forward pass of the MIL model.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, channels, height, width)
            
        Returns:
            Tuple of (patch_projections, patient_projections, logits, attention_weights)
        """
        # x shape: (batch_size, num_patches, channels, height, width)
        batch_size, num_patches = x.shape[:2]
        
        # Reshape and pass through feature extractor - clone to avoid view issues
        x_reshaped = x.contiguous().view(-1, *x.shape[2:])
        patch_features = self.feature_extractor(x_reshaped)
        patch_features = patch_features.contiguous().view(batch_size, num_patches, -1)
        
        # Project patch features
        patch_projections = self.patch_projector(patch_features)
        
        # Compute attention weights
        attention_weights = self.attention(patch_features).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Compute patient-level features using attention-weighted aggregation
        patient_features = torch.sum(attention_weights.unsqueeze(-1) * patch_features, dim=1)
        
        # Apply layer normalization to patient features
        patient_features = self.patient_layer_norm(patient_features)
        
        # Project patient features
        patient_projections = self.patient_projector(patient_features)
        
        # Compute class logits
        logits = self.classifier(patient_features)
        
        return patch_projections, patient_projections, logits, attention_weights 