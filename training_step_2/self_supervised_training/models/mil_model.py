from utils.imports import *

class MILHistopathModel(nn.Module):
    def __init__(self, num_classes=2, feat_dim=512, proj_dim=128):
        super(MILHistopathModel, self).__init__()
        
        # Load the pre-trained ResNet18 model
        model_name = "hf-hub:1aurent/resnet18.tiatoolbox-kather100k"
        #model_name = "hf-hub:1aurent/resnet50.tiatoolbox-kather100k"
        #model_name = "hf-hub:1aurent/resnet50.tcga_brca_simclr"
        #model_name = "resnet50.a1_in1k"   #using imagenet
        
        self.feature_extractor = timm.create_model(model_name, pretrained=True)
        
        # Remove the last layer of the feature extractor
        self.feature_extractor.fc = nn.Identity()
        
        # Patch-level projection head
        self.patch_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            #nn.Dropout(p=0.01),  # Add dropout with 50% probability
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
            #nn.Dropout(p=0.1),  # Add dropout with 50% probability
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Add dropout with 50% probability
            nn.Linear(feat_dim // 2, num_classes)
        )
        
        # Layer normalization for patient features
        self.patient_layer_norm = nn.LayerNorm(feat_dim)
        

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        #patch_features = self.feature_extractor(x)
        #print(f"After feature extractor shape: {patch_features.shape}")
    
    
        # x shape: (batch_size, num_patches, channels, height, width)
        batch_size, num_patches = x.shape[:2]
        
        # Reshape and pass through feature extractor
        x = x.view(-1, *x.shape[2:])
        patch_features = self.feature_extractor(x)
        #print(patch_features.shape) 
        patch_features = patch_features.view(batch_size, num_patches, -1)
        
        # Project patch features
        patch_projections = self.patch_projector(patch_features)
        
        # Compute attention weights
        attention_weights = self.attention(patch_features).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Compute patient-level features
        patient_features = torch.sum(attention_weights.unsqueeze(-1) * patch_features, dim=1)
        
        # Apply layer normalization to patient features
        patient_features = self.patient_layer_norm(patient_features)
        
        # Project patient features
        patient_projections = self.patient_projector(patient_features)
        
        # Compute class logits
        logits = self.classifier(patient_features)
        
        return patch_projections, patient_projections, logits, attention_weights
        
        
        
        
        
        
        
        