# In data/dataset.py
from utils.imports import *
from data.patch_extractor import MultiscalePatchExtractor
from models.mil_model import MILHistopathModel
from models.graph_builder import HierarchicalGraphBuilder
from data.slide_processor import CustomSlide


# In data/dataset.py
from torch_geometric.loader import DataLoader  # Use PyG's DataLoader instead
from torch_geometric.data import Batch  # For batching PyG Data objects

class CoreImageProcessor:
    def __init__(self, patch_size=224, levels=3):
        """
        Initialize processor with patch extraction and feature extraction
        
        Args:
            patch_size: Size of patches to extract
            levels: Number of scales in hierarchy
        """
        self.level_thresholds = {
            0: 0.9,  # Highest resolution
            1: 0.9,  # Medium resolution 
            2: 0.9   # Lowest resolution
        }
        
        self.patch_extractor = MultiscalePatchExtractor(
            patch_size=patch_size,
            level_thresholds=self.level_thresholds
        )
        
        # Load the fine-tuned MIL model for feature extraction
        print("Loading fine-tuned MIL model for feature extraction...")
        
        # Determine the best device to use
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.mil_model = MILHistopathModel()
        model_path = 'output/mil_fine_tuned_model/best_model.pth'
        
        try:
            # Load model directly to the target device for better efficiency
            self.mil_model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=False), 
                strict=False
            )
            print(f"Successfully loaded fine-tuned model from {model_path} to {self.device}")
        except Exception as e:
            print(f"Warning: Could not load fine-tuned model from {model_path}: {e}")
            print("Using pre-trained model instead...")

        # Keep only feature extraction parts and move to device
        self.feature_extractor = nn.Sequential(
            self.mil_model.feature_extractor,
            self.mil_model.patch_projector
        ).eval().to(self.device)
        
        # Get the data configuration and transformation for the feature extractor
        self.data_config = timm.data.resolve_model_data_config(self.mil_model.feature_extractor)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)
        
    def preprocess_patch(self, patch):
        """
        Preprocess a single patch
        """
        # Convert numpy array to PIL Image
        if isinstance(patch, np.ndarray):
            patch = Image.fromarray(patch)
        
        # Apply the transformation
        patch = self.transform(patch)
        return patch
        
    def process_core_image(self, image_path):
        """
        Process a single core image
        
        Args:
            image_path: Path to core image
            
        Returns:
            patch_features: Dict of patch features by level
            patch_metadata: Dict containing patch coordinates and extraction info
        """
        # Load slide
        slide = CustomSlide(image_path, levels=3)
        
        # Extract patches using your existing pipeline
        kept_patches, patch_metadata = self.patch_extractor.extract_patches(slide)
        
        # Extract features for each level
        patch_features = {}
        for level in range(3):
            level_patches = [p for p in kept_patches if p['level'] == level]
            if level_patches:
                # Preprocess patches
                processed_patches = torch.stack([
                    self.preprocess_patch(p['patch']) 
                    for p in level_patches
                ])
                
                # Move to device if available
                device = next(self.feature_extractor.parameters()).device
                processed_patches = processed_patches.to(device)
                
                with torch.no_grad():
                    features = self.feature_extractor(processed_patches)
                    patch_features[level] = {
                        'features': features.cpu(),  # Move back to CPU
                        'coords': [(p['level_coords'], p['base_coords']) 
                                 for p in level_patches]
                    }
        
        return patch_features, patch_metadata


class HierGATSSLDataset(Dataset):
    def __init__(self, image_dir, patch_size=224, levels=3):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.png'))
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")
        
        self.processor = CoreImageProcessor(patch_size=patch_size, levels=levels)
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        try:
            patch_features, metadata = self.processor.process_core_image(image_path)
            
            graph_builder = HierarchicalGraphBuilder(
                spatial_threshold=1.5,
                scale_threshold=2.0,
                levels=3
            )
            graph_data = graph_builder.build_hierarchical_graph(patch_features)
            
            return graph_data
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise e