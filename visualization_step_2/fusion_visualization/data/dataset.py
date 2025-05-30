# In data/dataset.py
from utils.imports import *
from data.patch_extractor import MultiscalePatchExtractor
from models.mil_model import MILHistopathModel
from models.graph_builder import HierarchicalGraphBuilder
from data.slide_processor import CustomSlide
from utils.visualization import save_visualization_data

# In data/dataset.py
from torch_geometric.loader import DataLoader  # Use PyG's DataLoader instead
from torch_geometric.data import Batch  # For batching PyG Data objects



class CoreImageProcessor:
    def __init__(self, patch_size=224, levels=3, mil_model_path=None):
        """
        Initialize processor with patch extraction and feature extraction
        
        Args:
            patch_size: Size of patches to extract
            levels: Number of scales in hierarchy
            mil_model_path: Path to the MIL model file
        """
        # Store patch_size as instance variable
        self.patch_size = patch_size
        self.levels = levels
        
        self.level_thresholds = {
            0: 0.9,  # Highest resolution
            1: 0.9,  # Medium resolution 
            2: 0.9   # Lowest resolution
        }
        
        self.patch_extractor = MultiscalePatchExtractor(
            patch_size=patch_size,
            level_thresholds=self.level_thresholds
        )
        
        # Load pre-trained MIL model for feature extraction
        self.mil_model = MILHistopathModel()
        try:
            if mil_model_path is None:
                mil_model_path = "/scratch/nk53/rm8989/gene_prediction/code/GRAPHITE/best_fine_tuned_model_for_resnet18_cancervsnormal_v4.pth"
            
            print(f"Loading MIL model for feature extraction from: {mil_model_path}")
            self.mil_model.load_state_dict(
                torch.load(mil_model_path), 
                strict=False
            )
            print(f"✅ MIL model for feature extraction loaded successfully")
        except Exception as e:
            print(f"❌ Error loading MIL model for feature extraction: {str(e)}")
            raise

        # Keep only feature extraction parts
        self.feature_extractor = nn.Sequential(
            self.mil_model.feature_extractor,
            self.mil_model.patch_projector
        ).eval()
        
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
            viz_data: Visualization data for reconstruction
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
        
        # Save visualization data
        viz_data = save_visualization_data(
            os.path.dirname(image_path),
            kept_patches,
            patch_metadata,
            slide
        )
        
        return patch_features, patch_metadata, viz_data


class HierGATSSLDataset(Dataset):
    def __init__(self, image_dir, patch_size=224, levels=3, mil_model_path=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.png'))
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")
        
        self.processor = CoreImageProcessor(patch_size=patch_size, levels=levels, mil_model_path=mil_model_path)
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = str(self.image_paths[idx])
        try:
            patch_features, metadata, viz_data = self.processor.process_core_image(image_path)
            
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