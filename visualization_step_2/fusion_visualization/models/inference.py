import torch
from models.hiergat import HierGATSSL
from data.dataset import CoreImageProcessor
from models.graph_builder import HierarchicalGraphBuilder

class HierGATSSLInference:
    def __init__(self, 
                 model_path,
                 device=None,
                 mil_model_path=None):
        """
        Initialize inference setup
        
        Args:
            model_path: Path to saved model checkpoint
            device: torch device (will use CUDA if available when None)
            mil_model_path: Path to MIL model for feature extraction
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = HierGATSSL(
            input_dim=128,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            num_levels=3,
            dropout=0.1
        ).to(self.device)
        
        # Load model weights
        print(f"Loading HierGAT SSL model from: {model_path}")
        self._load_checkpoint(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize processors
        self.image_processor = CoreImageProcessor(patch_size=224, levels=3, mil_model_path=mil_model_path)
        self.graph_builder = HierarchicalGraphBuilder(
            spatial_threshold=1.5,
            scale_threshold=2.0,
            levels=3
        )
        
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint['epoch']}")
                print(f"Best loss: {checkpoint['best_loss']:.4f}")
                print(f"✅ HierGAT SSL model loaded successfully")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded model weights")
                print(f"✅ HierGAT SSL model loaded successfully")
                
        except Exception as e:
            print(f"❌ Error loading HierGAT SSL model: {str(e)}")
            raise
            
    def process_image(self, image_path):
        """
        Process a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict containing embeddings and attention weights
        """
        # Process image and build graph
        patch_features, metadata, viz_data = self.image_processor.process_core_image(image_path)
        graph_data = self.graph_builder.build_hierarchical_graph(patch_features)
        
        # Move to device
        graph_data = graph_data.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(graph_data)
            
        return {
            'node_embeddings': outputs['node_embeddings'].cpu(),
            'graph_embedding': outputs['graph_embedding'].cpu(),
            'attention_weights': {
                k: v.cpu() for k, v in outputs['attention_weights'].items()
            },
            'metadata': metadata,
            'viz_data': viz_data
        }
