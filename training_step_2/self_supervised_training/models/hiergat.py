from utils.imports import *

from models.attention import ScaleWiseAttention, HierarchicalGAT


class HierGATSSL(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_gat_layers=3,
                 num_heads=4,
                 num_levels=3,
                 dropout=0.1):
        """
        Main model combining GAT and Scale-wise attention
        """
        super().__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            HierarchicalGAT(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for i in range(num_gat_layers)
        ])
        
        # Scale-wise attention
        self.scale_attention = ScaleWiseAttention(
            hidden_dim,
            num_levels=num_levels
        )
        
        # Projection head for self-supervised learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def to_device(self, data, device):
        """Helper function to move data to device"""
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.edge_type = data.edge_type.to(device)
        data.pos = data.pos.to(device)
        
        # Move level indices to device
        for level in range(3):  # Assuming 3 levels
            key = f'level_{level}_indices'
            if hasattr(data, key):
                setattr(data, key, getattr(data, key).to(device))
        
        return data
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data object
        """
        # Move data to same device as model
        device = next(self.parameters()).device
        data = self.to_device(data, device)
        
        x, edge_index = data.x, data.edge_index
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, data.edge_type)
        
        # Apply scale-wise attention
        graph_embedding, attention_weights = self.scale_attention(
            x, data, return_attentions=True
        )
        
        # Project for self-supervised learning
        projection = self.projection_head(graph_embedding)
        
        return {
            'node_embeddings': x,
            'graph_embedding': graph_embedding,
            'projection': projection,
            'attention_weights': attention_weights
        }