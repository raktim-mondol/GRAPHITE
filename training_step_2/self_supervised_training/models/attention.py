from utils.imports import *
# In models/attention.py

class ScaleWiseAttention(nn.Module):
    def __init__(self, hidden_dim, num_levels=3):
        """
        Enhanced Scale-wise attention module
        
        Args:
            hidden_dim: Dimension of node features
            num_levels: Number of scales in hierarchy
        """
        super().__init__()
        
        self.num_levels = num_levels
        
        # Level-specific attention networks
        self.level_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_levels)
        ])
        
        # Cross-scale attention
        self.cross_scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_levels)
        )
        
    def forward(self, x, level_indices, return_attentions=True):
        """
        Forward pass
        
        Args:
            x: Node features (N x hidden_dim)
            level_indices: Dict of node indices for each level
            return_attentions: Whether to return attention weights
            
        Returns:
            output: Graph-level representation
            attentions: Dict of attention weights if return_attentions=True
        """
        level_outputs = []
        level_attentions = {}
        
        # Process each level
        for level in range(self.num_levels):
            indices = level_indices[f'level_{level}_indices']
            if len(indices) == 0:
                continue
                
            # Get features for this level
            level_x = x[indices]
            
            # Compute attention scores
            scores = self.level_attention[level](level_x)
            weights = F.softmax(scores, dim=0)
            
            # Weighted aggregation
            level_out = torch.sum(weights * level_x, dim=0)
            level_outputs.append(level_out)
            
            if return_attentions:
                level_attentions[f'level_{level}'] = weights
        
        # Stack level outputs
        level_outputs = torch.stack(level_outputs)
        
        # Cross-scale attention
        cross_scale_weights = self.cross_scale_attention(level_outputs)
        cross_scale_weights = F.softmax(cross_scale_weights, dim=-1)
        
        # Final weighted combination
        output = torch.sum(level_outputs * cross_scale_weights.unsqueeze(-1), dim=0)
        
        if return_attentions:
            level_attentions['cross_scale'] = cross_scale_weights
            return output, level_attentions
        
        return output
        
       
        

class HierarchicalGAT(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 num_heads=4,
                 dropout=0.1):
        """
        Hierarchical GAT layer
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Make sure hidden_dim is divisible by num_heads
        self.head_dim = hidden_dim // num_heads
        
        # GAT for spatial edges (within same level)
        self.gat_spatial = GATConv(
            input_dim, 
            self.head_dim,  # Output dim per head
            heads=num_heads,
            dropout=dropout,
            add_self_loops=True
        )
        
        # GAT for cross-scale edges
        self.gat_cross_scale = GATConv(
            input_dim,  # Changed from hidden_dim to input_dim
            self.head_dim,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass
        
        Args:
            x: Node features
            edge_index: Edge connections
            edge_type: Edge types (0: spatial, 1: cross-scale)
            
        Returns:
            x: Updated node features
        """
        # Ensure edge_type is on the same device as the model
        device = x.device
        edge_type = edge_type.to(device)
        
        # Split edges by type
        spatial_mask = edge_type == 0
        cross_scale_mask = edge_type == 1
        
        spatial_edges = edge_index[:, spatial_mask]
        cross_scale_edges = edge_index[:, cross_scale_mask]
        
        # Ensure edges are on the correct device
        spatial_edges = spatial_edges.to(device)
        cross_scale_edges = cross_scale_edges.to(device)
        
        # Process spatial and cross-scale edges
        if spatial_edges.size(1) > 0:
            x_spatial = self.gat_spatial(x, spatial_edges)
        else:
            x_spatial = torch.zeros_like(x)
            
        if cross_scale_edges.size(1) > 0:
            x_cross = self.gat_cross_scale(x, cross_scale_edges)
        else:
            x_cross = torch.zeros_like(x)
        
        # Combine and normalize
        x = x_spatial + x_cross
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x
