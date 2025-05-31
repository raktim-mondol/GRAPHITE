from utils.imports import *

class HierarchicalGraphBuilder:
    def __init__(self, 
                 spatial_threshold=1.5,
                 scale_threshold=2.0,
                 levels=3):
        """
        Enhanced Hierarchical Graph Builder
        
        Args:
            spatial_threshold: Maximum distance for spatial edges within same level
            scale_threshold: Maximum distance for cross-scale edges
            levels: Number of scales in hierarchy
        """
        self.spatial_threshold = spatial_threshold
        self.scale_threshold = scale_threshold
        self.levels = levels
        
    def build_hierarchical_graph(self, patch_features):
        """
        Build hierarchical graph from multi-scale patch features
        
        Args:
            patch_features: Dict containing features and coordinates for each level
                {level: {'features': tensor, 'coords': list of tuples}}
                
        Returns:
            data: PyG Data object containing:
                - x: Node features
                - edge_index: Edge connections
                - edge_type: Edge types (0: spatial, 1: cross-scale)
                - level_indices: Indices for nodes at each level
                - pos: Node positions
        """
        # Combine features and coordinates from all levels
        all_features = []
        all_coords = []
        level_indices = {i: [] for i in range(self.levels)}
        current_idx = 0
        
        for level in range(self.levels):
            if level in patch_features:
                features = patch_features[level]['features']
                coords = patch_features[level]['coords']
                
                all_features.append(features)
                all_coords.extend(coords)
                
                # Store indices for this level
                num_patches = len(features)
                level_indices[level] = list(range(current_idx, current_idx + num_patches))
                current_idx += num_patches
        
        # Stack all features
        node_features = torch.cat(all_features, dim=0)
        
        # Build spatial edges (within same level)
        spatial_edges = []
        for level, indices in level_indices.items():
            for i, idx1 in enumerate(indices):
                coord1 = all_coords[idx1][0]  # Use level_coords
                for j, idx2 in enumerate(indices[i+1:], i+1):
                    coord2 = all_coords[idx2][0]
                    
                    # Compute normalized distance
                    dist = self._compute_spatial_distance(coord1, coord2)
                    if dist <= self.spatial_threshold:
                        spatial_edges.append([idx1, idx2])
                        spatial_edges.append([idx2, idx1])  # Add both directions
        
        # Build cross-scale edges
        cross_scale_edges = []
        for level1 in range(self.levels-1):
            indices1 = level_indices[level1]
            for level2 in range(level1+1, self.levels):
                indices2 = level_indices[level2]
                
                for idx1 in indices1:
                    coord1 = all_coords[idx1][1]  # Use base_coords
                    for idx2 in indices2:
                        coord2 = all_coords[idx2][1]
                        
                        # Check if patches are hierarchically related
                        if self._check_hierarchical_relationship(coord1, coord2, level1, level2):
                            cross_scale_edges.append([idx1, idx2])
                            cross_scale_edges.append([idx2, idx1])
        
        # Combine edges and create edge types
        all_edges = spatial_edges + cross_scale_edges
        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        edge_type = torch.cat([
            torch.zeros(len(spatial_edges), dtype=torch.long),
            torch.ones(len(cross_scale_edges), dtype=torch.long)
        ])
        
        # Create node positions tensor
        pos = torch.tensor([[c[0][0], c[0][1]] for c in all_coords], dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            pos=pos
        )
        
        # Add level indices as attributes
        for level, indices in level_indices.items():
            data[f'level_{level}_indices'] = torch.tensor(indices, dtype=torch.long)
        
        return data
    
    def _compute_spatial_distance(self, coord1, coord2):
        """Compute normalized distance between patches at same level"""
        x1, y1 = coord1
        x2, y2 = coord2
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2) / 224  # Normalize by patch size
    
    def _check_hierarchical_relationship(self, coord1, coord2, level1, level2):
        """Check if two patches at different levels are hierarchically related"""
        x1, y1 = coord1
        x2, y2 = coord2
        level_diff = level2 - level1
        scale_factor = 2 ** level_diff
        
        # Convert coordinates to same scale
        x1_scaled = x1 * scale_factor
        y1_scaled = y1 * scale_factor
        
        # Compute overlap
        overlap_threshold = self.scale_threshold * 224  # Use patch size
        distance = math.sqrt((x1_scaled - x2)**2 + (y1_scaled - y2)**2)
        
        return distance <= overlap_threshold