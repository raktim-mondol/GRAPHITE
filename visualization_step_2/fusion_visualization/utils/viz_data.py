from utils.imports import *

import matplotlib.pyplot as plt
import numpy as np

   
    
def save_visualization_data(output_dir, kept_patches, patch_metadata, slide):
    """
    Save enhanced visualization data including level-specific thresholds and patch information
    """
    viz_data = {
        'image_dimensions': {
            'width': slide.original_dimensions[0],
            'height': slide.original_dimensions[1]
        },
        'levels': slide.levels,
        'level_dimensions': slide.level_dimensions,
        'downsamples': slide.level_downsamples,
        'patch_size': patch_metadata['extraction_params']['patch_size'],
        'level_thresholds': patch_metadata['extraction_params']['level_thresholds'],
        'default_threshold': patch_metadata['extraction_params']['default_threshold'],
        'patches': {
            'kept': [{
                'level': p['level'],
                'level_coords': p['level_coords'],
                'base_coords': p['base_coords'],
                'size': p['size'],
                'downsample': p['downsample'],
                'is_regular': p.get('is_regular', True),
                'edge_type': p.get('edge_type', 'regular')
            } for p in kept_patches],
            'filtered': []
        }
    }
    
    # Add filtered patches with threshold information
    for level in range(slide.levels):
        level_filtered = patch_metadata['filtered_patches'].get(f'level_{level}', [])
        # Add threshold information to each filtered patch
        for patch in level_filtered:
            patch['threshold_used'] = viz_data['level_thresholds'].get(
                level, viz_data['default_threshold']
            )
        viz_data['patches']['filtered'].extend(level_filtered)
    
    viz_path = os.path.join(output_dir, 'visualization_data.json')
    with open(viz_path, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    return viz_data

def reconstruct_patch_layout(viz_data, level=0):
    """
    Creates a binary mask showing patch locations at specified level
    Returns: 
        - mask with 1 for kept patches, 2 for filtered patches, 0 for empty space
        - metadata about the reconstruction
    """
    width, height = viz_data['level_dimensions'][level]
    patch_size = viz_data['patch_size']
    threshold = viz_data['level_thresholds'].get(
        level, viz_data['default_threshold']
    )
    
    # Create mask and metadata tracking
    mask = np.zeros((height, width), dtype=np.uint8)
    reconstruction_metadata = {
        'level': level,
        'threshold_used': threshold,
        'patches': {
            'kept': [],
            'filtered': []
        }
    }
    
    # Add kept patches
    for patch in viz_data['patches']['kept']:
        if patch['level'] == level:
            x, y = patch['level_coords']
            if y + patch_size <= height and x + patch_size <= width:
                mask[y:y+patch_size, x:x+patch_size] = 1
                reconstruction_metadata['patches']['kept'].append({
                    'coords': (x, y),
                    'edge_type': patch.get('edge_type', 'regular')
                })
    
    # Add filtered patches
    for patch in viz_data['patches']['filtered']:
        if patch['level'] == level:
            x, y = patch['level_coords']
            if y + patch_size <= height and x + patch_size <= width:
                mask[y:y+patch_size, x:x+patch_size] = 2
                reconstruction_metadata['patches']['filtered'].append({
                    'coords': (x, y),
                    'edge_type': patch.get('edge_type', 'regular'),
                    'threshold_used': patch.get('threshold_used', threshold)
                })
    
    # Add statistics to metadata
    reconstruction_metadata['statistics'] = {
        'total_kept_patches': len(reconstruction_metadata['patches']['kept']),
        'total_filtered_patches': len(reconstruction_metadata['patches']['filtered']),
        'coverage_percentage': (np.sum(mask > 0) / mask.size) * 100
    }
    
    return mask, reconstruction_metadata

def visualize_reconstruction(mask, output_path=None):
    """
    Create a colored visualization of the reconstruction
    """
    # Create RGB image
    vis_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Kept patches in green
    vis_image[mask == 1] = [0, 255, 0]
    
    # Filtered patches in red
    vis_image[mask == 2] = [255, 0, 0]
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image

# Example usage:
def create_level_visualization(viz_data, output_dir, level):
    """
    Create and save visualization for a specific level
    """
    mask, metadata = reconstruct_patch_layout(viz_data, level)
    
    # Save the visualization
    vis_path = os.path.join(output_dir, f'level_{level}_visualization.png')
    vis_image = visualize_reconstruction(mask, vis_path)
    
    # Save the reconstruction metadata
    metadata_path = os.path.join(output_dir, f'level_{level}_reconstruction_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return vis_image, metadata
    
def reconstruct_from_patches(viz_data, patches_dir, level=0):
    """
    Reconstruct the full image using actual patches for a specific level
    """
    width, height = viz_data['level_dimensions'][level]
    patch_size = viz_data['patch_size']
    
    # Create empty canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Directory containing patches for this level
    level_dir = os.path.join(patches_dir, f'level_{level}_patches')
    
    # Add kept patches
    for patch_info in viz_data['patches']['kept']:
        if patch_info['level'] == level:
            x, y = patch_info['level_coords']
            base_x, base_y = patch_info['base_coords']
            edge_type = patch_info.get('edge_type', 'regular')
            
            # Construct patch filename
            filename = f'patch_l{level}_x{x}_y{y}_base_x{base_x}_y{base_y}_{edge_type}.png'
            patch_path = os.path.join(level_dir, filename)
            
            if os.path.exists(patch_path):
                patch = cv2.imread(patch_path)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                
                # Check boundaries
                if y + patch_size <= height and x + patch_size <= width:
                    canvas[y:y+patch_size, x:x+patch_size] = patch
    
    return canvas