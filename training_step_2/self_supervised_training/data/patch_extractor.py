from utils.imports import *

# data/patch_extractor.py
#from data.slide_processor import CustomSlide



class MultiscalePatchExtractor:
    def __init__(self, patch_size=224, level_thresholds=None):
        """
        Initialize with level-specific white thresholds
        
        Args:
            patch_size: Size of patches to extract
            level_thresholds: Dict of level-specific white thresholds
                            e.g., {0: 0.8, 1: 0.7, 2: 0.6}
                            If a level is not specified, default is 0.8
        """
        self.patch_size = patch_size
        # Default threshold is 0.8 if not specified for a level
        self.level_thresholds = level_thresholds if level_thresholds else {}
        self.default_threshold = 0.8
        
    def _get_threshold_for_level(self, level: int) -> float:
        """Get white threshold for specific level"""
        return self.level_thresholds.get(level, self.default_threshold)
        
#    def _is_white_patch(self, patch: np.ndarray, level: int) -> bool:
#        if len(patch.shape) == 3:
#            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
#        else:
#            gray = patch
#        white_pixels = np.sum(gray > 230)
#        white_percentage = white_pixels / gray.size
#        threshold = self._get_threshold_for_level(level)
#        return white_percentage > threshold
    def _is_white_patch(self, patch: np.ndarray, level: int) -> bool:
        return False


    def _extract_level_patches(self, slide, level: int) -> Tuple[List[Dict], List[Dict]]:
        level_width, level_height = slide.level_dimensions[level]
        kept_patches = []
        filtered_patches = []
        downsample = slide.level_downsamples[level]
        
        threshold = self._get_threshold_for_level(level)
        #print(f"\nProcessing Level {level} with white threshold: {threshold}")
        #print(f"Level dimensions: {level_width}x{level_height}")
        
        # First, extract regular grid patches
        for y in range(0, level_height, self.patch_size):
            for x in range(0, level_width, self.patch_size):
                try:
                    patch = slide.read_region(
                        location=(x, y),
                        level=level,
                        size=(self.patch_size, self.patch_size)
                    )
                    
                    if not isinstance(patch, np.ndarray):
                        patch = np.array(patch)
                    
                    coord_info = {
                        'level_coords': (x, y),
                        'base_coords': (x * downsample, y * downsample),
                        'size': (self.patch_size, self.patch_size),
                        'downsample': downsample,
                        'level': level,
                        'is_regular': True
                    }
                    
                    patch_info = {
                        **coord_info,
                        'patch': patch,
                        'is_white': self._is_white_patch(patch, level)
                    }
                    
                    if not patch_info['is_white']:
                        kept_patches.append(patch_info)
                    else:
                        filtered_patches.append({
                            **coord_info, 
                            'reason': 'white_content',
                            'threshold_used': threshold
                        })
                        
                except Exception as e:
                    print(f"Error at position ({x}, {y}): {str(e)}")
                    continue

        # Handle remaining areas at edges
        remaining_width = level_width % self.patch_size
        remaining_height = level_height % self.patch_size
        
        # Add patches for remaining width
        if remaining_width > 0:
            for y in range(0, level_height - self.patch_size + 1, self.patch_size):
                x = level_width - self.patch_size
                try:
                    patch = slide.read_region(
                        location=(x, y),
                        level=level,
                        size=(self.patch_size, self.patch_size)
                    )
                    
                    if not isinstance(patch, np.ndarray):
                        patch = np.array(patch)
                    
                    coord_info = {
                        'level_coords': (x, y),
                        'base_coords': (x * downsample, y * downsample),
                        'size': (self.patch_size, self.patch_size),
                        'downsample': downsample,
                        'level': level,
                        'is_regular': False,
                        'edge_type': 'width'
                    }
                    
                    patch_info = {
                        **coord_info,
                        'patch': patch,
                        'is_white': self._is_white_patch(patch, level)
                    }
                    
                    if not patch_info['is_white']:
                        kept_patches.append(patch_info)
                    else:
                        filtered_patches.append({
                            **coord_info, 
                            'reason': 'white_content',
                            'threshold_used': threshold
                        })
                        
                except Exception as e:
                    print(f"Error at edge position ({x}, {y}): {str(e)}")
                    continue

        # Add patches for remaining height
        if remaining_height > 0:
            for x in range(0, level_width - self.patch_size + 1, self.patch_size):
                y = level_height - self.patch_size
                try:
                    patch = slide.read_region(
                        location=(x, y),
                        level=level,
                        size=(self.patch_size, self.patch_size)
                    )
                    
                    if not isinstance(patch, np.ndarray):
                        patch = np.array(patch)
                    
                    coord_info = {
                        'level_coords': (x, y),
                        'base_coords': (x * downsample, y * downsample),
                        'size': (self.patch_size, self.patch_size),
                        'downsample': downsample,
                        'level': level,
                        'is_regular': False,
                        'edge_type': 'height'
                    }
                    
                    patch_info = {
                        **coord_info,
                        'patch': patch,
                        'is_white': self._is_white_patch(patch, level)
                    }
                    
                    if not patch_info['is_white']:
                        kept_patches.append(patch_info)
                    else:
                        filtered_patches.append({
                            **coord_info, 
                            'reason': 'white_content',
                            'threshold_used': threshold
                        })
                        
                except Exception as e:
                    print(f"Error at edge position ({x}, {y}): {str(e)}")
                    continue

        #print(f"Level {level} Summary:")
        #print(f"White threshold used: {threshold}")
        #print(f"Regular patches: {len([p for p in kept_patches if p.get('is_regular', False)])}")
        #print(f"Edge patches: {len([p for p in kept_patches if not p.get('is_regular', True)])}")
        #print(f"Total kept patches: {len(kept_patches)}")
        #print(f"Filtered patches: {len(filtered_patches)}")
        
        return kept_patches, filtered_patches
        
    def extract_patches(self, slide) -> Tuple[List[Dict], Dict]:
        all_kept_patches = []
        patch_metadata = {
            'filtered_patches': {},
            'level_statistics': {},
            'extraction_params': {
                'patch_size': self.patch_size,
                'level_thresholds': self.level_thresholds,  # Add level-specific thresholds
                'default_threshold': self.default_threshold  # Add default threshold
            }
        }
        
        for level in range(slide.levels):
            kept_patches, filtered_patches = self._extract_level_patches(slide, level)
            all_kept_patches.extend(kept_patches)
            
            patch_metadata['filtered_patches'][f'level_{level}'] = filtered_patches
            patch_metadata['level_statistics'][f'level_{level}'] = {
                'total_patches': len(kept_patches) + len(filtered_patches),
                'kept_patches': len(kept_patches),
                'filtered_patches': len(filtered_patches),
                'resolution': slide.level_dimensions[level],
                'white_threshold_used': self._get_threshold_for_level(level)  # Add threshold used for this level
            }
        
        return all_kept_patches, patch_metadata