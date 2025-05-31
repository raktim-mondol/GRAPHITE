# data/slide_processor.py

from utils.imports import *



class CustomSlide:
    def __init__(self, file_path, levels=3):
        self.file_path = file_path
        self.levels = levels
        self.image = self._load_image()
        self.original_dimensions = self.image.shape[1], self.image.shape[0]
        self.level_dimensions = self._calculate_level_dimensions()
        self.level_downsamples = self._calculate_level_downsamples()
        
    def _load_image(self):
        try:
            img = cv2.imread(self.file_path)
            if img is None:
                raise ValueError(f"Failed to load image: {self.file_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            raise ValueError(f"Error loading image {self.file_path}: {str(e)}")
            
    def _calculate_level_dimensions(self):
        dimensions = []
        base_width, base_height = self.original_dimensions
        for i in range(self.levels):
            level_width = base_width // (2 ** i)
            level_height = base_height // (2 ** i)
            dimensions.append((level_width, level_height))
        return dimensions
        
    def _calculate_level_downsamples(self):
        return [2 ** i for i in range(self.levels)]
        
    def read_region(self, location, level, size):
        try:
            x, y = location
            width, height = size
            downsample = self.level_downsamples[level]
            
            x_level = x * downsample
            y_level = y * downsample
            
            region = cv2.resize(
                self.image[y_level:y_level + height * downsample, 
                          x_level:x_level + width * downsample],
                (width, height),
                interpolation=cv2.INTER_AREA
            )
            
            return region
            
        except Exception as e:
            raise RuntimeError(f"Error reading region at ({x}, {y}), level {level}: {str(e)}")