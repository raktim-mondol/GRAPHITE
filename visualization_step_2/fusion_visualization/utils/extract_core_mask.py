from utils.imports import *

class CoreExtractor:
    """Class for extracting tissue core from background in histopathology images"""
    
    def __init__(self, threshold=200, kernel_size=7):
        """
        Initialize CoreExtractor
        
        Args:
            threshold: Threshold value for binary segmentation (0-255)
            kernel_size: Size of morphological kernel
        """
        self.threshold = threshold
        self.kernel_size = kernel_size
    
    def extract_core_mask(self, image):
        """
        Extract tissue core mask from background
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            mask: Binary mask where 255 is core and 0 is background
        """
        # Convert to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply binary threshold
        _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest contour (core)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            return mask
        else:
            return np.ones_like(gray) * 255
    
    def apply_core_mask(self, attention_map, mask):
        """
        Apply core mask to attention map
        
        Args:
            attention_map: Attention map as numpy array
            mask: Binary mask from extract_core_mask
            
        Returns:
            Masked attention map
        """
        return np.where(mask == 255, attention_map, 0)
    
    def get_core_boundary(self, mask):
        """
        Get core boundary contours for visualization
        
        Args:
            mask: Binary mask from extract_core_mask
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def process_image(self, image_path):
        """
        Process an image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            original: Original RGB image
            mask: Core mask
            contours: Core boundary contours
        """
        original = cv2.imread(str(image_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        mask = self.extract_core_mask(original)
        contours = self.get_core_boundary(mask)
        
        return original, mask, contours

def visualize_core_extraction(image_path, save_dir=None):
    """
    Visualize core extraction results
    
    Args:
        image_path: Path to image file
        save_dir: Directory to save visualization
    """
    extractor = CoreExtractor()
    original, mask, contours = extractor.process_image(image_path)
    
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(131)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    # Core Mask
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Core Mask')
    plt.axis('off')
    
    # Original with Core Boundary
    plt.subplot(133)
    plt.imshow(original)
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
    plt.title('Core Boundary')
    plt.axis('off')
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f'core_extraction_{Path(image_path).stem}.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.close()

