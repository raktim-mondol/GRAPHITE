"""
Color normalization utilities for histopathology images.
Implementation of the Macenko color normalization method.

References:
    Macenko, Marc, et al. "A method for normalizing histology slides for quantitative analysis." 
    2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.
"""

import numpy as np
from PIL import Image

class NormalizationError(Exception):
    """Exception raised when color normalization fails."""
    pass

class MacenkoColorNormalization:
    """
    Macenko color normalization for histopathology images.
    
    This class implements the Macenko method for normalizing H&E stained histopathology images.
    It separates the Hematoxylin and Eosin stain components and normalizes them to 
    a reference standard, making the appearance more consistent across images from 
    different labs, scanners, or staining batches.
    """
    def __init__(self, Io=240, alpha=1, beta=0.15):
        """
        Initialize the Macenko color normalizer.
        
        Args:
            Io (int): Intensity normalization parameter (white level)
            alpha (float): Percentile for phi angle thresholding
            beta (float): Minimum threshold for optical density
        """
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        
        # Reference H&E stain matrix and maximum stain concentrations
        self.HERef = np.array([[0.5626, 0.2159],
                              [0.7201, 0.8012],
                              [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])
        
    def __call__(self, img):
        """
        Apply color normalization to an image.
        
        Args:
            img (PIL.Image): Input image to normalize
            
        Returns:
            PIL.Image: Normalized image
            
        Raises:
            NormalizationError: If normalization fails
        """
        img = np.array(img)
        h, w, c = img.shape
        img = img.reshape((-1,3))
        
        # Calculate optical density
        OD = -np.log10((img.astype(np.float64)+1)/self.Io)
        
        # Remove pixels with low optical density
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        
        if ODhat.size == 0:
            raise NormalizationError("No valid pixels found after OD thresholding")
        
        try:
            # Check if we have enough unique points
            if ODhat.shape[0] <= 1:
                raise NormalizationError("Not enough unique points for normalization")
            
            # Calculate covariance matrix and eigenvectors
            cov = np.cov(ODhat.T)
            if np.isnan(cov).any() or np.isinf(cov).any():
                raise NormalizationError("Invalid covariance matrix")
            
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Check for non-negative eigenvalues
            if not np.all(eigvals > 0):
                raise NormalizationError("Non-positive eigenvalues in covariance matrix")
            
            # Find the two eigenvectors corresponding to the two largest eigenvalues
            That = ODhat.dot(eigvecs[:,1:3])
            phi = np.arctan2(That[:,1],That[:,0])
            minPhi = np.percentile(phi, self.alpha)
            maxPhi = np.percentile(phi, 100-self.alpha)
            vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
            vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
            
            # Project on the plane spanned by the eigenvectors corresponding to the two
            # largest eigenvalues
            if vMin[0] > vMax[0]:    
                HE = np.array((vMin[:,0], vMax[:,0])).T
            else:
                HE = np.array((vMax[:,0], vMin[:,0])).T
            
            # Calculate stain concentration matrix
            Y = np.reshape(OD, (-1, 3)).T
            C = np.linalg.lstsq(HE,Y, rcond=None)[0]
            maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
            
            # Avoid division by zero
            tmp = np.divide(maxC, self.maxCRef, out=np.zeros_like(maxC), where=self.maxCRef!=0)
            
            # Normalize stain concentrations
            C2 = np.divide(C, tmp[:, np.newaxis], out=np.zeros_like(C), where=tmp[:, np.newaxis]!=0)
            
            # Recreate the normalized image
            Inorm = np.multiply(self.Io, np.exp(-self.HERef.dot(C2)))
            Inorm[Inorm>255] = 254
            Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
            
            return Image.fromarray(Inorm)
        
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
            raise NormalizationError(f"Normalization failed: {str(e)}")
