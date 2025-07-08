import numpy as np
from scipy.ndimage import zoom
from .radon import radon_transform

def parallel_project(image: np.ndarray, 
                    angles: np.ndarray,
                    detector_size: int = None) -> np.ndarray:
    """
    Simplified parallel-beam projection
    
    Args:
        image: 2D numpy array (square shape)
        angles: Projection angles in radians
        detector_size: Number of detector elements (optional)
        
    Returns:
        Sinogram (angles x detector_pixels)
    """
    # Use image size as default detector resolution
    if detector_size is None:
        detector_size = image.shape[1]
    
    # Handle different detector resolution
    if detector_size != image.shape[1]:
        # Resize image to match detector resolution
        zoom_factor = detector_size / image.shape[1]
        image_resized = zoom(image, (zoom_factor, zoom_factor), order=1)
    else:
        image_resized = image
    
    # Perform Radon transform
    sinogram = radon_transform(image_resized, angles)
    
    return sinogram