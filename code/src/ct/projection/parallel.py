import numpy as np
import scipy
from .radon import radon_transform

def parallel_project(image: np.ndarray,
                    angles: np.ndarray,
                    detector_size: int) -> np.ndarray:
    """
    Scanner-realistic parallel-beam projection
    
    Args:
        detector_size: Number of detector elements
    """
    # Simple downsampling for detector resolution
    if detector_size != image.shape[0]:
        image = scipy.ndimage.zoom(image, 
                                  detector_size/image.shape[0], 
                                  order=1)
    
    return radon_transform(image, angles)