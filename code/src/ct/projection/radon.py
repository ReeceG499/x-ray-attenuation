import numpy as np
import scipy

def radon_transform(image: np.ndarray, 
                   angles: np.ndarray) -> np.ndarray:
    """
    Pure Radon transform without detector modeling
    
    Args:
        image: 2D numpy array (square shape)
        angles: Projection angles in radians
        
    Returns:
        Sinogram (angles x detector_pixels)
    """
    size = image.shape[0]
    sinogram = np.zeros((len(angles), size))

    for i, angle in enumerate(angles):
        rotated = scipy.ndimage.rotate(image, np.degrees(angle), 
                                      reshape=False, order=1)
        sinogram[i] = rotated.sum(axis=0)
        
    return sinogram

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