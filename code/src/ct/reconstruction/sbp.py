import numpy as np
from scipy.ndimage import rotate

def simple_backprojection(sinogram: np.ndarray, 
                         angles: np.ndarray) -> np.ndarray:
    """
    Perform simple backprojection reconstruction
    
    Args:
        sinogram: 2D array of projection data (angles x detector_bins)
        angles: Array of projection angles in radians
        
    Returns:
        Reconstructed image (square array)
    """
    num_angles, num_detectors = sinogram.shape

    image_size = num_detectors
    
    reconstruction = np.zeros((image_size, image_size), dtype=np.float32)

    x = np.arange(image_size) - image_size / 2
    y = np.arange(image_size) - image_size / 2
    X, Y = np.meshgrid(x, y)

    detector_center = (num_detectors - 1) / 2.0 

    for i, angle in enumerate(angles):
        current_projection = sinogram[i, :]

        s_prime = X * np.cos(angle) + Y * np.sin(angle)
        
        detector_indices_float = s_prime + detector_center 

        detector_indices = np.arange(num_detectors)

        interpolated_values = np.interp(
            detector_indices_float.flatten(),
            detector_indices, 
            current_projection 
        ).reshape(image_size, image_size)
        
        reconstruction += interpolated_values

    return reconstruction / num_angles