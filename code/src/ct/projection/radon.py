import numpy as np
from skimage.transform import radon

def radon_transform(image: np.ndarray, 
                   angles: np.ndarray) -> np.ndarray:
    """
    Parallel-beam projection using scikit-image's accurate Radon transform 
    (simplified as nnumpys rotate was giving issues).
    
    Args:
        image: 2D numpy array (square shape)
        angles: Projection angles in radians
    Returns:
        Sinogram (num_angles x detector_pixels)
    """
    angles_deg = np.degrees(angles)
    sinogram = radon(image, theta=angles_deg, circle=False) 
    
    return sinogram