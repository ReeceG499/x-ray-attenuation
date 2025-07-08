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
        
    Note:
        This is an unfiltered backprojection that will produce a blurred image.
        Use filtered_backprojection for clinical-quality reconstructions.
    """
    num_angles, num_bins = sinogram.shape
    size = num_bins  # Assume square reconstruction
    reconstruction = np.zeros((size, size))
    
    # Convert angles to degrees for rotation function
    angles_deg = np.degrees(angles)
    
    for i, angle in enumerate(angles_deg):
        # Create backprojection for this angle
        backproj = np.tile(sinogram[i], (size, 1))
        
        # Rotate back to original orientation
        # Note: Use negative angle to reverse the projection rotation
        rotated = rotate(backproj, -angle, reshape=False, order=1)
        
        # Add to reconstruction
        reconstruction += rotated
    
    # Normalize by number of projections
    return reconstruction / num_angles