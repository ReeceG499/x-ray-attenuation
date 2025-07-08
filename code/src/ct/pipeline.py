import numpy as np
from .phantom import SheppLoganPhantom
from .projection import parallel_project
from .reconstruction import simple_backprojection

def run_ct_simulation(phantom_size=256, num_angles=180, detector_size=None):
    """
    Complete CT simulation pipeline
    
    Args:
        phantom_size: Size of phantom image (N x N)
        num_angles: Number of projection angles
        detector_size: Detector elements (default: phantom_size)
        
    Returns:
        tuple: (phantom, sinogram, reconstruction)
    """
    # 1. Generate phantom
    phantom = SheppLoganPhantom(size=phantom_size).generate()
    
    # 2. Create projections
    angles = np.linspace(0, np.pi, num_angles)  # 0 to 180 degrees in radians
    sinogram = parallel_project(
        phantom, 
        angles, 
        detector_size=detector_size or phantom_size
    )
    
    # 3. Reconstruct image
    reconstruction = simple_backprojection(sinogram, angles)
    
    return phantom, sinogram, reconstruction