import numpy as np
from .phantom.shepplogan import SheppLoganPhantom
from .projection.parallel import parallel_project
from .reconstruction.sbp import simple_backprojection
from .reconstruction.fbp import filtered_backprojection

def run_ct_simulation(phantom_size=256, num_angles=180, detector_size=None, filter_name="ramlak"):
    """
    Complete CT simulation pipeline
    
    Args:
        phantom_size: Size of phantom image (N x N)
        num_angles: Number of projection angles
        detector_size: Detector elements (default: phantom_size)
        
    Returns:
        tuple: (phantom, sinogram, sbp reconstruction, fbp reconstruction)
    """
    # 1. Generate phantom
    phantom = SheppLoganPhantom(size=phantom_size).generate()
    
    # 2. Create projections
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)  # 0 to 180 degrees in radians
    sinogram_raw = parallel_project(
        phantom, 
        angles, 
        detector_size=detector_size or phantom_size
    )
    sinogram = sinogram_raw.T

    sbp_reconstruction = simple_backprojection(sinogram, angles)
    fbp_reconstruction = filtered_backprojection(sinogram, angles, filter_name=filter_name)
    
    return phantom, sinogram, sbp_reconstruction, fbp_reconstruction