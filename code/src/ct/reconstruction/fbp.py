import numpy as np
from scipy.fft import fft, ifft
from .sbp import simple_backprojection
from .filters import get_filter

def filtered_backprojection(sinogram: np.ndarray, angles: np.ndarray, filter_name: str = "ramlak") -> np.ndarray:
    """
    Perform filtered backprojection reconstruction.

    Args:
        sinogram: 2D numpy array of shape (num_detectors, num_angles)
        angles: 1D numpy array of projection angles in radians
        filter_name: Name of the filter to apply (e.g., "ramlak", "hann")

    Returns:
        2D numpy array representing the reconstructed image.
    """
    num_detectors = sinogram.shape[1]

    # Ensure FFT size is a power of 2 for speed
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * num_detectors))))
    pad_width = ((0, 0), (0, projection_size_padded - num_detectors))
    padded_sinogram = np.pad(sinogram, pad_width, mode="constant", constant_values=0)

    # Get the filter
    filter_func = get_filter(filter_name)
    filter_kernel = filter_func(projection_size_padded)

    # Apply filter in frequency domain
    sino_fft = fft(padded_sinogram, axis=1)

    filtered_fft = sino_fft * filter_kernel[np.newaxis, :]
    filtered_sinogram_padded = np.real(ifft(filtered_fft, axis=1)[:num_detectors, :])
    filtered_sinogram = filtered_sinogram_padded[:, :num_detectors]

    # Backproject
    return simple_backprojection(filtered_sinogram, angles)
