import numpy as np
from .base import Filter

class RamLakFilter(Filter):
    def __call__(self, size: int) -> np.ndarray:
        freqs = np.fft.fftfreq(size) * size
        filter_kernel = np.abs(freqs)
        filter_kernel[0] = 0
        return filter_kernel