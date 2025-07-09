from abc import ABC, abstractmethod
import numpy as np

class Filter(ABC):
    @abstractmethod
    def __call__(self, size: int) -> np.ndarray:
        """Return 1D frequency-domain filter of given size."""
        pass