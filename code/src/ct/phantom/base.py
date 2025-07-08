import numpy as np
from abc import ABC, abstractmethod

class Phantom(ABC):
    def __init__(self, size=256):
        self.size = size
        self.image = np.zeros((size, size), dtype=np.float32)
        
    @abstractmethod
    def generate(self):
        pass
        
    def add_ellipse(self, intensity, x_center, y_center, a, b, theta=0):
        """Add an ellipse to the phantom"""
        xx, yy = np.mgrid[:self.size, :self.size]
        x_norm = (xx - self.size/2) / (self.size/2)
        y_norm = (yy - self.size/2) / (self.size/2)
        
        x_rot = (x_norm-x_center)*np.cos(theta) + (y_norm-y_center)*np.sin(theta)
        y_rot = -(x_norm-x_center)*np.sin(theta) + (y_norm-y_center)*np.cos(theta)
        
        mask = (x_rot/a)**2 + (y_rot/b)**2 <= 1
        self.image[mask] = intensity