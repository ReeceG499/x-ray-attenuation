import numpy as np
from .base import Phantom

class SheppLoganPhantom(Phantom):
    def generate(self):
        """Generate Shepp-Logan phantom"""
        ellipses = [
            # Format: (intensity, x_center, y_center, a, b, theta_degrees)
            (2.0,   0.0,   0.0,    0.69,  0.92,  0),
            (-0.8,  0.0,  -0.0184, 0.6624, 0.874, 0),
            (-0.2,  0.22,  0.0,    0.11,  0.31, -18),
            (-0.2, -0.22,  0.0,    0.16,  0.41,  18),
            (0.1,   0.0,   0.35,   0.21,  0.25,  0),
            (0.1,   0.0,   0.1,    0.046, 0.046, 0),
            (0.1,  -0.08, -0.605,  0.046, 0.023, 0),
            (0.1,   0.0,  -0.605,  0.023, 0.023, 0),
            (0.1,   0.06, -0.605,  0.046, 0.023, 90)
        ]
        
        for params in ellipses:
            intensity, xc, yc, a, b, theta = params
            self.add_ellipse(intensity, xc, yc, a, b, np.radians(theta))
            
        return self.image