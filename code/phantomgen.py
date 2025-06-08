import numpy as np

def circlemask(radii, centre_x, centre_y, mu, width, height):
    x, y = np.ogrid[:width, :height]
    mask = (x-centre_x)**2 + (y-centre_y)**2 <= radii**2
    return mask