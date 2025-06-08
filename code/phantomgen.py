import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import loading as l
import spekpy as sp

def circlemask(radii, centre_x, centre_y, mu, width, height):
    x, y = np.ogrid[:width, :height]
    mask = (x-centre_x)**2 + (y-centre_y)**2 <= radii**2
    return mask

def shepplogan(energies, mu_tissue, mu_bone):
    E = len(energies)

    # Generate Shepp-Logan phantom (returns 128x128 by default)
    phantom_2d = shepp_logan_phantom()

    # Create 3D phantom by replicating for each energy level
    phantom = np.zeros((400, 400, E))
    for e in range(E):
        phantom[:, :, e] = np.where(phantom_2d == 0, 0,  # background
                        np.where(phantom_2d == 0.1, mu_tissue[e]*0.5,
                        np.where(phantom_2d == 0.2, mu_tissue[e]*0.8,
                        np.where(phantom_2d >= 0.3, mu_bone[e], mu_tissue[e]))))
    
    return phantom