import numpy as np
import matplotlib.pyplot as plt
import spekpy as sp
import phantomgen as pg
import loading as l
from scipy import ndimage

s = sp.Spek(kvp=150,th=12)
energies = s.get_k()
mu_tissue = l.load_mu_data(r"code\soft_tissue.csv", 1.06, energies)
mu_bone = l.load_mu_data(r"code\bone.csv", 1.85, energies)

phantom = pg.shepplogan(energies, mu_tissue, mu_bone)

# Visualize one energy slice
energy_idx = 50  # example energy bin index
plt.imshow(phantom[:, :, energy_idx], cmap='gray')
plt.title(f"Shepp-Logan μ map at energy bin {energy_idx}")
plt.colorbar(label="μ (cm⁻¹)")
plt.show()

# Projection calculation
voxel_thickness_cm = 0.1  # Assume each pixel is 0.1cm thick
mu_total = phantom.sum(axis=0)

I0 = s.get_spk() 
I = I0 * np.exp(-mu_total * voxel_thickness_cm)

detector_readout = I.sum(axis=1)

plt.plot(detector_readout)
plt.title("1D Detector Readout (Shepp-Logan Phantom)")
plt.xlabel("Detector Pixel Index")
plt.ylabel("Photon Counts (arbitrary)")
plt.grid(True)
plt.show()

width = 400
height = 400

angles = np.linspace(0, 180, num=180)
sinogram = np.zeros((len(angles), width))

for i, angle in enumerate(angles):
    # Rotate all energy slices at once using 3D rotation
    rotated_phantom = ndimage.rotate(phantom, angle, axes=(1, 0), reshape=False, order=1)
    
    # Sum along y-axis for all energy slices (results in (width, E) array)
    projection_per_energy = rotated_phantom.sum(axis=1)
    
    # Apply attenuation and sum
    I_rotated = I0 * np.exp(-projection_per_energy * voxel_thickness_cm)
    sinogram[i] = I_rotated.sum(axis=1)  # sum over energy bins

sinogram = np.array(sinogram)  # shape: (n_angles, W)

plt.imshow(sinogram.T, cmap='gray', aspect='auto')
plt.title("Simulated Sinogram")
plt.ylabel("Detector Pixel")
plt.xlabel("Projection Angle")
plt.colorbar()
plt.show()

# Initialize reconstruction
reconstruction = np.zeros((width, height))

# Backproject each angle's projection
for i, angle in enumerate(angles):
    # Expand the 1D projection into a 2D image by smearing along the angle
    projection = sinogram[i]
    expanded = np.tile(projection, (width, 1))  # Repeat along y-axis
    
    # Rotate back to original angle and accumulate
    rotated_back = ndimage.rotate(expanded, -angle, reshape=False, order=1)
    reconstruction += rotated_back

# Normalize
reconstruction /= len(angles)

plt.imshow(reconstruction, cmap='gray')
plt.title("Simple Backprojection Reconstruction")
plt.colorbar()
plt.show()