import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ct import run_ct_simulation
import matplotlib.pyplot as plt

# Run complete simulation
phantom, sinogram, sbp_recon, fbp_recon = run_ct_simulation(
    phantom_size=256,
    num_angles=180,
    detector_size=256
)

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('Phantom')

axes[1].imshow(sinogram, cmap='gray', aspect='auto')
axes[1].set_title('Sinogram')
axes[1].set_xlabel('Detector Position')
axes[1].set_ylabel('Projection Angle')

axes[2].imshow(sbp_recon, cmap='gray')
axes[2].set_title('SBP Reconstruction')

axes[3].imshow(fbp_recon, cmap='gray')
axes[3].set_title('FBP Reconstruction')

plt.savefig('ct_simulation_results.png')
plt.show()