"""
Computed Tomography (CT) Simulation Package

This package provides tools for:
- Digital phantom generation
- CT projection simulation
- Image reconstruction

Main Components:
- phantom: Tools for creating digital phantoms
- projection: Radon transform and parallel-beam projection
- reconstruction: Reconstruction algorithms
- pipeline: Complete CT simulation workflow
"""

from .phantom import SheppLoganPhantom
from .projection import radon_transform, parallel_project
from .reconstruction import simple_backprojection
from .pipeline import run_ct_simulation

__all__ = [
    'SheppLoganPhantom',
    'radon_transform',
    'parallel_project',
    'simple_backprojection',
    'run_ct_simulation'
]