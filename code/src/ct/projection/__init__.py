"""
Parallel-Beam Projection Module

Core functionality:
- radon_transform: Pure Radon transform
- parallel_project: Scanner-realistic projection
"""

# Import from their respective modules
from .radon import radon_transform
from .parallel import parallel_project

__all__ = [
    'radon_transform',
    'parallel_project'
]