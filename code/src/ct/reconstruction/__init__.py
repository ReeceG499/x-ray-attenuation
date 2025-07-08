"""
CT Reconstruction Module

Core functionality:
- simple_backprojection: Unfiltered backprojection (basic reconstruction)
"""

from .sbp import simple_backprojection

__all__ = [
    'simple_backprojection'
]