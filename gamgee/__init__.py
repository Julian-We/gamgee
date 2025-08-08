"""
gamgee - Image analysis and segmentation toolkit

A Python package for biological image analysis, including:
- Cell segmentation with SAM-based models
- Granule and nucleus detection
- Distance measurements and spatial analysis
- Denoising with CARE models
"""

__version__ = "0.1.0"
__author__ = "Julian Wegner"

# Core imports
from .instance import ModelHandler
from .instance import TheCell

# Utility functions
from .utils import upsampling

__all__ = [
    "ModelHandler",
    "TheCell",
    "upsampling",
]