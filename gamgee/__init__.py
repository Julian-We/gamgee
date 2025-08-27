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
from .segmenter import SegmentationModel
from .features import catch_error

# Utility functions
from .utils.utils import upsampling
from .utils.denoising import get_shared_memory_info
from .utils.memory_sharer import *

__all__ = [
    "ModelHandler",
    "SegmentationModel",
    "catch_error",
    "upsampling",
    "get_shared_memory_info",
]