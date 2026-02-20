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
from .denoising_interface import denoise_with_care
from .instance.thecell import TheCell
from .instance.modelhandler import ModelHandler
from .instance.marker import Marker


# Utility functions
from .utils.utils import upsampling
from .utils.denoising import get_shared_memory_info, encode_memmap_info, get_memmap_info


__all__ = [
    "ModelHandler",
    "SegmentationModel",
    "catch_error",
    "upsampling",
    "get_shared_memory_info",
]