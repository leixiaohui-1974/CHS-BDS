"""
CHS-BDS: GNSS Comprehensive Monitoring System
==============================================

A comprehensive GNSS-based environmental monitoring system integrating:
- GNSS-IR water level monitoring
- High-precision deformation monitoring
- Precipitable Water Vapor (PWV) estimation
- Rainfall prediction

Author: Lei Xiaohui
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Lei Xiaohui"
__email__ = "leixiaohui@example.com"

from .gnss_ir import GNSSIRWaterLevel
from .deformation import DeformationMonitor
from .pwv import PWVEstimator
from .rainfall_model import RainfallPredictor

__all__ = [
    'GNSSIRWaterLevel',
    'DeformationMonitor',
    'PWVEstimator',
    'RainfallPredictor',
]
