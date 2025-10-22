"""
CHS-BDS: GNSS Comprehensive Monitoring System
==============================================

A comprehensive GNSS-based environmental monitoring system integrating:
- GNSS-IR water level monitoring
- High-precision deformation monitoring
- Precipitable Water Vapor (PWV) estimation
- Rainfall prediction
- Data loading and export utilities
- Logging and configuration management

Author: Lei Xiaohui
License: MIT
"""

__version__ = "0.2.0"
__author__ = "Lei Xiaohui"
__email__ = "leixiaohui@example.com"

# Core monitoring modules
from .gnss_ir import GNSSIRWaterLevel
from .deformation import DeformationMonitor
from .pwv import PWVEstimator
from .rainfall_model import RainfallPredictor

# Data handling modules
from .data_loader import DataLoader, DataGenerator
from .data_export import DataExporter, ResultsArchiver

# Configuration and logging
from .config import Config, get_config
from .logger import setup_logger, get_logger

# Exceptions
from .exceptions import CHSBDSException

__all__ = [
    # Core modules
    'GNSSIRWaterLevel',
    'DeformationMonitor',
    'PWVEstimator',
    'RainfallPredictor',
    # Data handling
    'DataLoader',
    'DataGenerator',
    'DataExporter',
    'ResultsArchiver',
    # Configuration
    'Config',
    'get_config',
    # Logging
    'setup_logger',
    'get_logger',
    # Exceptions
    'CHSBDSException',
]
