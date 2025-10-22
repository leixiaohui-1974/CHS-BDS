"""
CHS-BDS: GNSS Comprehensive Monitoring System
==============================================

A comprehensive GNSS-based environmental monitoring system integrating:
- GNSS-IR water level monitoring
- High-precision deformation monitoring
- Precipitable Water Vapor (PWV) estimation
- Rainfall prediction
- Data loading and export utilities
- Quality control and validation
- Performance optimization
- Alert and notification system
- Automated monitoring
- Web dashboard visualization

Author: Lei Xiaohui
License: MIT
"""

__version__ = "0.3.0"
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

# Quality control
from .quality_control import DataQualityChecker, QualityMetrics

# Performance tools
from .performance import (
    PerformanceProfiler,
    PerformanceTimer,
    BatchProcessor,
    measure_performance
)

# Alert system
from .alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    GNSSAlertRules
)

# Automation
from .automation import AutomatedMonitor, BatchAnalyzer, SimpleAPI

# Dashboard (optional)
try:
    from .dashboard import CHSBDSDashboard, create_dashboard_app
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

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
    # Quality control
    'DataQualityChecker',
    'QualityMetrics',
    # Performance
    'PerformanceProfiler',
    'PerformanceTimer',
    'BatchProcessor',
    'measure_performance',
    # Alerts
    'Alert',
    'AlertLevel',
    'AlertManager',
    'GNSSAlertRules',
    # Automation
    'AutomatedMonitor',
    'BatchAnalyzer',
    'SimpleAPI',
    # Configuration
    'Config',
    'get_config',
    # Logging
    'setup_logger',
    'get_logger',
    # Exceptions
    'CHSBDSException',
]

# Add dashboard if available
if DASHBOARD_AVAILABLE:
    __all__.extend(['CHSBDSDashboard', 'create_dashboard_app'])
