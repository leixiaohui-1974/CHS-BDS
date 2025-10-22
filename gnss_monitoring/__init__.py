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
- Database integration (PostgreSQL/TimescaleDB)
- RESTful API (FastAPI)
- Real-time WebSocket communication
- PDF report generation
- User authentication and authorization
- Advanced 3D visualization
- System monitoring and metrics

Author: Lei Xiaohui
License: MIT
"""

__version__ = "0.4.0"
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

# Phase 4: Database integration
from .database import DatabaseManager, DatabaseConfig, get_database_manager

# Phase 4: RESTful API (optional)
try:
    from .api import CHSBDSAPI, create_api, run_api_server
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Phase 4: RINEX parser
from .rinex_parser import RINEXParser, parse_rinex_file, extract_snr_from_rinex

# Phase 4: PDF reports
try:
    from .report_generator import (
        PDFReportGenerator,
        GNSSMonitoringReport,
        generate_monitoring_report
    )
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Phase 4: WebSocket server (optional)
try:
    from .websocket_server import (
        WebSocketServer,
        WebSocketConnectionManager,
        get_websocket_server,
        Topics
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Phase 4: Authentication (optional)
try:
    from .auth import (
        AuthenticationService,
        UserManager,
        JWTManager,
        User,
        Token,
        Role,
        Permission,
        create_auth_service
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Phase 4: Advanced visualization (optional)
try:
    from .advanced_viz import (
        Advanced3DVisualizer,
        InteractiveTimeSeriesVisualizer,
        HeatmapVisualizer,
        MultiPanelVisualizer,
        save_plotly_figure
    )
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False

# Phase 4: System monitoring
from .monitoring import (
    MonitoringService,
    MetricsCollector,
    SystemMonitor,
    ApplicationMonitor,
    HealthChecker,
    get_monitoring_service
)

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
    # Phase 4: Database
    'DatabaseManager',
    'DatabaseConfig',
    'get_database_manager',
    # Phase 4: RINEX
    'RINEXParser',
    'parse_rinex_file',
    'extract_snr_from_rinex',
    # Phase 4: Monitoring
    'MonitoringService',
    'MetricsCollector',
    'SystemMonitor',
    'ApplicationMonitor',
    'HealthChecker',
    'get_monitoring_service',
    # Configuration
    'Config',
    'get_config',
    # Logging
    'setup_logger',
    'get_logger',
    # Exceptions
    'CHSBDSException',
]

# Add optional modules if available
if DASHBOARD_AVAILABLE:
    __all__.extend(['CHSBDSDashboard', 'create_dashboard_app'])

if API_AVAILABLE:
    __all__.extend(['CHSBDSAPI', 'create_api', 'run_api_server'])

if PDF_AVAILABLE:
    __all__.extend(['PDFReportGenerator', 'GNSSMonitoringReport', 'generate_monitoring_report'])

if WEBSOCKET_AVAILABLE:
    __all__.extend(['WebSocketServer', 'WebSocketConnectionManager', 'get_websocket_server', 'Topics'])

if AUTH_AVAILABLE:
    __all__.extend([
        'AuthenticationService', 'UserManager', 'JWTManager',
        'User', 'Token', 'Role', 'Permission', 'create_auth_service'
    ])

if ADVANCED_VIZ_AVAILABLE:
    __all__.extend([
        'Advanced3DVisualizer', 'InteractiveTimeSeriesVisualizer',
        'HeatmapVisualizer', 'MultiPanelVisualizer', 'save_plotly_figure'
    ])
