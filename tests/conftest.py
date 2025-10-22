"""
pytest configuration and fixtures
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing


@pytest.fixture
def sample_elevation_data():
    """Provide sample satellite elevation data."""
    return np.linspace(5, 30, 100)


@pytest.fixture
def sample_coordinates():
    """Provide sample ECEF coordinates."""
    return {
        'ref': [3275650.0, 553640.0, 5201550.0],
        'mon': [3275750.0, 553650.0, 5201545.0]
    }


@pytest.fixture
def sample_satellite_positions():
    """Provide sample satellite positions."""
    return {
        'G01': [20000000, 10000000, 15000000],
        'G05': [22000000, -5000000, 18000000],
        'G12': [15000000, 15000000, 20000000],
    }
