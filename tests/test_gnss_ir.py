"""
Unit tests for GNSS-IR module
"""

import unittest
import numpy as np
from gnss_monitoring.gnss_ir import GNSSIRWaterLevel


class TestGNSSIRWaterLevel(unittest.TestCase):
    """Test cases for GNSSIRWaterLevel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.elevation = np.linspace(5, 30, 100)
        self.azimuth = np.linspace(120, 150, 100)
        self.analyzer = GNSSIRWaterLevel(
            satellite_elevation=self.elevation,
            satellite_azimuth=self.azimuth,
            antenna_height=1.5,
            wavelength=0.1903
        )

    def test_initialization(self):
        """Test correct initialization of GNSSIRWaterLevel."""
        self.assertEqual(self.analyzer.antenna_height, 1.5)
        self.assertEqual(self.analyzer.wavelength, 0.1903)
        self.assertIsNone(self.analyzer.snr_data)

    def test_simulate_snr_data(self):
        """Test SNR data simulation."""
        true_height = 4.5
        self.analyzer._simulate_snr_data(true_height)

        self.assertIsNotNone(self.analyzer.snr_data)
        self.assertEqual(len(self.analyzer.snr_data), len(self.elevation))

    def test_preprocess_snr(self):
        """Test SNR preprocessing."""
        self.analyzer._simulate_snr_data(4.5)
        self.analyzer.preprocess_snr()

        self.assertIsNotNone(self.analyzer.snr_residual)
        self.assertEqual(len(self.analyzer.snr_residual), len(self.elevation))

    def test_analyze_frequency(self):
        """Test frequency analysis."""
        true_height = 4.5
        self.analyzer._simulate_snr_data(true_height)
        self.analyzer.preprocess_snr()
        self.analyzer.analyze_frequency()

        self.assertIsNotNone(self.analyzer.reflector_height)
        # Check that estimated height is reasonably close to true height
        error = abs(true_height - self.analyzer.reflector_height)
        self.assertLess(error, 1.0, "Height estimation error too large")

    def test_run_analysis(self):
        """Test complete analysis workflow."""
        true_height = 5.0
        estimated_height = self.analyzer.run_analysis(true_height, plot=False)

        self.assertIsNotNone(estimated_height)
        self.assertIsInstance(estimated_height, (float, np.floating))

        # Verify reasonable accuracy
        error = abs(true_height - estimated_height)
        self.assertLess(error, 1.5, "Analysis error too large")


if __name__ == '__main__':
    unittest.main()
