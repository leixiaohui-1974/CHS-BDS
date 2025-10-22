"""
Unit tests for PWV Estimation module
"""

import unittest
import numpy as np
from gnss_monitoring.pwv import PWVEstimator


class TestPWVEstimator(unittest.TestCase):
    """Test cases for PWVEstimator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.estimator = PWVEstimator(latitude=34.0, height=150.0)

    def test_initialization(self):
        """Test correct initialization."""
        self.assertAlmostEqual(self.estimator.latitude, np.deg2rad(34.0))
        self.assertEqual(self.estimator.height, 150.0)

    def test_simulate_inputs(self):
        """Test input data simulation."""
        timestamps, ztd, pressure, temp = self.estimator.simulate_inputs(
            num_epochs=100,
            interval_minutes=5
        )

        self.assertEqual(len(timestamps), 100)
        self.assertEqual(len(ztd), 100)
        self.assertEqual(len(pressure), 100)
        self.assertEqual(len(temp), 100)

        # Check reasonable ranges
        self.assertTrue(np.all(ztd > 2.0) and np.all(ztd < 3.0))
        self.assertTrue(np.all(pressure > 900) and np.all(pressure < 1100))
        self.assertTrue(np.all(temp > -10) and np.all(temp < 40))

    def test_calculate_zhd(self):
        """Test ZHD calculation."""
        self.estimator.simulate_inputs()
        zhd = self.estimator.calculate_zhd()

        self.assertIsNotNone(zhd)
        self.assertEqual(len(zhd), len(self.estimator.pressure))

        # ZHD should be positive and reasonable
        self.assertTrue(np.all(zhd > 2.0) and np.all(zhd < 2.5))

    def test_calculate_pwv(self):
        """Test PWV calculation."""
        self.estimator.simulate_inputs()
        self.estimator.calculate_zhd()
        pwv = self.estimator.calculate_pwv()

        self.assertIsNotNone(pwv)
        self.assertEqual(len(pwv), len(self.estimator.ztd))

        # PWV should be positive and reasonable (0-100 mm)
        self.assertTrue(np.all(pwv > 0) and np.all(pwv < 100))

    def test_full_workflow(self):
        """Test complete PWV estimation workflow."""
        self.estimator.simulate_inputs(num_epochs=50)
        self.estimator.calculate_zhd()
        self.estimator.calculate_pwv()

        # Verify all components are calculated
        self.assertIsNotNone(self.estimator.ztd)
        self.assertIsNotNone(self.estimator.zhd)
        self.assertIsNotNone(self.estimator.zwd)
        self.assertIsNotNone(self.estimator.pwv)


if __name__ == '__main__':
    unittest.main()
