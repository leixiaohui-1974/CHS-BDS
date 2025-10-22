"""
Unit tests for Deformation Monitoring module
"""

import unittest
import numpy as np
from gnss_monitoring.deformation import DeformationMonitor


class TestDeformationMonitor(unittest.TestCase):
    """Test cases for DeformationMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ref_pos = [3275650.0, 553640.0, 5201550.0]
        self.mon_pos = [3275750.0, 553650.0, 5201545.0]
        self.satellite_positions = {
            'G01': [20000000, 10000000, 15000000],
            'G05': [22000000, -5000000, 18000000],
            'G12': [15000000, 15000000, 20000000],
        }

        self.monitor = DeformationMonitor(
            ref_pos=self.ref_pos,
            mon_pos=self.mon_pos,
            satellite_positions=self.satellite_positions
        )

    def test_initialization(self):
        """Test correct initialization."""
        np.testing.assert_array_equal(self.monitor.ref_pos, self.ref_pos)
        np.testing.assert_array_equal(self.monitor.mon_pos, self.mon_pos)
        self.assertEqual(len(self.monitor.satellite_positions), 3)

    def test_simulate_observations(self):
        """Test observation simulation."""
        obs = self.monitor._simulate_phase_observations()

        self.assertIn('ref', obs)
        self.assertIn('mon', obs)
        self.assertEqual(len(obs['ref']), 3)
        self.assertEqual(len(obs['mon']), 3)

    def test_double_differencing(self):
        """Test double differencing calculation."""
        self.monitor._simulate_phase_observations()
        dd = self.monitor.perform_double_differencing()

        # Should have n-1 double differences for n satellites
        self.assertEqual(len(dd), 2)

        # Check structure
        self.assertIn('dd_value', dd[0])
        self.assertIn('sat1', dd[0])
        self.assertIn('sat2', dd[0])

    def test_solve_baseline(self):
        """Test baseline solution."""
        self.monitor._simulate_phase_observations()
        self.monitor.perform_double_differencing()

        approx_pos = [self.mon_pos[0] + 0.1, self.mon_pos[1] - 0.1, self.mon_pos[2] + 0.05]
        estimated_baseline = self.monitor.solve_baseline(approx_pos)

        self.assertEqual(len(estimated_baseline), 3)

        # Check accuracy (should be within a few meters)
        error = np.linalg.norm(self.monitor.true_baseline - estimated_baseline)
        self.assertLess(error, 10.0, "Baseline error too large")


if __name__ == '__main__':
    unittest.main()
