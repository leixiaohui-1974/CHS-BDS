"""
Unit tests for Configuration module
"""

import unittest
import tempfile
import os
from pathlib import Path
from gnss_monitoring.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()

    def test_get_value(self):
        """Test getting configuration values."""
        # Test dot notation access
        wavelength = self.config.get('gnss_ir.wavelength')
        self.assertIsNotNone(wavelength)

        # Test with default value
        value = self.config.get('nonexistent.key', default=42)
        self.assertEqual(value, 42)

    def test_set_value(self):
        """Test setting configuration values."""
        self.config.set('test.key', 'test_value')
        self.assertEqual(self.config.get('test.key'), 'test_value')

    def test_get_section(self):
        """Test getting configuration section."""
        gnss_ir_section = self.config.get_section('gnss_ir')
        self.assertIsInstance(gnss_ir_section, dict)
        self.assertIn('wavelength', gnss_ir_section)

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            temp_config_path = f.name

        try:
            # Save configuration
            self.config.set('test.value', 123)
            self.config.save_config(temp_config_path)

            # Load configuration
            new_config = Config(temp_config_path)
            self.assertEqual(new_config.get('test.value'), 123)

        finally:
            os.unlink(temp_config_path)

    def test_merge_config(self):
        """Test merging configurations."""
        override = {
            'gnss_ir': {'wavelength': 0.244},  # GPS L2
            'new_section': {'key': 'value'}
        }

        self.config.merge_config(override)

        self.assertEqual(self.config.get('gnss_ir.wavelength'), 0.244)
        self.assertEqual(self.config.get('new_section.key'), 'value')


if __name__ == '__main__':
    unittest.main()
