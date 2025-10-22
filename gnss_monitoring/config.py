"""
Configuration Management Module
================================

Handles loading and parsing of configuration files for the CHS-BDS system.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for CHS-BDS system.

    Loads and manages configuration from YAML files with support for
    environment variable overrides and default values.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration YAML file. If None, searches for
                        config.yaml in standard locations.
        """
        self._config: Dict[str, Any] = {}
        self._config_path: Optional[Path] = None

        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_config()

    def _load_default_config(self):
        """Search for and load configuration from standard locations."""
        search_paths = [
            Path.cwd() / 'config.yaml',
            Path.cwd() / 'config' / 'config.yaml',
            Path(__file__).parent.parent / 'config.yaml',
            Path.home() / '.chs-bds' / 'config.yaml',
        ]

        for path in search_paths:
            if path.exists():
                self.load_config(str(path))
                return

        # If no config file found, use defaults
        print("Warning: No configuration file found. Using default settings.")
        self._load_builtin_defaults()

    def load_config(self, config_path: str):
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is malformed.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        self._config_path = path
        print(f"✅ Configuration loaded from: {config_path}")

    def _load_builtin_defaults(self):
        """Load built-in default configuration."""
        self._config = {
            'global': {
                'log_level': 'INFO',
                'output_dir': './output',
                'data_dir': './data',
                'enable_plots': True,
            },
            'gnss_ir': {
                'wavelength': 0.1903,
                'antenna_height': 1.5,
                'min_elevation': 5,
                'max_elevation': 30,
            },
            'deformation': {
                'wavelength': 0.1903,
            },
            'pwv': {
                'station': {'latitude': 34.0, 'height': 150.0},
                'num_epochs': 288,
                'interval_minutes': 5,
            },
            'rainfall': {
                'lookback_days': 30,
                'interval_minutes': 10,
                'look_ahead_minutes': 30,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'gnss_ir.wavelength')
            default: Default value if key not found.

        Returns:
            Configuration value or default.

        Examples:
            >>> config.get('gnss_ir.wavelength')
            0.1903
            >>> config.get('gnss_ir.min_elevation')
            5
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation.
            value: Value to set.
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name (e.g., 'gnss_ir', 'pwv').

        Returns:
            Dictionary containing all configuration in that section.
        """
        return self._config.get(section, {})

    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to a YAML file.

        Args:
            output_path: Path to save configuration. If None, overwrites original.
        """
        if output_path is None:
            if self._config_path is None:
                raise ValueError("No config path specified and no original path to overwrite")
            output_path = str(self._config_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Configuration saved to: {output_path}")

    def merge_config(self, other_config: Dict[str, Any]):
        """
        Merge another configuration dictionary into this one.

        Args:
            other_config: Dictionary to merge.
        """
        self._deep_merge(self._config, other_config)

    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def __repr__(self) -> str:
        return f"Config(path={self._config_path}, sections={list(self._config.keys())})"


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to configuration file. Only used on first call.

    Returns:
        Global Config instance.
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def reload_config(config_path: Optional[str] = None):
    """
    Reload the global configuration.

    Args:
        config_path: Path to configuration file.
    """
    global _global_config
    _global_config = Config(config_path)


if __name__ == '__main__':
    # Test the configuration system
    print("Testing Configuration System...")
    print("-" * 50)

    # Load config
    config = Config()

    # Test getting values
    print(f"GNSS-IR Wavelength: {config.get('gnss_ir.wavelength')} m")
    print(f"PWV Station Latitude: {config.get('pwv.station.latitude')}°")
    print(f"Output Directory: {config.get('global.output_dir')}")

    # Test getting section
    gnss_ir_config = config.get_section('gnss_ir')
    print(f"\nGNSS-IR Configuration: {gnss_ir_config}")

    # Test setting value
    config.set('gnss_ir.antenna_height', 2.0)
    print(f"\nUpdated Antenna Height: {config.get('gnss_ir.antenna_height')} m")

    print("\n✅ Configuration system test completed!")
