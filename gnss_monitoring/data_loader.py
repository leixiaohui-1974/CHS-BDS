"""
Data Loader Module
==================

Handles loading and preprocessing of GNSS data from various formats.
Supports RINEX, CSV, JSON, and other common data formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import json

from .logger import get_logger
from .exceptions import DataLoadError, DataFormatError, MissingDataError


logger = get_logger(__name__)


class DataLoader:
    """
    Universal data loader for GNSS monitoring data.

    Supports multiple data formats:
    - RINEX observation files (simplified parser)
    - CSV files
    - JSON files
    - Pandas DataFrames
    """

    def __init__(self):
        """Initialize data loader."""
        self.data = None
        self.metadata = {}
        logger.info("DataLoader initialized")

    def load_csv(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        skiprows: int = 0,
        delimiter: str = ','
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file.
            columns: Column names to use. If None, uses header from file.
            skiprows: Number of rows to skip at the beginning.
            delimiter: Column delimiter.

        Returns:
            DataFrame with loaded data.

        Raises:
            DataLoadError: If file cannot be loaded.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataLoadError(f"File not found: {file_path}")

            logger.info(f"Loading CSV file: {file_path}")

            # Load data
            if columns:
                df = pd.read_csv(
                    path,
                    names=columns,
                    skiprows=skiprows,
                    delimiter=delimiter
                )
            else:
                df = pd.read_csv(path, skiprows=skiprows, delimiter=delimiter)

            logger.info(f"Loaded {len(df)} rows from {file_path}")
            self.data = df
            self.metadata['source'] = str(path)
            self.metadata['format'] = 'csv'
            self.metadata['rows'] = len(df)
            self.metadata['columns'] = list(df.columns)

            return df

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise DataLoadError(f"Failed to load CSV file: {e}")

    def load_json(self, file_path: str) -> Dict:
        """
        Load data from JSON file.

        Args:
            file_path: Path to JSON file.

        Returns:
            Dictionary with loaded data.

        Raises:
            DataLoadError: If file cannot be loaded.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataLoadError(f"File not found: {file_path}")

            logger.info(f"Loading JSON file: {file_path}")

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.data = data
            self.metadata['source'] = str(path)
            self.metadata['format'] = 'json'

            logger.info(f"Loaded JSON data from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise DataLoadError(f"Failed to load JSON file: {e}")

    def load_rinex_obs(
        self,
        file_path: str,
        satellites: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load GNSS observation data from simplified RINEX format.

        Note: This is a simplified parser for demonstration.
        For production use, consider using specialized libraries like:
        - georinex
        - hatanaka
        - teqc

        Args:
            file_path: Path to RINEX observation file.
            satellites: List of satellites to load (e.g., ['G01', 'G05']).
                       If None, loads all satellites.

        Returns:
            Dictionary mapping satellite IDs to DataFrames with observations.

        Raises:
            DataLoadError: If file cannot be loaded.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise DataLoadError(f"File not found: {file_path}")

            logger.info(f"Loading RINEX file: {file_path}")
            logger.warning("Using simplified RINEX parser - for production use specialized library")

            # This is a placeholder implementation
            # Real RINEX parsing is complex and requires specialized libraries
            observations = self._parse_rinex_simplified(path, satellites)

            self.data = observations
            self.metadata['source'] = str(path)
            self.metadata['format'] = 'rinex'
            self.metadata['satellites'] = list(observations.keys())

            logger.info(f"Loaded observations for {len(observations)} satellites")
            return observations

        except Exception as e:
            logger.error(f"Failed to load RINEX: {e}")
            raise DataLoadError(f"Failed to load RINEX file: {e}")

    def _parse_rinex_simplified(
        self,
        path: Path,
        satellites: Optional[List[str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Simplified RINEX parser (demonstration only).

        For real applications, use specialized RINEX libraries.
        """
        # Placeholder: Generate simulated RINEX-like data
        logger.info("Generating simulated RINEX data for demonstration")

        if satellites is None:
            satellites = ['G01', 'G05', 'G12', 'G21']

        observations = {}
        epochs = 100

        for sat in satellites:
            # Simulate observations
            data = {
                'epoch': range(epochs),
                'elevation': np.random.uniform(10, 80, epochs),
                'azimuth': np.random.uniform(0, 360, epochs),
                'snr': np.random.uniform(35, 55, epochs),
                'phase': np.random.uniform(0, 1e8, epochs),
                'pseudorange': np.random.uniform(2e7, 2.5e7, epochs),
            }
            observations[sat] = pd.DataFrame(data)

        return observations

    def load_snr_data(
        self,
        file_path: str,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Load SNR (Signal-to-Noise Ratio) data for GNSS-IR analysis.

        Args:
            file_path: Path to SNR data file (CSV format).
            columns: Column names. Default: ['elevation', 'azimuth', 'snr']

        Returns:
            DataFrame with SNR data.
        """
        if columns is None:
            columns = ['elevation', 'azimuth', 'snr']

        logger.info(f"Loading SNR data from {file_path}")
        df = self.load_csv(file_path, columns=columns)

        # Validate required columns
        required = ['elevation', 'snr']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise DataFormatError(f"Missing required columns: {missing}")

        return df

    def load_coordinates(
        self,
        file_path: str,
        format: str = 'csv'
    ) -> Dict[str, np.ndarray]:
        """
        Load station coordinates (for deformation monitoring).

        Args:
            file_path: Path to coordinates file.
            format: File format ('csv' or 'json').

        Returns:
            Dictionary with station coordinates.
            Format: {'station_name': [X, Y, Z]}
        """
        logger.info(f"Loading station coordinates from {file_path}")

        if format == 'csv':
            df = self.load_csv(file_path)
            # Expect columns: station, X, Y, Z
            if not all(col in df.columns for col in ['station', 'X', 'Y', 'Z']):
                raise DataFormatError("CSV must have columns: station, X, Y, Z")

            coords = {}
            for _, row in df.iterrows():
                coords[row['station']] = np.array([row['X'], row['Y'], row['Z']])

            return coords

        elif format == 'json':
            data = self.load_json(file_path)
            # Convert to numpy arrays
            coords = {name: np.array(pos) for name, pos in data.items()}
            return coords

        else:
            raise DataFormatError(f"Unsupported format: {format}")

    def load_meteorological_data(
        self,
        file_path: str,
        time_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Load meteorological data (for PWV estimation).

        Args:
            file_path: Path to meteorological data file.
            time_column: Name of timestamp column.

        Returns:
            DataFrame with meteorological data.
            Expected columns: timestamp, temperature, pressure, humidity
        """
        logger.info(f"Loading meteorological data from {file_path}")

        df = self.load_csv(file_path)

        # Validate required columns
        required = ['temperature', 'pressure']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise DataFormatError(f"Missing required columns: {missing}")

        # Parse timestamps if present
        if time_column in df.columns:
            try:
                df[time_column] = pd.to_datetime(df[time_column])
                df.set_index(time_column, inplace=True)
            except Exception as e:
                logger.warning(f"Could not parse timestamps: {e}")

        return df

    def validate_data(
        self,
        data: Union[pd.DataFrame, Dict],
        required_columns: List[str] = None,
        value_ranges: Dict[str, Tuple[float, float]] = None
    ) -> bool:
        """
        Validate loaded data.

        Args:
            data: Data to validate.
            required_columns: List of required column names.
            value_ranges: Dictionary of column name to (min, max) tuples.

        Returns:
            True if data is valid.

        Raises:
            DataFormatError: If validation fails.
        """
        logger.info("Validating data")

        if isinstance(data, pd.DataFrame):
            # Check required columns
            if required_columns:
                missing = [col for col in required_columns if col not in data.columns]
                if missing:
                    raise DataFormatError(f"Missing required columns: {missing}")

            # Check value ranges
            if value_ranges:
                for col, (min_val, max_val) in value_ranges.items():
                    if col in data.columns:
                        if data[col].min() < min_val or data[col].max() > max_val:
                            raise DataFormatError(
                                f"Column {col} has values outside range [{min_val}, {max_val}]"
                            )

            # Check for NaN values
            nan_cols = data.columns[data.isna().any()].tolist()
            if nan_cols:
                logger.warning(f"Columns with NaN values: {nan_cols}")

        logger.info("Data validation passed")
        return True

    def get_metadata(self) -> Dict:
        """Get metadata about loaded data."""
        return self.metadata.copy()

    def save_to_csv(self, output_path: str, data: pd.DataFrame = None):
        """
        Save data to CSV file.

        Args:
            output_path: Output file path.
            data: Data to save. If None, uses self.data.
        """
        if data is None:
            if not isinstance(self.data, pd.DataFrame):
                raise DataFormatError("Cannot save non-DataFrame data as CSV")
            data = self.data

        logger.info(f"Saving data to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Data saved successfully")


class DataGenerator:
    """
    Generate sample/test data for demonstration and testing.
    """

    @staticmethod
    def generate_snr_data(
        output_path: str,
        num_epochs: int = 200,
        elevation_range: Tuple[float, float] = (5, 30)
    ):
        """
        Generate sample SNR data for GNSS-IR testing.

        Args:
            output_path: Output CSV file path.
            num_epochs: Number of data points.
            elevation_range: (min, max) elevation angles in degrees.
        """
        logger.info(f"Generating {num_epochs} SNR data points")

        elevation = np.linspace(elevation_range[0], elevation_range[1], num_epochs)
        azimuth = np.random.uniform(120, 150, num_epochs)

        # Simulate SNR with multipath effects
        base_snr = 45
        multipath = 5 * np.sin(4 * np.pi * np.sin(np.deg2rad(elevation)))
        noise = np.random.normal(0, 1, num_epochs)
        snr = base_snr + multipath + noise

        df = pd.DataFrame({
            'elevation': elevation,
            'azimuth': azimuth,
            'snr': snr
        })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"SNR data saved to {output_path}")

    @staticmethod
    def generate_coordinates_data(output_path: str):
        """
        Generate sample station coordinates.

        Args:
            output_path: Output JSON file path.
        """
        logger.info("Generating station coordinates")

        coordinates = {
            'STATION_REF': [3275650.0, 553640.0, 5201550.0],
            'STATION_MON': [3275750.0, 553650.0, 5201545.0],
            'STATION_A': [3275600.0, 553700.0, 5201500.0],
            'STATION_B': [3275800.0, 553600.0, 5201600.0],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(coordinates, f, indent=2)

        logger.info(f"Coordinates saved to {output_path}")

    @staticmethod
    def generate_meteorological_data(
        output_path: str,
        num_days: int = 7
    ):
        """
        Generate sample meteorological data.

        Args:
            output_path: Output CSV file path.
            num_days: Number of days of data.
        """
        logger.info(f"Generating {num_days} days of meteorological data")

        hours = num_days * 24
        timestamps = pd.date_range(
            start='2025-01-01',
            periods=hours,
            freq='H'
        )

        # Simulate temperature with diurnal cycle
        base_temp = 15
        diurnal = 8 * np.sin(2 * np.pi * np.arange(hours) / 24 - np.pi/2)
        temperature = base_temp + diurnal + np.random.normal(0, 1, hours)

        # Simulate pressure
        pressure = 1013 + np.random.normal(0, 5, hours)

        # Simulate humidity
        humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(hours) / 24) + np.random.normal(0, 5, hours)
        humidity = np.clip(humidity, 0, 100)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'pressure': pressure,
            'humidity': humidity
        })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Meteorological data saved to {output_path}")


if __name__ == '__main__':
    # Test data loader
    print("Testing Data Loader Module...")
    print("-" * 50)

    # Create test data directory
    test_dir = Path('./data/examples')
    test_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample data
    print("\n1. Generating sample data...")
    DataGenerator.generate_snr_data(str(test_dir / 'snr_data.csv'))
    DataGenerator.generate_coordinates_data(str(test_dir / 'coordinates.json'))
    DataGenerator.generate_meteorological_data(str(test_dir / 'meteo_data.csv'))

    # Load and validate data
    print("\n2. Testing data loading...")
    loader = DataLoader()

    # Test SNR data
    snr_data = loader.load_snr_data(str(test_dir / 'snr_data.csv'))
    print(f"✅ Loaded SNR data: {len(snr_data)} records")
    print(snr_data.head())

    # Test coordinates
    coords = loader.load_coordinates(str(test_dir / 'coordinates.json'), format='json')
    print(f"\n✅ Loaded coordinates for {len(coords)} stations")
    for station, pos in coords.items():
        print(f"  {station}: {pos}")

    # Test meteorological data
    meteo = loader.load_meteorological_data(str(test_dir / 'meteo_data.csv'))
    print(f"\n✅ Loaded meteorological data: {len(meteo)} records")
    print(meteo.head())

    print("\n" + "=" * 50)
    print("✅ Data loader test completed successfully!")
