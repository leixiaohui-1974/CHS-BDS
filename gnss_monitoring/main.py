"""
Main Integration Module
========================

Provides unified interface for running all CHS-BDS monitoring modules.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import json

from .config import get_config
from .logger import setup_logger, log_execution_time
from .exceptions import CHSBDSException
from .gnss_ir import GNSSIRWaterLevel
from .deformation import DeformationMonitor
from .pwv import PWVEstimator
from .rainfall_model import RainfallPredictor

import numpy as np


class CHSBDSSystem:
    """
    Main system controller for CHS-BDS monitoring.

    Integrates all four monitoring modules and provides unified interface.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CHS-BDS system.

        Args:
            config_path: Path to configuration file.
        """
        # Load configuration
        self.config = get_config(config_path)

        # Setup logging
        log_level = self.config.get('global.log_level', 'INFO')
        log_dir = Path(self.config.get('global.output_dir', './output')) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            name='chs_bds',
            level=log_level,
            log_file=str(log_dir / 'chs-bds.log'),
            console_output=True,
            colored_output=True
        )

        self.logger.info("=" * 70)
        self.logger.info("CHS-BDS System Initialized")
        self.logger.info("=" * 70)

        # Initialize modules
        self.gnss_ir = None
        self.deformation = None
        self.pwv = None
        self.rainfall = None

    @log_execution_time
    def run_gnss_ir(
        self,
        satellite_elevation: np.ndarray = None,
        satellite_azimuth: np.ndarray = None,
        true_reflector_height: float = 4.5,
        plot: bool = True
    ) -> dict:
        """
        Run GNSS-IR water level monitoring.

        Args:
            satellite_elevation: Satellite elevation angles (degrees).
            satellite_azimuth: Satellite azimuth angles (degrees).
            true_reflector_height: True reflector height for simulation (meters).
            plot: Whether to generate plots.

        Returns:
            Dictionary with analysis results.
        """
        self.logger.info("Starting GNSS-IR Water Level Analysis...")

        # Get configuration
        config = self.config.get_section('gnss_ir')

        # Use provided data or generate default
        if satellite_elevation is None:
            satellite_elevation = np.linspace(
                config.get('min_elevation', 5),
                config.get('max_elevation', 30),
                200
            )

        if satellite_azimuth is None:
            satellite_azimuth = np.linspace(120, 150, 200)

        # Initialize analyzer
        self.gnss_ir = GNSSIRWaterLevel(
            satellite_elevation=satellite_elevation,
            satellite_azimuth=satellite_azimuth,
            antenna_height=config.get('antenna_height', 1.5),
            wavelength=config.get('wavelength', 0.1903)
        )

        # Run analysis
        estimated_height = self.gnss_ir.run_analysis(
            true_reflector_height=true_reflector_height,
            plot=plot and self.config.get('global.enable_plots', True)
        )

        result = {
            'module': 'gnss_ir',
            'estimated_height': float(estimated_height),
            'true_height': float(true_reflector_height),
            'error': float(abs(true_reflector_height - estimated_height)),
            'antenna_height': config.get('antenna_height', 1.5),
        }

        self.logger.info(f"GNSS-IR Analysis Complete: Estimated height = {estimated_height:.3f} m")
        return result

    @log_execution_time
    def run_deformation(
        self,
        ref_pos: List[float] = None,
        mon_pos: List[float] = None,
        satellite_positions: dict = None
    ) -> dict:
        """
        Run deformation monitoring analysis.

        Args:
            ref_pos: Reference station ECEF coordinates [X, Y, Z].
            mon_pos: Monitoring station ECEF coordinates [X, Y, Z].
            satellite_positions: Dictionary of satellite positions.

        Returns:
            Dictionary with deformation results.
        """
        self.logger.info("Starting Deformation Monitoring Analysis...")

        # Get configuration
        config = self.config.get_section('deformation')

        # Use provided data or defaults
        if ref_pos is None:
            ref_pos = [
                config.get('ref_station.x', 3275650.0),
                config.get('ref_station.y', 553640.0),
                config.get('ref_station.z', 5201550.0)
            ]

        if mon_pos is None:
            mon_pos = [
                config.get('mon_station.x', 3275750.0),
                config.get('mon_station.y', 553650.0),
                config.get('mon_station.z', 5201545.0)
            ]

        if satellite_positions is None:
            satellite_positions = {
                'G01': [20000000, 10000000, 15000000],
                'G05': [22000000, -5000000, 18000000],
                'G12': [15000000, 15000000, 20000000],
                'G21': [18000000, -12000000, 16000000],
            }

        # Initialize monitor
        self.deformation = DeformationMonitor(
            ref_pos=ref_pos,
            mon_pos=mon_pos,
            satellite_positions=satellite_positions
        )

        # Run analysis
        self.deformation._simulate_phase_observations()
        self.deformation.perform_double_differencing()

        # Use approximate position (slightly offset from true position)
        approx_mon_pos = [
            mon_pos[0] + 0.1,
            mon_pos[1] - 0.1,
            mon_pos[2] + 0.05
        ]
        estimated_baseline = self.deformation.solve_baseline(approx_mon_pos=approx_mon_pos)

        error = self.deformation.true_baseline - estimated_baseline
        error_magnitude = float(np.linalg.norm(error))

        result = {
            'module': 'deformation',
            'true_baseline': self.deformation.true_baseline.tolist(),
            'estimated_baseline': estimated_baseline.tolist(),
            'error': error.tolist(),
            'error_magnitude': error_magnitude,
            'unit': 'meters'
        }

        self.logger.info(f"Deformation Analysis Complete: Error magnitude = {error_magnitude:.4f} m")
        return result

    @log_execution_time
    def run_pwv_estimation(self) -> dict:
        """
        Run PWV estimation analysis.

        Returns:
            Dictionary with PWV results.
        """
        self.logger.info("Starting PWV Estimation...")

        # Get configuration
        config = self.config.get_section('pwv')

        # Initialize estimator
        self.pwv = PWVEstimator(
            latitude=config.get('station.latitude', 34.0),
            height=config.get('station.height', 150.0)
        )

        # Run analysis
        self.pwv.simulate_inputs(
            num_epochs=config.get('num_epochs', 288),
            interval_minutes=config.get('interval_minutes', 5)
        )
        self.pwv.calculate_zhd()
        self.pwv.calculate_pwv()

        if self.config.get('global.enable_plots', True):
            self.pwv.plot_results()

        result = {
            'module': 'pwv',
            'mean_pwv': float(np.mean(self.pwv.pwv)),
            'max_pwv': float(np.max(self.pwv.pwv)),
            'min_pwv': float(np.min(self.pwv.pwv)),
            'std_pwv': float(np.std(self.pwv.pwv)),
            'unit': 'mm',
            'num_epochs': len(self.pwv.pwv)
        }

        self.logger.info(f"PWV Estimation Complete: Mean PWV = {result['mean_pwv']:.2f} mm")
        return result

    @log_execution_time
    def run_rainfall_prediction(self) -> dict:
        """
        Run rainfall prediction analysis.

        Returns:
            Dictionary with prediction results.
        """
        self.logger.info("Starting Rainfall Prediction...")

        # Get configuration
        config = self.config.get_section('rainfall')

        # Simulate data
        timestamps, pwv_data, rainfall_data = RainfallPredictor.simulate_data(
            days=config.get('lookback_days', 30),
            interval_minutes=config.get('interval_minutes', 10)
        )

        # Initialize predictor
        self.rainfall = RainfallPredictor(timestamps, pwv_data, rainfall_data)

        # Feature engineering and labeling
        self.rainfall.engineer_features(windows=config.get('feature_windows', [1, 3, 6]))
        self.rainfall.label_data(look_ahead_minutes=config.get('look_ahead_minutes', 30))

        # Train and evaluate model
        model = self.rainfall.train_and_evaluate(
            test_size=config.get('test_size', 0.3),
            random_state=config.get('random_state', 42)
        )

        result = {
            'module': 'rainfall_prediction',
            'model_type': config.get('model_type', 'logistic_regression'),
            'look_ahead_minutes': config.get('look_ahead_minutes', 30),
            'training_samples': len(self.rainfall.df),
            'status': 'completed'
        }

        self.logger.info("Rainfall Prediction Complete")
        return result

    def run_all(self) -> dict:
        """
        Run all monitoring modules sequentially.

        Returns:
            Dictionary with all results.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("Running All CHS-BDS Modules")
        self.logger.info("=" * 70 + "\n")

        results = {
            'system': 'CHS-BDS',
            'version': '0.1.0',
            'modules': {}
        }

        try:
            # Run each module
            results['modules']['gnss_ir'] = self.run_gnss_ir()
            results['modules']['deformation'] = self.run_deformation()
            results['modules']['pwv'] = self.run_pwv_estimation()
            results['modules']['rainfall'] = self.run_rainfall_prediction()

            results['status'] = 'success'
            self.logger.info("\n" + "=" * 70)
            self.logger.info("All Modules Completed Successfully")
            self.logger.info("=" * 70)

        except CHSBDSException as e:
            self.logger.error(f"CHS-BDS Error: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
            results['status'] = 'error'
            results['error'] = str(e)

        # Save results
        output_dir = Path(self.config.get('global.output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / 'results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nResults saved to: {results_file}")

        return results


def main():
    """
    Main entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description='CHS-BDS: GNSS Comprehensive Monitoring System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all modules
  python -m gnss_monitoring.main --all

  # Run specific module
  python -m gnss_monitoring.main --module gnss_ir

  # Use custom configuration
  python -m gnss_monitoring.main --all --config my_config.yaml
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--module', '-m',
        type=str,
        choices=['gnss_ir', 'deformation', 'pwv', 'rainfall'],
        help='Run specific module'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all modules'
    )

    args = parser.parse_args()

    # Initialize system
    system = CHSBDSSystem(config_path=args.config)

    try:
        if args.all:
            system.run_all()
        elif args.module:
            if args.module == 'gnss_ir':
                system.run_gnss_ir()
            elif args.module == 'deformation':
                system.run_deformation()
            elif args.module == 'pwv':
                system.run_pwv_estimation()
            elif args.module == 'rainfall':
                system.run_rainfall_prediction()
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        system.logger.warning("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        system.logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
