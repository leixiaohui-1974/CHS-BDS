"""
Automation and Scheduling Module
=================================

Tools for automated monitoring, scheduling, and batch operations.
"""

import schedule
import time
from typing import Callable, Optional, Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from .main import CHSBDSSystem
from .data_loader import DataLoader
from .data_export import DataExporter, ResultsArchiver
from .alerts import AlertManager, GNSSAlertRules
from .logger import get_logger


logger = get_logger(__name__)


class AutomatedMonitor:
    """
    Automated monitoring system with scheduling.

    Runs monitoring tasks on a schedule and handles results automatically.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = './output/automated'
    ):
        """
        Initialize automated monitor.

        Args:
            config_path: Path to configuration file.
            output_dir: Directory for output files.
        """
        self.system = CHSBDSSystem(config_path=config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exporter = DataExporter(output_dir=str(self.output_dir))
        self.archiver = ResultsArchiver(archive_dir=str(self.output_dir / 'archive'))

        self.alert_manager = AlertManager()
        self._setup_default_alerts()

        self.job_history = []

        logger.info(f"AutomatedMonitor initialized: {output_dir}")

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        from .alerts import ConsoleNotifier, FileNotifier

        # Add notification channels
        self.alert_manager.add_channel(ConsoleNotifier())
        self.alert_manager.add_channel(
            FileNotifier(output_dir=str(self.output_dir / 'alerts'))
        )

        # Add rules
        self.alert_manager.add_rule(GNSSAlertRules.create_deformation_rule())
        self.alert_manager.add_rule(GNSSAlertRules.create_pwv_rule())
        self.alert_manager.add_rule(GNSSAlertRules.create_quality_rule())

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """
        Run a complete monitoring cycle.

        Returns:
            Results dictionary.
        """
        logger.info("="*70)
        logger.info(f"Starting automated monitoring cycle at {datetime.now()}")
        logger.info("="*70)

        try:
            # Run all modules
            results = self.system.run_all()

            # Check alerts
            self._check_alerts(results)

            # Export results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.exporter.export_json(results, f'results_{timestamp}')
            self.archiver.archive_results(results, tags=['automated'])

            # Record job
            self.job_history.append({
                'timestamp': datetime.now().isoformat(),
                'status': results.get('status', 'unknown'),
                'modules_run': len(results.get('modules', {}))
            })

            logger.info("✅ Monitoring cycle completed successfully")
            return results

        except Exception as e:
            logger.error(f"❌ Monitoring cycle failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def _check_alerts(self, results: Dict):
        """Check results against alert rules."""
        if 'modules' not in results:
            return

        # Check each module's results
        for module_name, module_data in results['modules'].items():
            alert_data = {
                'module': module_name,
                **module_data
            }
            self.alert_manager.check_all_rules(alert_data)

    def schedule_monitoring(
        self,
        interval_minutes: int = 60,
        run_immediately: bool = True
    ):
        """
        Schedule periodic monitoring.

        Args:
            interval_minutes: Interval between monitoring cycles (minutes).
            run_immediately: Run first cycle immediately.
        """
        logger.info(f"Scheduling monitoring every {interval_minutes} minutes")

        # Schedule the job
        schedule.every(interval_minutes).minutes.do(self.run_monitoring_cycle)

        # Run immediately if requested
        if run_immediately:
            self.run_monitoring_cycle()

        # Run scheduler
        logger.info("Scheduler started. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

    def get_job_history(self) -> List[Dict]:
        """Get history of executed jobs."""
        return self.job_history


class BatchAnalyzer:
    """
    Batch analysis tool for processing multiple datasets.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize batch analyzer.

        Args:
            config_path: Path to configuration file.
        """
        self.system = CHSBDSSystem(config_path=config_path)
        self.loader = DataLoader()
        logger.info("BatchAnalyzer initialized")

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.csv"
    ) -> Dict[str, Any]:
        """
        Process all files in a directory.

        Args:
            input_dir: Input directory path.
            output_dir: Output directory path.
            file_pattern: File pattern to match.

        Returns:
            Dictionary of processing results.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find matching files
        files = list(input_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files to process")

        results = {}
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")

            try:
                # Load data
                data = self.loader.load_csv(str(file_path))

                # Process (example: run GNSS-IR analysis)
                # This would be customized based on data type
                result = {
                    'status': 'success',
                    'file': file_path.name,
                    'records': len(data),
                    'processed_at': datetime.now().isoformat()
                }

                results[file_path.name] = result

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                results[file_path.name] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Save summary
        summary_path = output_path / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Batch processing complete. Summary saved to {summary_path}")
        return results


class SimpleAPI:
    """
    Simple REST-like API for CHS-BDS (基础框架).

    Note: For production use, consider Flask or FastAPI.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize API.

        Args:
            config_path: Path to configuration file.
        """
        self.system = CHSBDSSystem(config_path=config_path)
        self.loader = DataLoader()
        self.exporter = DataExporter()
        logger.info("SimpleAPI initialized")

    def get_status(self) -> Dict[str, Any]:
        """
        Get system status.

        Returns:
            Status dictionary.
        """
        return {
            'status': 'online',
            'version': '0.3.0',
            'timestamp': datetime.now().isoformat(),
            'modules': ['gnss_ir', 'deformation', 'pwv', 'rainfall']
        }

    def run_analysis(
        self,
        module: str,
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run analysis for specific module.

        Args:
            module: Module name ('gnss_ir', 'deformation', 'pwv', 'rainfall').
            parameters: Module-specific parameters.

        Returns:
            Analysis results.
        """
        logger.info(f"API request: run_analysis(module={module})")

        try:
            if module == 'gnss_ir':
                result = self.system.run_gnss_ir()
            elif module == 'deformation':
                result = self.system.run_deformation()
            elif module == 'pwv':
                result = self.system.run_pwv_estimation()
            elif module == 'rainfall':
                result = self.system.run_rainfall_prediction()
            else:
                return {'error': f'Unknown module: {module}'}

            return {
                'status': 'success',
                'module': module,
                'result': result
            }

        except Exception as e:
            logger.error(f"API error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_results(
        self,
        limit: int = 10,
        module: Optional[str] = None
    ) -> List[Dict]:
        """
        Get recent results (from archive).

        Args:
            limit: Maximum number of results to return.
            module: Filter by module name.

        Returns:
            List of results.
        """
        # This would query the archive directory
        # Simplified implementation
        return [{
            'timestamp': datetime.now().isoformat(),
            'module': module or 'all',
            'status': 'success'
        }]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CHS-BDS Automation Tools')
    parser.add_argument(
        'command',
        choices=['monitor', 'batch', 'api-test'],
        help='Command to run'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Monitoring interval in minutes'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory for batch processing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/automated',
        help='Output directory'
    )

    args = parser.parse_args()

    if args.command == 'monitor':
        print("Starting Automated Monitoring...")
        print("-" * 70)
        monitor = AutomatedMonitor(output_dir=args.output_dir)
        monitor.schedule_monitoring(interval_minutes=args.interval)

    elif args.command == 'batch':
        if not args.input_dir:
            print("Error: --input-dir required for batch processing")
            exit(1)

        print("Starting Batch Analysis...")
        print("-" * 70)
        analyzer = BatchAnalyzer()
        results = analyzer.process_directory(
            args.input_dir,
            args.output_dir
        )
        print(f"\n✅ Processed {len(results)} files")

    elif args.command == 'api-test':
        print("Testing Simple API...")
        print("-" * 70)

        api = SimpleAPI()

        print("\n1. Get status:")
        status = api.get_status()
        print(json.dumps(status, indent=2))

        print("\n2. Run GNSS-IR analysis:")
        result = api.run_analysis('gnss_ir')
        print(f"Status: {result['status']}")

        print("\n✅ API test completed!")
