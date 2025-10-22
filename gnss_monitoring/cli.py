"""
Command Line Interface Module
==============================

Provides rich CLI for the CHS-BDS system with subcommands for each module.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from .main import CHSBDSSystem
from .config import get_config
from .logger import get_logger


class CLI:
    """Command Line Interface for CHS-BDS."""

    def __init__(self):
        """Initialize CLI."""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            prog='chs-bds',
            description='CHS-BDS: GNSS Comprehensive Monitoring System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all modules
  chs-bds run --all

  # Run specific module
  chs-bds run --module gnss_ir

  # Show system info
  chs-bds info

  # Validate configuration
  chs-bds config --validate

For more information, visit: https://github.com/leixiaohui-1974/CHS-BDS
            """
        )

        parser.add_argument(
            '--version',
            action='version',
            version='CHS-BDS v0.1.0'
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Run command
        self._add_run_command(subparsers)

        # Info command
        self._add_info_command(subparsers)

        # Config command
        self._add_config_command(subparsers)

        # Test command
        self._add_test_command(subparsers)

        return parser

    def _add_run_command(self, subparsers):
        """Add 'run' subcommand."""
        run_parser = subparsers.add_parser(
            'run',
            help='Run monitoring modules',
            description='Execute CHS-BDS monitoring modules'
        )

        run_parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file',
            metavar='FILE'
        )

        run_parser.add_argument(
            '--module', '-m',
            type=str,
            choices=['gnss_ir', 'deformation', 'pwv', 'rainfall'],
            help='Run specific module'
        )

        run_parser.add_argument(
            '--all', '-a',
            action='store_true',
            help='Run all modules sequentially'
        )

        run_parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output directory',
            metavar='DIR'
        )

        run_parser.add_argument(
            '--no-plots',
            action='store_true',
            help='Disable plot generation'
        )

        run_parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Verbose output'
        )

    def _add_info_command(self, subparsers):
        """Add 'info' subcommand."""
        info_parser = subparsers.add_parser(
            'info',
            help='Display system information',
            description='Show CHS-BDS system information'
        )

        info_parser.add_argument(
            '--modules',
            action='store_true',
            help='Show available modules'
        )

        info_parser.add_argument(
            '--config',
            action='store_true',
            help='Show current configuration'
        )

    def _add_config_command(self, subparsers):
        """Add 'config' subcommand."""
        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management',
            description='Manage CHS-BDS configuration'
        )

        config_parser.add_argument(
            '--show',
            action='store_true',
            help='Show current configuration'
        )

        config_parser.add_argument(
            '--validate',
            action='store_true',
            help='Validate configuration file'
        )

        config_parser.add_argument(
            '--generate',
            type=str,
            help='Generate default configuration file',
            metavar='FILE'
        )

        config_parser.add_argument(
            '--file', '-f',
            type=str,
            help='Configuration file to use',
            metavar='FILE'
        )

    def _add_test_command(self, subparsers):
        """Add 'test' subcommand."""
        test_parser = subparsers.add_parser(
            'test',
            help='Run system tests',
            description='Test CHS-BDS modules'
        )

        test_parser.add_argument(
            '--module',
            type=str,
            choices=['gnss_ir', 'deformation', 'pwv', 'rainfall', 'all'],
            default='all',
            help='Module to test'
        )

    def run(self, args: List[str] = None):
        """
        Execute CLI command.

        Args:
            args: Command line arguments (defaults to sys.argv).
        """
        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return 1

        try:
            if parsed_args.command == 'run':
                return self._handle_run(parsed_args)
            elif parsed_args.command == 'info':
                return self._handle_info(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config(parsed_args)
            elif parsed_args.command == 'test':
                return self._handle_test(parsed_args)
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        return 0

    def _handle_run(self, args) -> int:
        """Handle 'run' command."""
        # Initialize system
        system = CHSBDSSystem(config_path=args.config)

        # Override config with command-line arguments
        if args.output:
            system.config.set('global.output_dir', args.output)

        if args.no_plots:
            system.config.set('global.enable_plots', False)

        if args.verbose:
            system.config.set('global.log_level', 'DEBUG')
            system.logger.setLevel('DEBUG')

        # Run requested modules
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
            print("Error: Specify --all or --module")
            return 1

        return 0

    def _handle_info(self, args) -> int:
        """Handle 'info' command."""
        print("\n" + "=" * 70)
        print("CHS-BDS: GNSS Comprehensive Monitoring System")
        print("=" * 70)
        print("\nVersion: 0.1.0")
        print("Author: Lei Xiaohui")
        print("License: MIT")

        if args.modules:
            print("\n" + "-" * 70)
            print("Available Modules:")
            print("-" * 70)
            modules = [
                ("gnss_ir", "GNSS-IR Water Level Monitoring"),
                ("deformation", "High-Precision Deformation Monitoring"),
                ("pwv", "Precipitable Water Vapor Estimation"),
                ("rainfall", "Rainfall Prediction Model")
            ]
            for name, description in modules:
                print(f"  • {name:15} - {description}")

        if args.config:
            print("\n" + "-" * 70)
            print("Current Configuration:")
            print("-" * 70)
            config = get_config()
            import yaml
            print(yaml.dump(config._config, default_flow_style=False))

        print()
        return 0

    def _handle_config(self, args) -> int:
        """Handle 'config' command."""
        import yaml

        if args.generate:
            # Generate default config
            output_path = Path(args.generate)
            if output_path.exists():
                response = input(f"File {output_path} exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Operation cancelled")
                    return 0

            config = get_config()
            config.save_config(str(output_path))
            print(f"✅ Configuration file generated: {output_path}")
            return 0

        if args.validate:
            # Validate config
            config_file = args.file if args.file else 'config.yaml'
            try:
                config = get_config(config_file)
                print(f"✅ Configuration file is valid: {config_file}")
                return 0
            except Exception as e:
                print(f"❌ Configuration validation failed: {e}", file=sys.stderr)
                return 1

        if args.show:
            # Show config
            config_file = args.file if args.file else None
            config = get_config(config_file)
            print("\nCurrent Configuration:")
            print("-" * 70)
            print(yaml.dump(config._config, default_flow_style=False))
            return 0

        print("Use --show, --validate, or --generate")
        return 1

    def _handle_test(self, args) -> int:
        """Handle 'test' command."""
        print(f"\n🧪 Testing CHS-BDS module: {args.module}")
        print("-" * 70)

        system = CHSBDSSystem()

        try:
            if args.module == 'all' or args.module == 'gnss_ir':
                print("\nTesting GNSS-IR module...")
                system.run_gnss_ir(plot=False)
                print("✅ GNSS-IR module test passed")

            if args.module == 'all' or args.module == 'deformation':
                print("\nTesting Deformation module...")
                system.run_deformation()
                print("✅ Deformation module test passed")

            if args.module == 'all' or args.module == 'pwv':
                print("\nTesting PWV module...")
                system.run_pwv_estimation()
                print("✅ PWV module test passed")

            if args.module == 'all' or args.module == 'rainfall':
                print("\nTesting Rainfall module...")
                system.run_rainfall_prediction()
                print("✅ Rainfall module test passed")

            print("\n" + "=" * 70)
            print("All tests passed!")
            print("=" * 70 + "\n")
            return 0

        except Exception as e:
            print(f"\n❌ Test failed: {e}", file=sys.stderr)
            return 1


def main():
    """Main entry point for CLI."""
    cli = CLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
