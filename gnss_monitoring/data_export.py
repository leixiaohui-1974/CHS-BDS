"""
Data Export Module
==================

Provides functionality to export analysis results to various formats.
Supports CSV, JSON, HDF5, and Excel formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .logger import get_logger
from .exceptions import DataFormatError


logger = get_logger(__name__)


class DataExporter:
    """
    Export GNSS monitoring results to various formats.

    Supports:
    - CSV (Comma-Separated Values)
    - JSON (JavaScript Object Notation)
    - Excel (XLSX)
    - Markdown (for reports)
    """

    def __init__(self, output_dir: str = './output'):
        """
        Initialize data exporter.

        Args:
            output_dir: Base directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataExporter initialized with output dir: {output_dir}")

    def export_results(
        self,
        results: Dict[str, Any],
        format: str = 'json',
        filename: Optional[str] = None
    ) -> Path:
        """
        Export analysis results to specified format.

        Args:
            results: Dictionary containing analysis results.
            format: Output format ('json', 'csv', 'excel', 'markdown').
            filename: Custom filename. If None, generates timestamp-based name.

        Returns:
            Path to exported file.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results_{timestamp}"

        if format == 'json':
            return self.export_json(results, filename)
        elif format == 'csv':
            return self.export_csv(results, filename)
        elif format == 'excel':
            return self.export_excel(results, filename)
        elif format == 'markdown':
            return self.export_markdown(results, filename)
        else:
            raise DataFormatError(f"Unsupported export format: {format}")

    def export_json(
        self,
        data: Dict[str, Any],
        filename: str,
        indent: int = 2
    ) -> Path:
        """
        Export data to JSON format.

        Args:
            data: Data to export.
            filename: Output filename (without extension).
            indent: JSON indentation level.

        Returns:
            Path to exported file.
        """
        output_path = self.output_dir / f"{filename}.json"
        logger.info(f"Exporting JSON to {output_path}")

        # Convert numpy types to Python types for JSON serialization
        data_cleaned = self._clean_for_json(data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_cleaned, f, indent=indent, ensure_ascii=False)

        logger.info(f"✅ JSON exported successfully")
        return output_path

    def export_csv(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Export data to CSV format.

        Args:
            data: Data to export (converts dict to DataFrame).
            filename: Output filename (without extension).

        Returns:
            Path to exported file.
        """
        output_path = self.output_dir / f"{filename}.csv"
        logger.info(f"Exporting CSV to {output_path}")

        # Convert to DataFrame
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = self._dict_to_dataframe(data)
        else:
            raise DataFormatError(f"Cannot convert {type(data)} to CSV")

        df.to_csv(output_path, index=False)
        logger.info(f"✅ CSV exported successfully: {len(df)} rows")
        return output_path

    def export_excel(
        self,
        data: Dict[str, Any],
        filename: str,
        sheet_name: str = 'Results'
    ) -> Path:
        """
        Export data to Excel format.

        Args:
            data: Data to export.
            filename: Output filename (without extension).
            sheet_name: Excel sheet name.

        Returns:
            Path to exported file.
        """
        output_path = self.output_dir / f"{filename}.xlsx"
        logger.info(f"Exporting Excel to {output_path}")

        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict):
                df = self._dict_to_dataframe(data)
            else:
                raise DataFormatError(f"Cannot convert {type(data)} to Excel")

            df.to_excel(output_path, sheet_name=sheet_name, index=False)
            logger.info(f"✅ Excel exported successfully")
            return output_path

        except ImportError:
            logger.error("openpyxl not installed. Install with: pip install openpyxl")
            raise

    def export_markdown(
        self,
        results: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Export results to Markdown report.

        Args:
            results: Analysis results dictionary.
            filename: Output filename (without extension).

        Returns:
            Path to exported file.
        """
        output_path = self.output_dir / f"{filename}.md"
        logger.info(f"Exporting Markdown report to {output_path}")

        report = self._generate_markdown_report(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✅ Markdown report exported successfully")
        return output_path

    def export_timeseries(
        self,
        data: pd.DataFrame,
        filename: str,
        format: str = 'csv'
    ) -> Path:
        """
        Export time series data.

        Args:
            data: Time series DataFrame.
            filename: Output filename.
            format: Output format ('csv' or 'excel').

        Returns:
            Path to exported file.
        """
        logger.info(f"Exporting time series data")

        if format == 'csv':
            return self.export_csv(data, filename)
        elif format == 'excel':
            return self.export_excel(data, filename, sheet_name='TimeSeries')
        else:
            raise DataFormatError(f"Unsupported format for time series: {format}")

    def _clean_for_json(self, obj: Any) -> Any:
        """
        Recursively clean object for JSON serialization.

        Converts numpy types to Python native types.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        else:
            return obj

    def _dict_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert nested dictionary to flat DataFrame.

        Args:
            data: Dictionary to convert.

        Returns:
            Flattened DataFrame.
        """
        # Flatten nested dictionary
        flat_data = {}
        self._flatten_dict(data, flat_data)

        # Convert to DataFrame
        if all(isinstance(v, (list, np.ndarray)) for v in flat_data.values()):
            # All values are sequences - create normal DataFrame
            df = pd.DataFrame(flat_data)
        else:
            # Mixed types - create single-row DataFrame
            df = pd.DataFrame([flat_data])

        return df

    def _flatten_dict(
        self,
        d: Dict,
        result: Dict,
        prefix: str = ''
    ):
        """
        Flatten nested dictionary with dot notation keys.

        Args:
            d: Dictionary to flatten.
            result: Output dictionary.
            prefix: Key prefix for nested items.
        """
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_dict(value, result, new_key)
            else:
                result[new_key] = value

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """
        Generate Markdown report from results.

        Args:
            results: Analysis results.

        Returns:
            Markdown-formatted report string.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""# CHS-BDS Analysis Report

**Generated**: {timestamp}
**System**: CHS-BDS v{results.get('version', '0.1.0')}

---

## Executive Summary

Analysis completed successfully with {len(results.get('modules', {}))} modules.

"""

        # Add module results
        if 'modules' in results:
            report += "## Module Results\n\n"

            for module_name, module_data in results['modules'].items():
                report += f"### {module_name.upper()}\n\n"
                report += self._format_module_results(module_data)
                report += "\n"

        # Add status
        report += f"""
---

## Status

**Overall Status**: {results.get('status', 'unknown')}

"""

        if 'error' in results:
            report += f"\n### Errors\n\n```\n{results['error']}\n```\n"

        return report

    def _format_module_results(self, data: Dict[str, Any]) -> str:
        """Format module results as Markdown table."""
        lines = []

        for key, value in data.items():
            if isinstance(value, (int, float, str)):
                lines.append(f"- **{key}**: {value}")
            elif isinstance(value, (list, tuple)):
                if len(value) <= 3:
                    lines.append(f"- **{key}**: {value}")
                else:
                    lines.append(f"- **{key}**: {type(value).__name__} ({len(value)} items)")
            elif isinstance(value, dict):
                lines.append(f"- **{key}**: {len(value)} items")

        return "\n".join(lines)


class ResultsArchiver:
    """
    Archive and organize historical results.
    """

    def __init__(self, archive_dir: str = './output/archive'):
        """
        Initialize results archiver.

        Args:
            archive_dir: Directory for archived results.
        """
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ResultsArchiver initialized: {archive_dir}")

    def archive_results(
        self,
        results: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> Path:
        """
        Archive results with timestamp and tags.

        Args:
            results: Results to archive.
            tags: Optional tags for categorization.

        Returns:
            Path to archived file.
        """
        timestamp = datetime.now()
        date_dir = self.archive_dir / timestamp.strftime('%Y%m%d')
        date_dir.mkdir(exist_ok=True)

        filename = timestamp.strftime('%H%M%S')
        if tags:
            filename += f"_{'_'.join(tags)}"
        filename += '.json'

        output_path = date_dir / filename

        # Add timestamp to results
        results['archived_at'] = timestamp.isoformat()
        if tags:
            results['tags'] = tags

        # Export
        exporter = DataExporter(output_dir=date_dir)
        exporter.export_json(results, filename.replace('.json', ''))

        logger.info(f"Results archived to {output_path}")
        return output_path

    def list_archives(
        self,
        date: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Path]:
        """
        List archived results.

        Args:
            date: Filter by date (YYYYMMDD format).
            tags: Filter by tags.

        Returns:
            List of paths to archived files.
        """
        if date:
            search_dir = self.archive_dir / date
            if not search_dir.exists():
                return []
            pattern = '*.json'
        else:
            search_dir = self.archive_dir
            pattern = '**/*.json'

        files = list(search_dir.glob(pattern))

        # Filter by tags if specified
        if tags:
            filtered = []
            for file in files:
                if any(tag in file.stem for tag in tags):
                    filtered.append(file)
            files = filtered

        return sorted(files)


if __name__ == '__main__':
    # Test data export
    print("Testing Data Export Module...")
    print("-" * 50)

    # Create test results
    test_results = {
        'system': 'CHS-BDS',
        'version': '0.1.0',
        'status': 'success',
        'modules': {
            'gnss_ir': {
                'estimated_height': 4.52,
                'true_height': 4.50,
                'error': 0.02,
                'antenna_height': 1.5
            },
            'pwv': {
                'mean_pwv': 15.23,
                'max_pwv': 18.45,
                'min_pwv': 12.67,
                'unit': 'mm'
            }
        }
    }

    # Test exporter
    exporter = DataExporter(output_dir='./output/test')

    print("\n1. Exporting to JSON...")
    json_path = exporter.export_json(test_results, 'test_results')
    print(f"✅ Exported to: {json_path}")

    print("\n2. Exporting to Markdown...")
    md_path = exporter.export_markdown(test_results, 'test_report')
    print(f"✅ Exported to: {md_path}")

    print("\n3. Testing archiver...")
    archiver = ResultsArchiver(archive_dir='./output/archive')
    archive_path = archiver.archive_results(
        test_results,
        tags=['test', 'demo']
    )
    print(f"✅ Archived to: {archive_path}")

    print("\n4. Listing archives...")
    archives = archiver.list_archives()
    print(f"✅ Found {len(archives)} archived results")

    print("\n" + "=" * 50)
    print("✅ Data export test completed successfully!")
