"""
Data Quality Control Module
============================

Comprehensive data quality checking and validation for GNSS monitoring.
Includes outlier detection, completeness checks, and quality reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

from .logger import get_logger
from .exceptions import DataQualityError


logger = get_logger(__name__)


class QualityMetrics:
    """Container for quality control metrics."""

    def __init__(self):
        self.completeness = 0.0
        self.outlier_count = 0
        self.outlier_percentage = 0.0
        self.data_gaps = []
        self.suspicious_values = []
        self.quality_score = 0.0
        self.warnings = []
        self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'completeness': self.completeness,
            'outlier_count': self.outlier_count,
            'outlier_percentage': self.outlier_percentage,
            'data_gaps': self.data_gaps,
            'suspicious_values': self.suspicious_values,
            'quality_score': self.quality_score,
            'warnings': self.warnings,
            'errors': self.errors
        }


class DataQualityChecker:
    """
    Comprehensive data quality control for GNSS data.

    Performs multiple quality checks:
    - Completeness analysis
    - Outlier detection
    - Range validation
    - Consistency checks
    - Time series analysis
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize quality checker.

        Args:
            strict_mode: If True, raises errors on quality issues.
                        If False, logs warnings only.
        """
        self.strict_mode = strict_mode
        self.metrics = QualityMetrics()
        logger.info(f"DataQualityChecker initialized (strict_mode={strict_mode})")

    def check_completeness(
        self,
        data: pd.DataFrame,
        required_columns: List[str],
        min_completeness: float = 0.95
    ) -> float:
        """
        Check data completeness.

        Args:
            data: DataFrame to check.
            required_columns: List of required column names.
            min_completeness: Minimum acceptable completeness (0-1).

        Returns:
            Completeness score (0-1).
        """
        logger.info("Checking data completeness")

        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            self.metrics.errors.append(error_msg)
            if self.strict_mode:
                raise DataQualityError(error_msg)
            logger.error(error_msg)
            return 0.0

        # Calculate completeness for each required column
        completeness_scores = {}
        for col in required_columns:
            non_null_count = data[col].notna().sum()
            total_count = len(data)
            completeness = non_null_count / total_count if total_count > 0 else 0.0
            completeness_scores[col] = completeness

            if completeness < min_completeness:
                warning_msg = f"Column '{col}' completeness {completeness:.2%} < {min_completeness:.2%}"
                self.metrics.warnings.append(warning_msg)
                logger.warning(warning_msg)

        # Overall completeness
        overall_completeness = np.mean(list(completeness_scores.values()))
        self.metrics.completeness = overall_completeness

        logger.info(f"Data completeness: {overall_completeness:.2%}")
        return overall_completeness

    def detect_outliers(
        self,
        data: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers in data.

        Args:
            data: Data series to check.
            method: Detection method ('iqr', 'zscore', or 'mad').
            threshold: Threshold for outlier detection.

        Returns:
            Tuple of (outlier_mask, outlier_indices).
        """
        logger.info(f"Detecting outliers using {method} method")

        # Remove NaN values
        clean_data = data.dropna()

        if len(clean_data) == 0:
            logger.warning("No valid data for outlier detection")
            return np.array([]), np.array([])

        if method == 'iqr':
            # Interquartile Range method
            q1 = clean_data.quantile(0.25)
            q3 = clean_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (clean_data < lower_bound) | (clean_data > upper_bound)

        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((clean_data - clean_data.mean()) / clean_data.std())
            outlier_mask = z_scores > threshold

        elif method == 'mad':
            # Median Absolute Deviation
            median = clean_data.median()
            mad = np.median(np.abs(clean_data - median))
            modified_z_scores = 0.6745 * (clean_data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        outlier_indices = clean_data[outlier_mask].index.tolist()
        outlier_count = len(outlier_indices)
        outlier_percentage = (outlier_count / len(clean_data)) * 100

        self.metrics.outlier_count += outlier_count
        self.metrics.outlier_percentage = outlier_percentage

        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} outliers ({outlier_percentage:.2f}%)")

        return outlier_mask.values, np.array(outlier_indices)

    def check_value_ranges(
        self,
        data: pd.DataFrame,
        value_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, List[int]]:
        """
        Check if values are within expected ranges.

        Args:
            data: DataFrame to check.
            value_ranges: Dictionary of {column: (min, max)} ranges.

        Returns:
            Dictionary of {column: [out_of_range_indices]}.
        """
        logger.info("Checking value ranges")

        out_of_range = {}

        for col, (min_val, max_val) in value_ranges.items():
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found in data")
                continue

            # Find values outside range
            mask = (data[col] < min_val) | (data[col] > max_val)
            oor_indices = data[mask].index.tolist()

            if oor_indices:
                out_of_range[col] = oor_indices
                warning_msg = f"Column '{col}': {len(oor_indices)} values outside range [{min_val}, {max_val}]"
                self.metrics.warnings.append(warning_msg)
                logger.warning(warning_msg)

                # Store suspicious values
                for idx in oor_indices[:5]:  # Store first 5
                    self.metrics.suspicious_values.append({
                        'column': col,
                        'index': idx,
                        'value': float(data.loc[idx, col]),
                        'expected_range': [min_val, max_val]
                    })

        return out_of_range

    def detect_data_gaps(
        self,
        timestamps: pd.Series,
        expected_interval: float,
        max_gap_factor: float = 2.0
    ) -> List[Dict]:
        """
        Detect gaps in time series data.

        Args:
            timestamps: Series of timestamps.
            expected_interval: Expected time interval (in same units as timestamps).
            max_gap_factor: Gap is flagged if > expected_interval * max_gap_factor.

        Returns:
            List of gap dictionaries with start, end, and duration.
        """
        logger.info("Detecting data gaps")

        if len(timestamps) < 2:
            return []

        # Ensure timestamps are sorted
        timestamps = timestamps.sort_values()

        # Calculate intervals
        intervals = timestamps.diff()

        # Find gaps
        max_gap = expected_interval * max_gap_factor
        gap_mask = intervals > max_gap

        gaps = []
        for idx in intervals[gap_mask].index:
            gap = {
                'start': timestamps.loc[idx - 1],
                'end': timestamps.loc[idx],
                'duration': intervals.loc[idx],
                'expected': expected_interval
            }
            gaps.append(gap)

        self.metrics.data_gaps = gaps

        if gaps:
            logger.warning(f"Detected {len(gaps)} data gaps")

        return gaps

    def check_consistency(
        self,
        data: pd.DataFrame,
        consistency_rules: Dict[str, callable]
    ) -> Dict[str, List[int]]:
        """
        Check data consistency using custom rules.

        Args:
            data: DataFrame to check.
            consistency_rules: Dictionary of {rule_name: rule_function}.
                              Rule functions should return boolean mask.

        Returns:
            Dictionary of {rule_name: [failing_indices]}.
        """
        logger.info("Checking data consistency")

        inconsistencies = {}

        for rule_name, rule_func in consistency_rules.items():
            try:
                # Apply rule
                violation_mask = ~rule_func(data)
                violation_indices = data[violation_mask].index.tolist()

                if violation_indices:
                    inconsistencies[rule_name] = violation_indices
                    warning_msg = f"Consistency rule '{rule_name}': {len(violation_indices)} violations"
                    self.metrics.warnings.append(warning_msg)
                    logger.warning(warning_msg)

            except Exception as e:
                error_msg = f"Error applying rule '{rule_name}': {e}"
                self.metrics.errors.append(error_msg)
                logger.error(error_msg)

        return inconsistencies

    def calculate_quality_score(self) -> float:
        """
        Calculate overall quality score (0-100).

        Score is based on:
        - Completeness (40%)
        - Outlier rate (30%)
        - Errors and warnings (30%)

        Returns:
            Quality score (0-100).
        """
        # Completeness component (0-40)
        completeness_score = self.metrics.completeness * 40

        # Outlier component (0-30)
        # Lower outlier percentage = higher score
        outlier_penalty = min(self.metrics.outlier_percentage, 30)
        outlier_score = 30 - outlier_penalty

        # Errors and warnings component (0-30)
        error_penalty = min(len(self.metrics.errors) * 10, 30)
        warning_penalty = min(len(self.metrics.warnings) * 2, 30)
        issues_score = 30 - error_penalty - warning_penalty
        issues_score = max(issues_score, 0)

        total_score = completeness_score + outlier_score + issues_score
        self.metrics.quality_score = total_score

        return total_score

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate quality control report.

        Args:
            output_path: Optional path to save report.

        Returns:
            Report as string.
        """
        logger.info("Generating quality control report")

        # Calculate score
        score = self.calculate_quality_score()

        # Generate report
        report = f"""# Data Quality Control Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overall Quality Score

**Score**: {score:.1f}/100

"""

        # Score interpretation
        if score >= 90:
            report += "✅ **Excellent**: Data quality is very high\n"
        elif score >= 75:
            report += "✅ **Good**: Data quality is acceptable\n"
        elif score >= 60:
            report += "⚠️ **Fair**: Data quality has some issues\n"
        else:
            report += "❌ **Poor**: Data quality needs significant improvement\n"

        report += "\n---\n\n## Quality Metrics\n\n"
        report += f"- **Completeness**: {self.metrics.completeness:.2%}\n"
        report += f"- **Outliers**: {self.metrics.outlier_count} ({self.metrics.outlier_percentage:.2f}%)\n"
        report += f"- **Data Gaps**: {len(self.metrics.data_gaps)}\n"
        report += f"- **Warnings**: {len(self.metrics.warnings)}\n"
        report += f"- **Errors**: {len(self.metrics.errors)}\n"

        # Errors section
        if self.metrics.errors:
            report += "\n---\n\n## Errors\n\n"
            for i, error in enumerate(self.metrics.errors, 1):
                report += f"{i}. {error}\n"

        # Warnings section
        if self.metrics.warnings:
            report += "\n---\n\n## Warnings\n\n"
            for i, warning in enumerate(self.metrics.warnings, 1):
                report += f"{i}. {warning}\n"

        # Data gaps section
        if self.metrics.data_gaps:
            report += "\n---\n\n## Data Gaps\n\n"
            for i, gap in enumerate(self.metrics.data_gaps, 1):
                report += f"{i}. Gap from {gap['start']} to {gap['end']} (duration: {gap['duration']})\n"

        # Suspicious values
        if self.metrics.suspicious_values:
            report += "\n---\n\n## Suspicious Values\n\n"
            for i, sv in enumerate(self.metrics.suspicious_values, 1):
                report += f"{i}. Column '{sv['column']}' at index {sv['index']}: "
                report += f"value={sv['value']}, expected range={sv['expected_range']}\n"

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def get_metrics(self) -> QualityMetrics:
        """Get quality metrics."""
        return self.metrics


# Predefined consistency rules for GNSS data
class GNSSConsistencyRules:
    """Common consistency rules for GNSS data."""

    @staticmethod
    def elevation_range(data: pd.DataFrame) -> pd.Series:
        """Elevation should be between 0 and 90 degrees."""
        if 'elevation' not in data.columns:
            return pd.Series([True] * len(data))
        return (data['elevation'] >= 0) & (data['elevation'] <= 90)

    @staticmethod
    def azimuth_range(data: pd.DataFrame) -> pd.Series:
        """Azimuth should be between 0 and 360 degrees."""
        if 'azimuth' not in data.columns:
            return pd.Series([True] * len(data))
        return (data['azimuth'] >= 0) & (data['azimuth'] <= 360)

    @staticmethod
    def snr_reasonable(data: pd.DataFrame) -> pd.Series:
        """SNR should be between -50 and 60 dB."""
        if 'snr' not in data.columns:
            return pd.Series([True] * len(data))
        return (data['snr'] >= -50) & (data['snr'] <= 60)

    @staticmethod
    def temperature_reasonable(data: pd.DataFrame) -> pd.Series:
        """Temperature should be between -50 and 60 Celsius."""
        if 'temperature' not in data.columns:
            return pd.Series([True] * len(data))
        return (data['temperature'] >= -50) & (data['temperature'] <= 60)

    @staticmethod
    def pressure_reasonable(data: pd.DataFrame) -> pd.Series:
        """Pressure should be between 800 and 1100 hPa."""
        if 'pressure' not in data.columns:
            return pd.Series([True] * len(data))
        return (data['pressure'] >= 800) & (data['pressure'] <= 1100)


if __name__ == '__main__':
    # Test quality control
    print("Testing Data Quality Control Module...")
    print("-" * 70)

    # Create test data with some issues
    np.random.seed(42)
    n = 100

    test_data = pd.DataFrame({
        'elevation': np.random.uniform(5, 30, n),
        'azimuth': np.random.uniform(0, 360, n),
        'snr': np.random.normal(45, 5, n),
        'temperature': np.random.normal(20, 5, n)
    })

    # Introduce some issues
    test_data.loc[10:15, 'snr'] = np.nan  # Missing data
    test_data.loc[50, 'snr'] = 100  # Outlier
    test_data.loc[80, 'elevation'] = 95  # Out of range

    # Initialize checker
    checker = DataQualityChecker(strict_mode=False)

    # Run checks
    print("\n1. Checking completeness...")
    completeness = checker.check_completeness(
        test_data,
        required_columns=['elevation', 'azimuth', 'snr'],
        min_completeness=0.95
    )
    print(f"   Completeness: {completeness:.2%}")

    print("\n2. Detecting outliers...")
    outlier_mask, outlier_idx = checker.detect_outliers(
        test_data['snr'],
        method='iqr',
        threshold=3.0
    )
    print(f"   Found {len(outlier_idx)} outliers")

    print("\n3. Checking value ranges...")
    out_of_range = checker.check_value_ranges(
        test_data,
        value_ranges={
            'elevation': (0, 90),
            'azimuth': (0, 360),
            'snr': (-50, 60)
        }
    )
    print(f"   Out of range values in {len(out_of_range)} columns")

    print("\n4. Checking consistency...")
    rules = {
        'elevation_range': GNSSConsistencyRules.elevation_range,
        'snr_reasonable': GNSSConsistencyRules.snr_reasonable
    }
    inconsistencies = checker.check_consistency(test_data, rules)
    print(f"   Found inconsistencies in {len(inconsistencies)} rules")

    print("\n5. Generating report...")
    report = checker.generate_report(output_path='./output/quality_report.md')

    print("\n" + "=" * 70)
    print(f"Quality Score: {checker.metrics.quality_score:.1f}/100")
    print("=" * 70)
    print("\n✅ Quality control test completed!")
