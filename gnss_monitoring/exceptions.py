"""
Custom Exception Classes
=========================

Defines custom exceptions for the CHS-BDS system to provide better error handling
and more informative error messages.
"""


class CHSBDSException(Exception):
    """
    Base exception class for all CHS-BDS exceptions.
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize exception.

        Args:
            message: Error message.
            details: Optional dictionary with additional error details.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = ', '.join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


# Data-related exceptions
class DataLoadError(CHSBDSException):
    """Raised when data loading fails."""
    pass


class DataFormatError(CHSBDSException):
    """Raised when data format is invalid."""
    pass


class DataQualityError(CHSBDSException):
    """Raised when data quality is insufficient."""
    pass


class MissingDataError(CHSBDSException):
    """Raised when required data is missing."""
    pass


# Configuration exceptions
class ConfigurationError(CHSBDSException):
    """Raised when configuration is invalid or missing."""
    pass


class InvalidParameterError(CHSBDSException):
    """Raised when a parameter value is invalid."""
    pass


# Processing exceptions
class ProcessingError(CHSBDSException):
    """Raised when data processing fails."""
    pass


class ConvergenceError(CHSBDSException):
    """Raised when iterative algorithm fails to converge."""
    pass


class InsufficientObservationsError(CHSBDSException):
    """Raised when there are insufficient observations for processing."""
    pass


# GNSS-IR specific exceptions
class GNSSIRError(CHSBDSException):
    """Base exception for GNSS-IR module."""
    pass


class SNRDataError(GNSSIRError):
    """Raised when SNR data is invalid or insufficient."""
    pass


class FrequencyAnalysisError(GNSSIRError):
    """Raised when frequency analysis fails."""
    pass


# Deformation monitoring exceptions
class DeformationError(CHSBDSException):
    """Base exception for deformation monitoring module."""
    pass


class BaselineError(DeformationError):
    """Raised when baseline calculation fails."""
    pass


class CoordinateError(DeformationError):
    """Raised when coordinate transformation or validation fails."""
    pass


# PWV estimation exceptions
class PWVError(CHSBDSException):
    """Base exception for PWV estimation module."""
    pass


class ZTDError(PWVError):
    """Raised when ZTD data is invalid or processing fails."""
    pass


class MeteoDataError(PWVError):
    """Raised when meteorological data is invalid or missing."""
    pass


# Rainfall prediction exceptions
class RainfallPredictionError(CHSBDSException):
    """Base exception for rainfall prediction module."""
    pass


class ModelTrainingError(RainfallPredictionError):
    """Raised when model training fails."""
    pass


class FeatureEngineeringError(RainfallPredictionError):
    """Raised when feature engineering fails."""
    pass


# Alert and monitoring exceptions
class AlertError(CHSBDSException):
    """Raised when alert system encounters an error."""
    pass


class ThresholdExceededError(CHSBDSException):
    """Raised when a monitored value exceeds threshold."""

    def __init__(self, parameter: str, value: float, threshold: float, **kwargs):
        """
        Initialize threshold exceeded error.

        Args:
            parameter: Name of the parameter that exceeded threshold.
            value: Actual value.
            threshold: Threshold value.
        """
        message = f"{parameter} exceeded threshold: {value} > {threshold}"
        details = {'parameter': parameter, 'value': value, 'threshold': threshold}
        details.update(kwargs)
        super().__init__(message, details)


def handle_exceptions(default_return=None, raise_on_error=True):
    """
    Decorator for exception handling with logging.

    Args:
        default_return: Default value to return on exception (if raise_on_error=False).
        raise_on_error: Whether to re-raise exceptions after logging.

    Usage:
        @handle_exceptions(default_return=None, raise_on_error=True)
        def my_function():
            pass
    """
    import functools
    from .logger import get_logger

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            try:
                return func(*args, **kwargs)
            except CHSBDSException as e:
                logger.error(f"CHS-BDS Error in {func.__name__}: {e}")
                if raise_on_error:
                    raise
                return default_return
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}: {e}")
                if raise_on_error:
                    raise
                return default_return

        return wrapper

    return decorator


if __name__ == '__main__':
    # Test exception classes
    print("Testing Exception System...")
    print("-" * 50)

    # Test basic exception
    try:
        raise DataLoadError("Failed to load RINEX file", {'file': 'test.obs', 'line': 42})
    except CHSBDSException as e:
        print(f"Caught exception: {e}")

    # Test threshold exceeded
    try:
        raise ThresholdExceededError(
            parameter="displacement",
            value=0.05,
            threshold=0.01,
            station="STATION_A"
        )
    except ThresholdExceededError as e:
        print(f"\nCaught threshold error: {e}")
        print(f"Details: {e.details}")

    # Test exception handler decorator
    from .logger import setup_logger

    setup_logger(level='INFO')

    @handle_exceptions(default_return=None, raise_on_error=False)
    def failing_function():
        raise ProcessingError("Simulated processing error")

    result = failing_function()
    print(f"\nFunction returned: {result}")

    print("\n✅ Exception system test completed!")
