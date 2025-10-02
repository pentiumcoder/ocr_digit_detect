"""
Custom exceptions for Digital Number OCR system.
"""


class OCRException(Exception):
    """Base exception for OCR system."""
    pass


class ModelLoadError(OCRException):
    """Exception raised when model fails to load."""
    pass


class ImageLoadError(OCRException):
    """Exception raised when image fails to load."""
    pass


class DetectionError(OCRException):
    """Exception raised when detection fails."""
    pass


class RecognitionError(OCRException):
    """Exception raised when OCR recognition fails."""
    pass


class PreprocessingError(OCRException):
    """Exception raised during preprocessing."""
    pass


class ValidationError(OCRException):
    """Exception raised during result validation."""
    pass


class ConfigurationError(OCRException):
    """Exception raised for configuration issues."""
    pass


class InsufficientDataError(OCRException):
    """Exception raised when insufficient data is available."""
    pass
