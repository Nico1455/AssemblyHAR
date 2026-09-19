"""
This module provides the main classes that orchestrate the preprocessing pipeline:
- PreprocessingConfig: Centralized configuration management
- ActivityManager: Activity labeling and motor-based operations
- WindowGenerator: Window extraction strategies
- SessionProcessor: Single session preprocessing pipeline
- DatasetBuilder: Multi-session dataset aggregation
"""

from .signal_preprocessing_config import PreprocessingConfig
from .activity_manager import ActivityManager
from .window_generator import WindowGenerator
from .session_processor import SessionProcessor
from .dataset_manager import DatasetBuilder

__all__ = [
    "PreprocessingConfig",
    "ActivityManager",
    "WindowGenerator",
    "SessionProcessor",
    "DatasetBuilder",
]
