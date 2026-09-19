"""
Signal preprocessing pipeline for HAR (Human Activity Recognition).

This package provides a complete preprocessing pipeline for smartwatch sensor data,
including signal filtering, windowing, feature extraction, and dataset building.

Core Architecture:
    - core: Classes orchestrating the pipeline
    - helpers: Helper functions for signal processing and data handling

Example:
    Basic usage with the architecture::
    
        from signal_preprocessing import (
            PreprocessingConfig,
            ActivityManager,
            SessionProcessor,
            DatasetBuilder
        )
        
        # Create configuration
        config = PreprocessingConfig()
        
        # Create pipeline components
        activity_mgr = ActivityManager(config)
        session_proc = SessionProcessor(config, activity_mgr, session_folder)
        
        # Process session
        features_df, raw_df, *stats = session_proc.process()
"""

# Core classes
from .core import (
    PreprocessingConfig,
    ActivityManager,
    WindowGenerator,
    SessionProcessor,
    DatasetBuilder,
)

# Helper functions (commonly used)
from .helpers import (
    filter_signal,
    estimate_fs,
    generate_windows,
    visualize_features,
    encode_labels,
    get_feature_columns,
    save_dataframe,
    load_session_result,
)

__all__ = [
    # Core classes
    "PreprocessingConfig",
    "ActivityManager",
    "WindowGenerator",
    "SessionProcessor",
    "DatasetBuilder",
    # Common helpers
    "filter_signal",
    "estimate_fs",
    "generate_windows",
    "visualize_features",
    "encode_labels",
    "get_feature_columns",
    "save_dataframe",
    "load_session_result",
]


