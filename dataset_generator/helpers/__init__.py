"""
Helper functions for signal preprocessing pipeline.

This module provides utility and domain-specific functions used by core classes:

Signal Processing:
- signal_filtering: Low-pass filtering implementations
- frequency_estimation: Sampling frequency estimation
- window_extraction: Window generation and extraction

Feature Engineering:
- feature_extraction: Feature computation from windows
- feature_selection: Feature selection and dimensionality reduction

Data Handling:
- data_labels: Activity label definitions and encoding
- dataset_builder: Dataset creation from windowed data
- splits: Train/val/test splitting logic
- window_fusion: Window aggregation and fusion

Utilities:
- utils: General utility functions
- constants: Constant definitions
"""

# Signal processing
from .signal_filtering import filter_signal, plot_filtered_signal
from .frequency_estimation import estimate_fs, timestamp_outlier_detection, accelerometer_outlier_detection, compute_acceleration_statistics, plot_frequency

# Window operations
from .window_extraction import (
    sliding_window_features,
    sliding_window_raw,
    generate_windows,
    generate_activity_windows,
)
from .window_fusion import fuse_feature_windows, fuse_raw_windows

# Feature engineering
from .feature_extraction import visualize_features, scatter_features
from .feature_selection import select_features_ensemble, fit_kda, transform_kda

# Data handling
from .data_labels import (
    label_accelerometer_data,
    build_label_summary,
    encode_labels,
    MOTOR1_ACTIVITIES,
    MOTOR2_ACTIVITIES,
)
from .dataset_builder import generate_feature_dataset_df, generate_raw_dataset_df
from .splits import grouped_stratified_split_groups

# Wavelet utilities
from .wavelet_utils import (
    compute_cwt_features,
    add_wavelet_features_to_window,
)

# Utilities
from .signal_preprocessing_utils import (
    get_results_subfolder,
    save_dataframe,
    load_session_result,
    concat_session_results,
    parse_session_folder_name,
    write_dataset_overwrite,
    load_session_windowed_df,
    combine_activity_totals,
    setup_logger,
)
from .constants import get_feature_columns
from .dataset_stats import (
    compute_dataset_activity_statistics,
    compute_dataset_phase_statistics,
    compute_dataset_assembly_statistics,
    compute_phase_assembly_duration_breakdown,
    compute_dataset_acceleration_statistics,
)

__all__ = [
    # Signal processing
    "filter_signal",
    "plot_filtered_signal",
    "estimate_fs",
    "timestamp_outlier_detection",
    "accelerometer_outlier_detection",
    "compute_acceleration_statistics",
    "plot_frequency",
    # Window operations
    "sliding_window_features",
    "sliding_window_raw",
    "generate_windows",
    "generate_activity_windows",
    "fuse_feature_windows",
    "fuse_raw_windows",
    # Feature engineering
    "visualize_features",
    "scatter_features",
    "select_features_ensemble",
    "fit_kda",
    "transform_kda",
    # Data handling
    "label_accelerometer_data",
    "build_label_summary",
    "encode_labels",
    "MOTOR1_ACTIVITIES",
    "MOTOR2_ACTIVITIES",
    "generate_feature_dataset_df",
    "generate_raw_dataset_df",
    "grouped_stratified_split_groups",
    # Wavelet utilities
    "compute_cwt_features",
    "add_wavelet_features_to_window",
    # Utilities
    "get_results_subfolder",
    "save_dataframe",
    "load_session_result",
    "concat_session_results",
    "parse_session_folder_name",
    "write_dataset_overwrite",
    "load_session_windowed_df",
    "combine_activity_totals",
    "setup_logger",
    "get_feature_columns",
    "compute_dataset_activity_statistics",
    "compute_dataset_phase_statistics",
    "compute_dataset_assembly_statistics",
    "compute_phase_assembly_duration_breakdown",
    "compute_dataset_acceleration_statistics",
]
