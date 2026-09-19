from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .signal_preprocessing_config import (
    PreprocessingConfig,
    OUTPUT_DIR,
    PROCESSED_SESSION_DATA_DIR,
)
from ..helpers import (
    load_session_windowed_df,
    load_session_result,
    timestamp_outlier_detection,
    accelerometer_outlier_detection,
    compute_acceleration_statistics,
    estimate_fs,
    plot_frequency,
    label_accelerometer_data,
    build_label_summary,
    filter_signal,
    plot_filtered_signal,
    sliding_window_features,
    sliding_window_raw,
    visualize_features,
    scatter_features,
    save_dataframe
)
from .activity_manager import ActivityManager
from .window_generator import WindowGenerator


class SessionProcessor:
    """
    Processes a single experimental session through the preprocessing pipeline.
    
    This class encapsulates all logic for processing one session, from raw
    data to windowed features. It manages session-specific state including
    file paths, caching decisions, and progress tracking.
    
    The processor orchestrates the full preprocessing pipeline:
    1. Data loading and cleaning
    2. Frequency estimation
    3. Activity labeling
    4. Signal filtering
    5. Window generation
    6. Feature and raw data extraction
    
    Session State:
        The processor tracks paths for:
        - Raw data files (watch data, button logs)
        - Results folder (where intermediate results are saved)
        - Processed outputs (windowed features and raw data)
    
    Caching:
        The processor can detect when a session has already been processed
        and skip reprocessing. Reprocessing is forced when:
        - Reprocess flag is enabled in config
        - Activity grouping is enabled
        - Full activity windowing is enabled
        - Results files are missing
    
    Attributes:
        session_folder (Path): Folder containing the session's raw data.
        session_code (str): Session identifier (e.g., "1", "14", "lab1").
        order_condition (str): Experimental order (e.g., "O1" or "O2").
        config (PreprocessingConfig): Configuration instance.
        activity_manager (ActivityManager): Shared activity manager.
        window_generator (WindowGenerator): Shared window generator.
        data_file (Path): Path to watch data CSV.
        button_file (Path): Path to button log CSV.
        results_folder (Path): Path to results subfolder.
        windowed_features_file (Path): Path to cached features output.
        windowed_raw_file (Path): Path to cached raw data output.
    
    Example:
        Basic usage with caching::
        
            processor = SessionProcessor(
                session_folder=Path("Raw_Data/session_1_O1_data"),
                processed_session_folder=Path("Processed_Data/Processed_Session_Data/session_1_O1_data"),
                session_code="1",
                order_condition="O1",
                config=config,
                activity_manager=activity_manager,
                window_generator=window_gen
            )
            
            if processor.should_reprocess():
                results = processor.process()
            else:
                results = processor.load_cached_results()
    """
    
    def __init__(
        self,
        session_folder: Path,
        processed_session_folder: Path | None = None,
        session_code: str = "",
        order_condition: str = "",
        config: PreprocessingConfig | None = None,
        activity_manager: ActivityManager | None = None,
        window_generator: WindowGenerator | None = None
    ):
        """
        Initialize the SessionProcessor.
        
        Sets up all session-specific paths and state. Paths are derived
        from the session folder and session identifiers following the
        project's naming convention.
        
        Parameters:
            session_folder (Path): Folder containing the session's raw data.
                Must be named like "session_1_O1_data".
            session_code (str): Session identifier for naming files
                (e.g., "1", "14", "lab1").
            order_condition (str): Experimental order condition
                (e.g., "O1" or "O2").
            config (PreprocessingConfig): Configuration instance controlling
                processing behavior.
            activity_manager (ActivityManager): Activity manager instance
                for handling activity grouping and motor separation.
            window_generator (WindowGenerator): Window generator instance
                for creating window definitions.
        
        Example:
            >>> config = PreprocessingConfig()
            >>> processor = SessionProcessor(
            ...     session_folder=Path("Raw_Data/session_1_O1_data"),
            ...     processed_session_folder=Path("Processed_Data/Processed_Session_Data/session_1_O1_data"),
            ...     session_code="1",
            ...     order_condition="O1",
            ...     config=config,
            ...     activity_manager=ActivityManager(config),
            ...     window_generator=WindowGenerator(config)
            ... )
        """
        self.session_folder = session_folder
        self.processed_session_folder = processed_session_folder or (
            session_folder.parent.parent / OUTPUT_DIR / PROCESSED_SESSION_DATA_DIR / session_folder.name
        )
        self.session_code = session_code
        self.order_condition = order_condition
        self.config = config
        self.activity_manager = activity_manager
        self.window_generator = window_generator
        
        # Derive file paths following project naming convention
        self.data_file = session_folder / f"session_{session_code}_{order_condition}_watch_data.csv"
        self.button_file = session_folder / f"session_{session_code}_{order_condition}_button_log.csv"
        self.results_folder = self.processed_session_folder / f"session_{session_code}_{order_condition}_results"
        self.windowed_features_file = self.processed_session_folder / f"session_{session_code}_{order_condition}_windowed_features.csv"
        self.windowed_raw_file = self.processed_session_folder / f"session_{session_code}_{order_condition}_windowed_raw.csv"
    
    def should_reprocess(self) -> bool:
        """
        Determine if this session needs reprocessing.
        
        Returns True if any of these conditions are met:
        - Global reprocess flag is enabled in config
        - Results folder doesn't exist
        - Windowed features file doesn't exist
        - Windowed raw file doesn't exist
        - Activity grouping is enabled (forces reprocessing)
        - Full activity windowing is enabled (forces reprocessing)
        
        Returns:
            bool: True if session should be reprocessed, False if cached
                results can be used.
        
        Example:
            >>> if processor.should_reprocess():
            ...     features, raw = processor.process()
            ... else:
            ...     features, raw = processor.load_cached_results()
        """
        # Force reprocessing if global flag is enabled
        if getattr(self.config, "reprocess_all_sessions", False):
            return True

        # Check if all required output files exist
        if not (self.results_folder.exists() and 
                self.windowed_features_file.exists() and 
                self.windowed_raw_file.exists()):
            return True
        
        # Force reprocessing if certain features are enabled
        if self.config.activity_grouping or self.config.full_activity_windowing:
            return True
        
        return False
    
    def load_cached_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load previously computed results.
        
        Loads windowed data and summary statistics that were previously
        computed and saved during a full preprocessing run. This allows
        skipping expensive reprocessing when the input hasn't changed.
        
        Returns:
            tuple: A 7-tuple containing:
                - features_df (pd.DataFrame): Windowed features
                - raw_df (pd.DataFrame): Windowed raw data
                - activity_totals_df (pd.DataFrame | None): Activity duration totals
                - activity_segments_df (pd.DataFrame | None): Activity segments with timing
                - phase_stats_df (pd.DataFrame | None): Phase statistics
                - assembly_stats_df (pd.DataFrame | None): Assembly statistics
                - acceleration_stats_df (pd.DataFrame | None): Motor-specific acceleration stats
        
        Raises:
            FileNotFoundError: If cached files don't exist (should check
                should_reprocess() first).
        
        Example:
            >>> features, raw, activity_totals, activity_segments, *stats = processor.load_cached_results()
            >>> print(f"Loaded {len(features)} feature windows")
        """
        
        print(f"  Loading cached results for session {self.session_code}_{self.order_condition}")
        
        # Load windowed features and raw data from the processed session folder
        features_df = load_session_windowed_df(
            self.processed_session_folder, self.session_code, self.order_condition, "features"
        )
        raw_df = load_session_windowed_df(
            self.processed_session_folder, self.session_code, self.order_condition, "raw"
        )
        
        # Load summary statistics
        activity_totals_df = load_session_result(
            self.processed_session_folder, self.session_code, self.order_condition, "activity_totals"
        )
        activity_segments_df = load_session_result(
            self.processed_session_folder, self.session_code, self.order_condition, "activity_segments"
        )
        phase_stats_df = load_session_result(
            self.processed_session_folder, self.session_code, self.order_condition, "phase_stats"
        )
        assembly_stats_df = load_session_result(
            self.processed_session_folder, self.session_code, self.order_condition, "assembly_stats"
        )
        acceleration_stats_df = load_session_result(
            self.processed_session_folder, self.session_code, self.order_condition, "acceleration_motor_stats"
        )
        
        return features_df, raw_df, activity_totals_df, activity_segments_df, phase_stats_df, assembly_stats_df, acceleration_stats_df
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the full preprocessing pipeline for this session.
        
        Orchestrates all preprocessing steps in sequence:
        1. Load raw data
        2. Clean data (outlier detection)
        3. Estimate sampling frequency
        4. Label activities from button log
        5. Create label summaries
        6. Filter signals
        7. Generate windows
        8. Extract features
        9. Extract raw data
        10. Visualize results
        
        All intermediate results are saved to disk.
        
        Returns:
            tuple: A 7-tuple containing:
                - features_df (pd.DataFrame): Windowed features
                - raw_df (pd.DataFrame): Windowed raw data
                - phase_stats_df (pd.DataFrame): Phase statistics
                - assembly_stats_df (pd.DataFrame): Assembly statistics
                - activity_totals_df (pd.DataFrame): Activity totals
                - activity_segments_df (pd.DataFrame): Activity segments with timing
                - acceleration_stats_df (pd.DataFrame): Motor-specific acceleration stats
        
        Raises:
            FileNotFoundError: If raw data or button log files don't exist.
            ValueError: If validation checks fail (empty dataframes, missing columns).
        
        Example:
            >>> (features, raw, phase_stats, assembly_stats,
            ...  activity_totals, activity_segments) = processor.process()
            >>> print(f"Processed: {len(features)} windows")
        """
        print(f"\nProcessing session {self.session_code}_{self.order_condition}")
        
        # ===== VALIDATION =====
        # Validate files exist
        if not self.data_file.exists():
            raise FileNotFoundError(f"File not found: {self.data_file.resolve()}")
        if not self.button_file.exists():
            raise FileNotFoundError(f"File not found: {self.button_file.resolve()}")
        
        # ===== STEP 1: Load Data =====
        print(f"  Reading sensor data: {self.data_file.name}")
        df = pd.read_csv(self.data_file)
        
        print(f"  Reading button log: {self.button_file.name}")
        button_log = pd.read_csv(self.button_file)
        button_log.columns = button_log.columns.str.lower()
        button_log["timestamp"] = pd.to_datetime(button_log["timestamp"])
        
        # Create results folder
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        # ===== STEP 2: Clean Data =====
        clean_df = timestamp_outlier_detection(df, results_folder=self.results_folder)
        clean_df = accelerometer_outlier_detection(clean_df, results_folder=self.results_folder)
        if clean_df.empty:
            raise ValueError(f"Outlier detection resulted in empty dataframe")
        
        # Compute and save acceleration statistics
        compute_acceleration_statistics(clean_df, results_folder=self.results_folder)
        
        # Validate required columns
        required_cols = ["x", "y", "z", "mac", "abs_time"]
        missing = [c for c in required_cols if c not in clean_df.columns]
        if missing:
            raise ValueError(f"Missing required columns after cleaning: {missing}")
        
        # ===== STEP 3: Frequency Estimation =====
        freq_time, avg_freq = estimate_fs(clean_df, self.results_folder)
        plot_frequency(freq_time, self.results_folder)
        
        # ===== STEP 4: Labeling =====
        labelled_df = label_accelerometer_data(clean_df, button_log, self.order_condition)
        
        # Apply activity grouping if enabled
        if self.config.activity_grouping:
            labelled_df = self.activity_manager.apply_activity_grouping(labelled_df)
        
        self.processed_session_folder.mkdir(parents=True, exist_ok=True)
        save_dataframe(
            labelled_df, self.processed_session_folder,
            f"session_{self.session_code}_{self.order_condition}_labelled"
        )
        
        # ===== STEP 5: Label Summary =====
        summary_dict = build_label_summary(labelled_df)
        activity_segments_df = summary_dict["activity_segments"]
        phase_segments_df = summary_dict["phase_segments"]
        assembly_segments_df = summary_dict["assembly_segments"]
        phase_stats_df = summary_dict["phase_stats"]
        assembly_stats_df = summary_dict["assembly_stats"]
        activity_totals_df = summary_dict["activity_totals"]
        
        for name, df_out in summary_dict.items():
            save_dataframe(
                df_out, self.results_folder,
                f"session_{self.session_code}_{self.order_condition}_{name}"
            )

        # ===== STEP 6: Motor-Specific Acceleration Statistics =====
        # Build motor-specific acceleration statistics from returned function output
        acceleration_rows = []
        grouped = labelled_df.groupby("motor_type", dropna=False)

        for motor_type, motor_df in grouped:
            # Compute acceleration statistics for this motor type separately
            motor_stats = compute_acceleration_statistics(motor_df, results_folder=None, verbose=False)
            # For each axis (x, y, z and magnitude), create a row with motor type, axis, session code, and stats
            for axis, axis_stats in motor_stats.items():
                row = {
                    "motor_type": motor_type,
                    "axis": axis,
                    "session_code": self.session_code,
                }
                # Add the dict of axis stats to the row
                row.update(axis_stats)
                acceleration_rows.append(row)

        acceleration_stats_df = pd.DataFrame(acceleration_rows)
        save_dataframe(
            acceleration_stats_df,
            self.results_folder,
            f"session_{self.session_code}_{self.order_condition}_acceleration_motor_stats"
        )
        
        # ===== STEP 7: Signal Filtering =====
        filtered_df = filter_signal(labelled_df, avg_freq, cutoff=self.config.cutoff)

        # Save a plot of the filtered signal for each watch for Phase 1, Activity 3
        plot_filtered_signal(filtered_df, self.results_folder, assembly_id=1,phase_id=1, activity_id=3)
        
        if filtered_df.empty:
            raise ValueError(f"Signal filtering resulted in empty dataframe")
        
        required_motion_cols = [f"{axis}_motion_smooth" for axis in ["x", "y", "z"]]
        missing = [c for c in required_motion_cols if c not in filtered_df.columns]
        if missing:
            raise ValueError(f"Missing required columns after filtering: {missing}")
        
        # Add temporary label_encoded column for soft labels during windowing
        # (Global encoding across all sessions happens later in dataset_manager)
        if self.config.use_soft_labels and "label" in filtered_df.columns:
            unique_labels = sorted(filtered_df["label"].dropna().unique())
            le = LabelEncoder()
            le.fit(unique_labels)
            filtered_df["label_encoded"] = le.transform(filtered_df["label"])
            print(f"  Created temporary label encoding for soft labels ({len(unique_labels)} classes)")
        
        # ===== STEP 8: Window Generation =====
        window_definitions, num_windows_generated = self.window_generator.generate_windows(
            activity_segments=activity_segments_df,
            assembly_segments=assembly_segments_df,
            phase_segments=phase_segments_df
        )
        
        # Determine if non-learning session
        non_learning_session = self.config.is_non_learning_session(self.session_code)
        print(f"  Non-learning session: {non_learning_session}")
        
        # ===== STEP 9: Feature Extraction =====
        features_df = sliding_window_features(
            filtered_df,
            results_folder=self.results_folder,
            window_defs=window_definitions,
            avg_freq=avg_freq,
            session_code=self.session_code,
            num_windows_generated=num_windows_generated,
            first_assemblies_to_remove=self.config.first_assemblies_to_remove,
            non_learning_session=non_learning_session,
            use_soft_labels=self.config.use_soft_labels
        )
        
        # ===== STEP 10: Raw Data Extraction =====
        raw_df = sliding_window_raw(
            filtered_df,
            results_folder=self.results_folder,
            window_defs=window_definitions,
            session_code=self.session_code,
            num_windows_generated=num_windows_generated,
            first_assemblies_to_remove=self.config.first_assemblies_to_remove,
            non_learning_session=non_learning_session,
            use_soft_labels=self.config.use_soft_labels,
            add_wavelet_features=self.config.add_wavelet_features,
            wavelet_family=self.config.wavelet_family,
            wavelet_scales=self.config.wavelet_scales
        )
        
        # ===== VALIDATION =====
        if features_df.empty:
            raise ValueError(f"Feature extraction resulted in empty dataframe")
        if raw_df.empty:
            raise ValueError(f"Raw data extraction resulted in empty dataframe")
        
        # ===== STEP 11: Save and Visualize =====
        save_dataframe(
            features_df, self.data_file.parent,
            f"session_{self.session_code}_{self.order_condition}_windowed_features"
        )
        save_dataframe(
            raw_df, self.data_file.parent,
            f"session_{self.session_code}_{self.order_condition}_windowed_raw"
        )
        
        visualize_features(features_df, results_folder=self.results_folder)
        scatter_features(features_df, 'x_motion_smooth_rms', 'y_motion_smooth_rms', 
                        results_folder=self.results_folder)
        
        print(f" Preprocessing complete")
        print(f"    - Feature windows: {len(features_df)}")
        print(f"    - Raw windows: {len(raw_df)}")
        
        return (features_df, raw_df, phase_stats_df, assembly_stats_df,
            activity_totals_df, activity_segments_df, acceleration_stats_df)
