from __future__ import annotations
import pandas as pd
import pickle
import shutil
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder
from math import floor
import os

from .signal_preprocessing_config import PreprocessingConfig
from .activity_manager import ActivityManager
from ..helpers import (
    encode_labels,
    select_features_ensemble,
    fit_kda,
    transform_kda,
    save_dataframe,
    write_dataset_overwrite,
    get_feature_columns,
    generate_raw_dataset_df,
    generate_feature_dataset_df,
    compute_dataset_activity_statistics,
    compute_dataset_phase_statistics,
    compute_dataset_assembly_statistics,
    compute_phase_assembly_duration_breakdown,
    compute_dataset_acceleration_statistics,
)


class DatasetBuilder:
    """
    Dataset building for the preprocessing pipeline.

    This module handles the aggregation of multiple processed sessions into
    final train/val/test datasets. It manages all cross-session operations
    including dataset combining, splitting, feature selection, and motor separation.

    The DatasetBuilder class orchestrates the final stages of the preprocessing
    pipeline, taking individual session results and creating the final datasets
    used for model training and evaluation.

    Responsibilities:
        - Collect processed sessions from SessionProcessor
        - Combine and summarize statistics across sessions
        - Generate stratified train/val/test splits
        - Apply feature selection and KDA transformation
        - Handle motor-separated dataset creation
        - Manage dataset folder structure

    Example:
        Create a dataset builder and process multiple sessions::

            config = PreprocessingConfig.from_config_module(config_module)
            activity_manager = ActivityManager(config)
            builder = DatasetBuilder(config, activity_manager, project_root)
            
            # Process sessions and add results
            for folder in data_folder.iterdir():
                processor = SessionProcessor(...)
                features, raw, *stats = processor.process()
                builder.add_session_data(features, raw, *stats, session_code="1")
            
            # Build final datasets
            builder.build_datasets()
    
    The builder maintains separate aggregation pipelines for:
    - Feature datasets (extracted features from signal)
    - Raw datasets (raw IMU values from windows)
    
    Attributes:
        config (PreprocessingConfig): Configuration controlling dataset building.
        activity_manager (ActivityManager): Activity manager for motor separation.
        project_root (Path): Project root directory.
        data_folder (Path): Path to Data folder with raw sessions.
        datasets_folder (Path): Path to Datasets folder for outputs.
        dataset_root_features (Path): Output folder for feature datasets.
        dataset_root_raw (Path): Output folder for raw datasets.
        all_sessions_feat (list[pd.DataFrame]): Collected feature dataframes.
        all_sessions_raw (list[pd.DataFrame]): Collected raw dataframes.
        all_activity_totals (list[pd.DataFrame]): Activity duration summaries.
        all_activity_segments (list[pd.DataFrame]): Activity time segments.
        all_phase_stats (list[pd.DataFrame]): Phase statistics.
        all_assembly_stats (list[pd.DataFrame]): Assembly statistics.
    """
    
    def __init__(
        self,
        config: PreprocessingConfig,
        activity_manager: ActivityManager,
        project_root: Path
    ):
        """
        Initialize the DatasetBuilder.
        
        Creates necessary folder structure and sets up aggregation lists
        for collecting session data.
        
        Parameters:
            config (PreprocessingConfig): Configuration controlling dataset
                building behavior (splits, motor separation, etc.).
            activity_manager (ActivityManager): Activity manager for handling
                motor separation and activity filtering.
            project_root (Path): Project root directory. Expected to contain
                Data/ and Datasets/ folders.
        
        Example:
            >>> config = PreprocessingConfig()
            >>> activity_mgr = ActivityManager(config)
            >>> builder = DatasetBuilder(config, activity_mgr, Path("."))
        """
        self.config = config
        self.activity_manager = activity_manager
        self.project_root = project_root
        self.data_folder = project_root / "Data"
        self.datasets_folder = project_root / "Datasets"
        
        # Create datasets folder
        self.datasets_folder.mkdir(parents=True, exist_ok=True)
        
        # Get dataset folder names based on windowing configuration
        dataset_features_name, dataset_raw_name = config.get_dataset_folder_names()
        self.dataset_root_features = self.datasets_folder / dataset_features_name
        self.dataset_root_raw = self.datasets_folder / dataset_raw_name
        
        # Create dataset folders (only if not using motor separation)
        # Motor-separated folders are created later in _save_motor_separated_datasets
        if not config.separate_motors:
            self.dataset_root_features.mkdir(parents=True, exist_ok=True)
            self.dataset_root_raw.mkdir(parents=True, exist_ok=True)
        
        # Initialize aggregation lists
        self.all_sessions_feat: List[pd.DataFrame] = []
        self.all_sessions_raw: List[pd.DataFrame] = []
        self.all_activity_totals: List[pd.DataFrame] = []
        self.all_activity_segments: List[pd.DataFrame] = []
        self.all_phase_stats: List[pd.DataFrame] = []
        self.all_assembly_stats: List[pd.DataFrame] = []
        self.all_acceleration_stats: List[pd.DataFrame] = []
        
        # Global statistics (computed in build_datasets)
        self.global_activity_totals_df: Optional[pd.DataFrame] = None
        self.global_activity_segments_df: Optional[pd.DataFrame] = None
        self.global_phase_stats_df: Optional[pd.DataFrame] = None
        self.global_assembly_stats_df: Optional[pd.DataFrame] = None
        self.global_phase_assembly_breakdown_df: Optional[pd.DataFrame] = None
        self.global_acceleration_stats_df: Optional[pd.DataFrame] = None
    
    def add_session_data(
        self,
        features_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        activity_totals_df: Optional[pd.DataFrame],
        activity_segments_df: Optional[pd.DataFrame],
        phase_stats_df: Optional[pd.DataFrame],
        assembly_stats_df: Optional[pd.DataFrame],
        acceleration_stats_df: Optional[pd.DataFrame],
        session_code: str
    ):
        """
        Add processed session data to the aggregation lists.
        
        Collects windowed data and statistics from a single session.
        All dataframes are stored for later combination during build_datasets().
        
        Parameters:
            features_df (pd.DataFrame): Windowed features from the session.
                Expected columns: feature columns (L_*, R_*), label, label_encoded,
                window_id, mac, window_start, window_end.
            raw_df (pd.DataFrame): Windowed raw accelerometer data from the session.
                Similar structure to features_df but with raw accelerometer sensor values.
            activity_totals_df (pd.DataFrame | None): Summary of total durations
                per activity. If None, is skipped.
            activity_segments_df (pd.DataFrame | None): Time segments for each
                activity. If None, is skipped.
            phase_stats_df (pd.DataFrame | None): Phase-level statistics.
                If None, is skipped.
            assembly_stats_df (pd.DataFrame | None): Assembly-level statistics.
                If None, is skipped.
            acceleration_stats_df (pd.DataFrame | None): Motor-specific accelerometer
                summary statistics. If None, is skipped.
            session_code (str): Session identifier to add to all dataframes
                for traceability.
        
        Example:
            >>> builder.add_session_data(
            ...     features, raw, activity_totals, activity_segments,
            ...     phase_stats, assembly_stats,
            ...     session_code="1"
            ... )
        """
        self.all_sessions_feat.append(features_df)
        self.all_sessions_raw.append(raw_df)
        
        if activity_totals_df is not None:
            activity_totals_df["session_code"] = session_code
            self.all_activity_totals.append(activity_totals_df)
        
        if phase_stats_df is not None:
            phase_stats_df["session_code"] = session_code
            self.all_phase_stats.append(phase_stats_df)
        
        if assembly_stats_df is not None:
            assembly_stats_df["session_code"] = session_code
            self.all_assembly_stats.append(assembly_stats_df)
        
        if activity_segments_df is not None:
            activity_segments_df["session_code"] = session_code
            self.all_activity_segments.append(activity_segments_df)

        if acceleration_stats_df is not None:
            if "session_code" not in acceleration_stats_df.columns:
                acceleration_stats_df["session_code"] = session_code
            self.all_acceleration_stats.append(acceleration_stats_df)
    
    def build_datasets(self):
        """
        Build final train/val/test datasets from all collected sessions.
        
        Orchestrates the complete dataset building process:
        1. Combines all session data
        2. Combines and aggregates statistics
        3. Builds feature datasets (with feature selection and KDA)
        4. Builds raw datasets
        5. Handles motor separation if enabled
        
        This method calls internal methods that handle both feature and raw
        dataset building. Each can be configured independently and both
        produce identical structure but with different data.
        
        Example:
            >>> builder.build_datasets()
            # Datasets now at:
            # - Dataset_features_wsi1_ssi0_25/{train,val,test}.csv
            # - Dataset_raw_wsi1_ssi0_25/{train,val,test}.csv
        """
        
        # ===== Combine summary statistics =====
        print("\nCombining activity segments and totals across all sessions...")
        # Combine activity segments and totals across all sessions
        global_activity_segments_df, global_activity_totals_df = compute_dataset_activity_statistics(
            self.all_activity_segments,
            self.all_activity_totals,
        )

        # Combine phase stats (aggregate across all sessions)
        global_phase_stats_df = compute_dataset_phase_statistics(
            self.all_phase_stats,
            global_activity_segments_df,
        )

        # Compute per-phase/per-assembly duration breakdown from activity segments
        global_phase_assembly_breakdown_df = compute_phase_assembly_duration_breakdown(
            global_activity_segments_df,
            phase_ids=(1, 2, 3, 4),
            phase1_assembly_ids=(1, 2, 3),
        )

        # Attach breakdown columns to phase totals (one row per motor_type)
        if not global_phase_stats_df.empty and not global_phase_assembly_breakdown_df.empty:
            global_phase_stats_df = global_phase_stats_df.merge(
                global_phase_assembly_breakdown_df,
                on="motor_type",
                how="left",
            )
        
        # Combine assembly stats (aggregate across all sessions)
        global_assembly_stats_df = compute_dataset_assembly_statistics(
            self.all_assembly_stats,
            global_activity_segments_df,
        )
        
        # Store global statistics as instance variables for use in _build_*_datasets methods
        self.global_activity_totals_df = global_activity_totals_df
        self.global_activity_segments_df = global_activity_segments_df
        self.global_phase_stats_df = global_phase_stats_df
        self.global_assembly_stats_df = global_assembly_stats_df
        self.global_phase_assembly_breakdown_df = global_phase_assembly_breakdown_df
        # Aggregate acceleration stats across all sessions
        self.global_acceleration_stats_df = compute_dataset_acceleration_statistics(self.all_acceleration_stats)
        
        # Build feature and raw datasets
        self._build_feature_datasets()
        self._build_raw_datasets()
    
    def _build_feature_datasets(self):
        """
        Build feature datasets with feature selection and KDA.
        
        Processes extracted features including:
        - Concatenating sessions
        - Encoding labels
        - Creating train/val/test splits
        - Saving statistics summaries
        - Applying KDA if enabled
        - Running feature selection
        - Handling motor separation
        
        Uses instance variables: global_activity_totals_df, global_activity_segments_df,
        global_phase_stats_df, global_assembly_stats_df computed in build_datasets().
        """
        
        try:
            # Concatenate all session windowed data
            all_sessions_feat_df = pd.concat(
                self.all_sessions_feat, ignore_index=True, sort=False
            )
            # Disabled: all_sessions_windowed_features is massive and slows down copying processes
            # save_dataframe(
            #     all_sessions_feat_df, self.dataset_root_features,
            #     "all_sessions_windowed_features"
            # )
            print(f"\nCollected windowed features from {len(self.all_sessions_feat)} sessions; "
                  f"total windows: {len(all_sessions_feat_df)}")
            
            # If not separating motors, encode labels once and save encoder to base folder
            if not self.config.separate_motors:
                # Ensure target folder exists
                self.dataset_root_features.mkdir(parents=True, exist_ok=True)
                
                # Add encoded label column and persist encoder
                all_sessions_feat_df = encode_labels(
                    all_sessions_feat_df,
                    results_folder=self.dataset_root_features,
                    str_label_column="label"
                )
            
            # Generate dataset splits
            # In motor-separated mode, write split logs to motor_1 folder and
            # mirror them to motor_2 later.
            if self.config.separate_motors:
                split_log_dir = self.datasets_folder / f"{self.dataset_root_features.name}_motor_1"
                split_log_dir.mkdir(parents=True, exist_ok=True)
            else:
                split_log_dir = self.dataset_root_features
            print("\nGenerating dataset splits from pooled sessions...")
            train_df, val_df, test_df = generate_feature_dataset_df(
                all_sessions_feat_df,
                str(split_log_dir),
                left_macs=self.config.left_macs,
                right_macs=self.config.right_macs,
                train_frac=self.config.train_frac,
                val_frac=self.config.val_frac,
                test_frac=self.config.test_frac,
                tolerance=self.config.tolerance,
                max_attempts=self.config.max_attempts,
                random_state=self.config.random_state,
                shuffle_within_split=self.config.shuffle_within_split
            )
            
            # Save summary statistics (activity totals, phase totals, assembly totals, class counts)
            # In motor-separated mode, write these into the motor_1 folder and mirror
            # the feature-selection artifacts to motor_2 later.
            if self.config.separate_motors:
                stats_root = self.datasets_folder / f"{self.dataset_root_features.name}_motor_1"
            else:
                stats_root = self.dataset_root_features

            self._save_dataset_statistics(
                train_df, val_df, test_df,
                stats_root,
                self.global_activity_totals_df,
                self.global_activity_segments_df,
                self.global_phase_stats_df,
                self.global_assembly_stats_df,
                self.global_phase_assembly_breakdown_df,
                self.global_acceleration_stats_df,
                self.config.window_size_sec,
                self.config.step_size_sec
            )
            
            # Apply KDA if enabled
            if self.config.use_kda:
                print("\nApplying KDA transformation...")
                feature_cols = get_feature_columns(train_df)
                
                kda_model = fit_kda(
                    train_df[feature_cols],
                    train_df["label_encoded"],
                    n_components=self.config.kda_n_components,
                    gamma=self.config.kda_gamma,
                    results_folder=stats_root
                )
                
                train_df = transform_kda(train_df, kda_model, feature_cols)
                val_df = transform_kda(val_df, kda_model, feature_cols)
                test_df = transform_kda(test_df, kda_model, feature_cols)
                print("[KDA] Skipping feature selection (KDA already provides dimensionality reduction)")
            
            else:
                # Feature selection only if NOT using KDA
                print("\nRunning ensemble feature selection on training data...")
                train_df, selected_features = select_features_ensemble(
                    df=train_df,
                    results_folder=stats_root,
                    window_size_sec=self.config.window_size_sec,
                    step_size_sec=self.config.step_size_sec,
                    phase_totals=self.global_phase_stats_df,
                    assembly_totals=self.global_assembly_stats_df,
                    activity_totals=self.global_activity_totals_df,
                    activity_segments=self.global_activity_segments_df,
                    cumulative_threshold=self.config.cumulative_threshold
                )

                if self.config.separate_motors:
                    motor2_stats_root = self.datasets_folder / f"{self.dataset_root_features.name}_motor_2"
                    motor2_stats_root.mkdir(parents=True, exist_ok=True)
                    for filename in ("feature_ranking.csv", "selected_features.txt"):
                        source_path = stats_root / filename
                        target_path = motor2_stats_root / filename
                        if source_path.exists():
                            shutil.copy2(source_path, target_path)
                
                # Apply same feature selection to val and test
                metadata_cols = [c for c in val_df.columns if c not in get_feature_columns(val_df)]
                val_df = val_df[metadata_cols + selected_features]
                test_df = test_df[metadata_cols + selected_features]
                print(f"Applied feature selection to val/test: kept {len(selected_features)} features")
            
            # Add motor_id column to all splits
            print("\nAdding motor_id column to feature datasets...")
            train_df = self.activity_manager.assign_motor_ids(train_df)
            val_df = self.activity_manager.assign_motor_ids(val_df)
            test_df = self.activity_manager.assign_motor_ids(test_df)
            print("motor_id column added (1=motor1, 2=motor2)")
            
            # Save final datasets
            if self.config.separate_motors:
                # Motor-specific label encoders are created inside this method;
                # no encoder is written to the common base folder.
                self._save_motor_separated_datasets(
                    train_df, val_df, test_df,
                    self.dataset_root_features, "features",
                    save_statistics=True
                )
            else:
                write_dataset_overwrite(
                    train_df, val_df, test_df,
                    self.dataset_root_features
                )
            
            print("\nFeature dataset generation complete.")
            
        except Exception as e:
            if not self.all_sessions_feat:
                print("No session windowed feature data collected. Exiting.")
            else:
                print(f"Failed during feature dataset generation: {e}")
                raise
     
    def _build_raw_datasets(self):
        """
        Build raw data datasets.
        
        Processes raw IMU values including:
        - Concatenating sessions
        - Encoding labels
        - Creating train/val/test splits
        - Handling motor separation
        
        Uses instance variables: global_activity_totals_df, global_activity_segments_df,
        global_phase_stats_df, global_assembly_stats_df computed in build_datasets().
        """       
        try:
            # Concatenate all session windowed data
            all_sessions_raw_df = pd.concat(
                self.all_sessions_raw, ignore_index=True, sort=False
            )
            print(f"\nCollected windowed raw data from {len(self.all_sessions_raw)} sessions; "
                  f"total windows: {len(all_sessions_raw_df)}")
            
            # If not separating motors, encode labels once and save encoder to base folder
            if not self.config.separate_motors:
                # Ensure target folder exists
                self.dataset_root_raw.mkdir(parents=True, exist_ok=True)
                
                # Add encoded label column and persist encoder
                all_sessions_raw_df = encode_labels(
                    all_sessions_raw_df,
                    results_folder=self.dataset_root_raw,
                    str_label_column="label"
                )
            
            # Generate dataset splits
            # In motor-separated mode, write split logs to motor_1 folder and
            # mirror them to motor_2 later.
            if self.config.separate_motors:
                split_log_dir = self.datasets_folder / f"{self.dataset_root_raw.name}_motor_1"
                split_log_dir.mkdir(parents=True, exist_ok=True)
            else:
                split_log_dir = self.dataset_root_raw
            print("\nGenerating final dataset splits from pooled raw data...")
            train_df, val_df, test_df = generate_raw_dataset_df(
                all_sessions_raw_df,
                str(split_log_dir),
                left_macs=self.config.left_macs,
                right_macs=self.config.right_macs,
                train_frac=self.config.train_frac,
                val_frac=self.config.val_frac,
                test_frac=self.config.test_frac,
                tolerance=self.config.tolerance,
                max_attempts=self.config.max_attempts,
                random_state=self.config.random_state,
                shuffle_within_split=self.config.shuffle_within_split
            )
            
            # Add motor_id column to all splits
            print("\nAdding motor_id column to raw datasets...")
            train_df = self.activity_manager.assign_motor_ids(train_df)
            val_df = self.activity_manager.assign_motor_ids(val_df)
            test_df = self.activity_manager.assign_motor_ids(test_df)
            print("motor_id column added (1=motor1, 2=motor2)")
            
            # Save final datasets
            if self.config.separate_motors:
                # Motor-specific label encoders are created inside this method;
                # no encoder is written to the common base folder.
                self._save_motor_separated_datasets(
                    train_df, val_df, test_df,
                    self.dataset_root_raw, "raw",
                    save_statistics=True
                )
            else:
                write_dataset_overwrite(train_df, val_df, test_df, self.dataset_root_raw)
                
                # Save statistics for raw datasets (using instance variables)
                self._save_dataset_statistics(
                    train_df, val_df, test_df,
                    self.dataset_root_raw,
                    self.global_activity_totals_df,
                    self.global_activity_segments_df,
                    self.global_phase_stats_df,
                    self.global_assembly_stats_df,
                    self.global_phase_assembly_breakdown_df,
                    self.global_acceleration_stats_df,
                    self.config.window_size_sec,
                    self.config.step_size_sec
                )
            
            print("\nRaw dataset generation complete.")
            
        except Exception as e:
            if not self.all_sessions_raw:
                print("No session windowed raw data collected. Exiting.")
            else:
                print(f"Failed during raw dataset generation: {e}")
                raise

    def _save_dataset_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        results_folder: Path,
        global_activity_totals_df: pd.DataFrame,
        global_activity_segments_df: pd.DataFrame,
        global_phase_stats_df: pd.DataFrame,
        global_assembly_stats_df: pd.DataFrame,
        global_phase_assembly_breakdown_df: pd.DataFrame,
        global_acceleration_stats_df: pd.DataFrame,
        window_size_sec: float,
        step_size_sec: float
    ):
        """
        Save class counts and statistics for a dataset.
        
        This method computes and saves:
        - class_counts.csv: Class distribution with expected windows
        - activity_totals.csv: Activity duration totals
        - activity_segments.csv: Activity time segments
        - phase_totals.csv: Phase statistics
        - assembly_totals.csv: Assembly statistics
        - phase_assembly_duration_breakdown.csv: per-phase and per-assembly duration mean/std
        - acceleration_totals.csv: aggregated accelerometer statistics across all sessions
        
        Parameters:
            train_df, val_df, test_df: Dataset splits
            results_folder: Folder to save statistics
            global_*: Global statistics dataframes
            window_size_sec: Window size for expected window calculation
            step_size_sec: Step size for expected window calculation
        """       
        # Ensure folder exists
        results_folder.mkdir(parents=True, exist_ok=True)

        # Group activity totals by final label to avoid duplicate rows when
        # activity grouping maps multiple original activities to one label.
        activity_totals_grouped = global_activity_totals_df
        if global_activity_totals_df is not None and not global_activity_totals_df.empty and "label" in global_activity_totals_df.columns:
            grouped = (
                global_activity_totals_df
                .groupby("label", as_index=False)
                .agg(
                    total_duration_sec=("total_duration_sec", "sum"),
                    total_repetitions=("total_repetitions", "max"),
                )
            )
            grouped["avg_duration_sec"] = grouped["total_duration_sec"] / grouped["total_repetitions"]

            # Recompute std over grouped labels using pooled sample variance.
            # This preserves a statistically meaningful std when multiple original
            # activity IDs are merged into one grouped label.
            required_std_cols = {"label", "total_repetitions", "avg_duration_sec", "std_duration_sec"}
            if required_std_cols.issubset(global_activity_totals_df.columns):
                std_rows = []
                for label, label_df in global_activity_totals_df.groupby("label"):
                    n = label_df["total_repetitions"].astype(float)
                    mu_i = label_df["avg_duration_sec"].astype(float)
                    s_i = label_df["std_duration_sec"].astype(float).fillna(0.0)

                    n_total = float(n.sum())
                    if n_total <= 1:
                        pooled_std = pd.NA
                    else:
                        mu_total = float((n * mu_i).sum() / n_total)
                        # Sample-variance pooling:
                        # sum((n_i-1)*s_i^2 + n_i*(mu_i-mu)^2) / (N-1)
                        numerator = float((((n - 1.0) * (s_i ** 2)) + (n * ((mu_i - mu_total) ** 2))).sum())
                        pooled_var = numerator / (n_total - 1.0)
                        pooled_std = pooled_var ** 0.5

                    std_rows.append({"label": label, "std_duration_sec": pooled_std})

                grouped = grouped.merge(pd.DataFrame(std_rows), on="label", how="left")
            activity_totals_grouped = grouped
        
        # Save global statistics
        save_dataframe(activity_totals_grouped, results_folder, "activity_totals")
        save_dataframe(global_activity_segments_df, results_folder, "activity_segments")
        save_dataframe(global_phase_stats_df, results_folder, "phase_totals")
        save_dataframe(global_assembly_stats_df, results_folder, "assembly_totals")
        save_dataframe(global_phase_assembly_breakdown_df, results_folder, "phase_assembly_duration_breakdown")
        save_dataframe(global_acceleration_stats_df, results_folder, "acceleration_totals")
        
        # Combine all splits to compute class counts
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Compute class counts
        label_column = "label_encoded"
        class_counts = combined_df[label_column].value_counts().sort_index()
        
        # Build encoded → string mapping
        mapping = (
            combined_df[[label_column, "label"]]
            .drop_duplicates()
            .set_index(label_column)["label"]
        )
        
        # Base counts df
        class_counts_df = (
            class_counts.to_frame(name="num_windows") 
            .join(mapping)
            .reset_index()
            .rename(columns={
                label_column: "encoded_label",
                "label": "label"
            })
        )
        
        # Merge with activity totals
        if activity_totals_grouped is not None and not activity_totals_grouped.empty:
            merge_cols = ["label", "total_duration_sec", "total_repetitions", "avg_duration_sec"]
            if "std_duration_sec" in activity_totals_grouped.columns:
                merge_cols.append("std_duration_sec")
            class_counts_df = class_counts_df.merge(
                activity_totals_grouped[merge_cols],
                left_on="label",
                right_on="label",
                how="left"
            )
        
        # Compute expected windows from activity segments
        if not global_activity_segments_df.empty:
            activity_segments = global_activity_segments_df.copy()
            
            # Ensure datetime columns
            if "start_time" in activity_segments.columns:
                activity_segments["start_time"] = pd.to_datetime(activity_segments["start_time"])
            if "end_time" in activity_segments.columns:
                activity_segments["end_time"] = pd.to_datetime(activity_segments["end_time"])
            
            # Ensure duration column exists
            if "duration" not in activity_segments.columns and "start_time" in activity_segments.columns and "end_time" in activity_segments.columns:
                activity_segments["duration"] = (activity_segments["end_time"] - activity_segments["start_time"]).dt.total_seconds()
            
            # Function to compute windows from duration
            def segment_windows_from_duration(duration, eps: float = 1e-9):
                if pd.isna(duration) or duration < window_size_sec:
                    return 0
                n = floor((duration - window_size_sec) / step_size_sec + eps) + 1
                return max(0, int(n))
            
            # Compute windows per segment
            activity_segments["expected_windows_seg"] = activity_segments["duration"].apply(segment_windows_from_duration)
            
            # Sum windows per activity label
            expected_windows_by_label = (
                activity_segments.groupby("label")["expected_windows_seg"]
                .sum()
                .to_dict()
            )
            
            class_counts_df["expected_windows"] = class_counts_df["label"].map(
                lambda lab: expected_windows_by_label.get(lab, 0)
            )
        
        # Reorder columns
        available_cols = [c for c in ["encoded_label", "label", "total_repetitions", "total_duration_sec",
                                       "avg_duration_sec", "expected_windows", "num_windows"] if c in class_counts_df.columns]
        class_counts_df = class_counts_df[available_cols]
        
        # Save class counts
        class_counts_df.to_csv(os.path.join(results_folder, "class_counts.csv"), index=False)
        print(f"Saved class counts to {results_folder / 'class_counts.csv'}")

    def _filter_statistics_for_motor(self, motor_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filter global statistics for a specific motor.
        
        Parameters:
            motor_type: "M1" or "M2"
        
        Returns:
            Tuple of (activity_totals, activity_segments, phase_stats, assembly_stats,
            phase_assembly_breakdown, acceleration_stats) filtered for motor
        """
        activities = self.activity_manager.motor1_activities if motor_type == "M1" else self.activity_manager.motor2_activities
        
        # Filter activity totals and segments by activities
        activity_totals = self.global_activity_totals_df[
            self.global_activity_totals_df["label"].isin(activities)
        ].copy() if self.global_activity_totals_df is not None and not self.global_activity_totals_df.empty else pd.DataFrame()
        
        activity_segments = self.global_activity_segments_df[
            self.global_activity_segments_df["label"].isin(activities)
        ].copy() if self.global_activity_segments_df is not None and not self.global_activity_segments_df.empty else pd.DataFrame()
        
        # Filter phase and assembly stats by motor type
        phase_stats = self.global_phase_stats_df[
            self.global_phase_stats_df["motor_type"] == motor_type
        ].copy() if self.global_phase_stats_df is not None and not self.global_phase_stats_df.empty else pd.DataFrame()
        
        assembly_stats = self.global_assembly_stats_df[
            self.global_assembly_stats_df["motor_type"] == motor_type
        ].copy() if self.global_assembly_stats_df is not None and not self.global_assembly_stats_df.empty else pd.DataFrame()

        phase_assembly_breakdown = self.global_phase_assembly_breakdown_df[
            self.global_phase_assembly_breakdown_df["motor_type"] == motor_type
        ].copy() if self.global_phase_assembly_breakdown_df is not None and not self.global_phase_assembly_breakdown_df.empty else pd.DataFrame()

        # Acceleration stats are motor-specific and axis-level
        acceleration_stats = self.global_acceleration_stats_df[
            self.global_acceleration_stats_df["motor_type"] == motor_type
        ].copy() if self.global_acceleration_stats_df is not None and not self.global_acceleration_stats_df.empty else pd.DataFrame()
        
        return activity_totals, activity_segments, phase_stats, assembly_stats, phase_assembly_breakdown, acceleration_stats

    def _save_motor_separated_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        base_folder: Path,
        dataset_type: str,
        save_statistics: bool = False
    ):
        """
        Save datasets separated by motor.
        
        Creates separate datasets for motor1 and motor2 activities
        using the activity manager's filtering logic. Re-encodes labels
        for each motor separately to ensure consecutive label encoding
        (0 to N_motor_classes-1) within each dataset, which is required
        by classifiers like XGBoost.

        
        Parameters:
            train_df, val_df, test_df: Train/validation/test dataframes.
                Contain label_encoded column from combined encoding.
            base_folder: Base output folder.
            dataset_type: "features" or "raw" for description.
            save_statistics: If True, also save class counts and statistics for each motor.
        """
        
        
        print(f"Creating separate {dataset_type} datasets for motor1 and motor2...")
        
        # Filter for motor1
        train_motor1 = self.activity_manager.filter_by_motor(train_df, "motor1")
        val_motor1 = self.activity_manager.filter_by_motor(val_df, "motor1")
        test_motor1 = self.activity_manager.filter_by_motor(test_df, "motor1")
        
        # Filter for motor2
        train_motor2 = self.activity_manager.filter_by_motor(train_df, "motor2")
        val_motor2 = self.activity_manager.filter_by_motor(val_df, "motor2")
        test_motor2 = self.activity_manager.filter_by_motor(test_df, "motor2")
        
        print(f"Motor1 {dataset_type} dataset sizes - train: {len(train_motor1)}, "
              f"val: {len(val_motor1)}, test: {len(test_motor1)}")
        print(f"Motor2 {dataset_type} dataset sizes - train: {len(train_motor2)}, "
              f"val: {len(val_motor2)}, test: {len(test_motor2)}")
        
        # Save motor1 datasets with re-encoded labels
        dataset_root_motor1 = self.datasets_folder / f"{base_folder.name}_motor_1"
        dataset_root_motor1.mkdir(parents=True, exist_ok=True)
        
        print(f"Re-encoding labels for motor1 {dataset_type}...")
        # Combine to get consistent encoding across train/val/test
        combined_motor1 = pd.concat([train_motor1, val_motor1, test_motor1], ignore_index=True)
        unique_labels_motor1 = sorted(combined_motor1["label"].dropna().unique())
        le_motor1 = LabelEncoder()
        le_motor1.fit(unique_labels_motor1)
        
        # Apply to each split
        train_motor1["label_encoded"] = le_motor1.transform(train_motor1["label"])
        val_motor1["label_encoded"] = le_motor1.transform(val_motor1["label"])
        test_motor1["label_encoded"] = le_motor1.transform(test_motor1["label"])
        
        # Save encoder
        le_motor1_path = dataset_root_motor1 / "label_encoder.pkl"
        with open(le_motor1_path, "wb") as f:
            pickle.dump(le_motor1, f)
        print(f"Saved motor1 LabelEncoder to {le_motor1_path.resolve()}")
        print(f"Motor1 activities: {list(unique_labels_motor1)} -> encoded as {list(range(len(unique_labels_motor1)))}")
        
        write_dataset_overwrite(train_motor1, val_motor1, test_motor1, dataset_root_motor1)
        print(f"Saved motor1 {dataset_type} datasets to {dataset_root_motor1.resolve()}")
        
        # Save statistics for motor1 if requested
        if save_statistics:
            activity_totals_m1, activity_segments_m1, phase_stats_m1, assembly_stats_m1, phase_assembly_breakdown_m1, acceleration_stats_m1 = self._filter_statistics_for_motor("M1")
            self._save_dataset_statistics(
                train_motor1, val_motor1, test_motor1,
                dataset_root_motor1,
                activity_totals_m1,
                activity_segments_m1,
                phase_stats_m1,
                assembly_stats_m1,
                phase_assembly_breakdown_m1,
                acceleration_stats_m1,
                self.config.window_size_sec,
                self.config.step_size_sec
            )
        
        # Save motor2 datasets with re-encoded labels
        dataset_root_motor2 = self.datasets_folder / f"{base_folder.name}_motor_2"
        dataset_root_motor2.mkdir(parents=True, exist_ok=True)
        
        print(f"Re-encoding labels for motor2 {dataset_type}...")
        # Combine to get consistent encoding across train/val/test
        combined_motor2 = pd.concat([train_motor2, val_motor2, test_motor2], ignore_index=True)
        unique_labels_motor2 = sorted(combined_motor2["label"].dropna().unique())
        le_motor2 = LabelEncoder()
        le_motor2.fit(unique_labels_motor2)
        
        # Apply to each split
        train_motor2["label_encoded"] = le_motor2.transform(train_motor2["label"])
        val_motor2["label_encoded"] = le_motor2.transform(val_motor2["label"])
        test_motor2["label_encoded"] = le_motor2.transform(test_motor2["label"])
        
        # Save encoder
        le_motor2_path = dataset_root_motor2 / "label_encoder.pkl"
        with open(le_motor2_path, "wb") as f:
            pickle.dump(le_motor2, f)
        print(f"Saved motor2 LabelEncoder to {le_motor2_path.resolve()}")
        print(f"Motor2 activities: {list(unique_labels_motor2)} -> encoded as {list(range(len(unique_labels_motor2)))}")
        
        write_dataset_overwrite(train_motor2, val_motor2, test_motor2, dataset_root_motor2)
        print(f"Saved motor2 {dataset_type} datasets to {dataset_root_motor2.resolve()}")

        # Mirror split-generation log so both motor folders contain the logger output.
        motor1_log = dataset_root_motor1 / "dataset_generation.log"
        motor2_log = dataset_root_motor2 / "dataset_generation.log"
        if motor1_log.exists() and not motor2_log.exists():
            shutil.copy2(motor1_log, motor2_log)
        
        # Save statistics for motor2 if requested
        if save_statistics:
            activity_totals_m2, activity_segments_m2, phase_stats_m2, assembly_stats_m2, phase_assembly_breakdown_m2, acceleration_stats_m2 = self._filter_statistics_for_motor("M2")
            self._save_dataset_statistics(
                train_motor2, val_motor2, test_motor2,
                dataset_root_motor2,
                activity_totals_m2,
                activity_segments_m2,
                phase_stats_m2,
                assembly_stats_m2,
                phase_assembly_breakdown_m2,
                acceleration_stats_m2,
                self.config.window_size_sec,
                self.config.step_size_sec
            )
