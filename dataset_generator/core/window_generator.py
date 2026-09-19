from __future__ import annotations
from typing import List, Tuple
import pandas as pd

from .signal_preprocessing_config import PreprocessingConfig
from ..helpers import generate_windows, generate_activity_windows


class WindowGenerator:
    """
    Handles window extraction with support for multiple windowing strategies.
    
    This module implements the window extraction logic for the preprocessing
    pipeline, supporting multiple windowing strategies including sliding windows
    and full activity windows.
    
    Supported Strategies:
        1. Sliding Windows: Generates fixed-size overlapping windows.
           Configuration: full_activity_windowing=False
           Parameters: window_size_sec, step_size_sec
        
        2. Full Activity Windows: Generates one window per complete activity.
           Configuration: full_activity_windowing=True
           Parameters: None (uses complete activity boundaries)
    
    Attributes:
        config (PreprocessingConfig): Configuration instance specifying
            which windowing strategy to use and its parameters.
    
    Example:
        Initialize for sliding windows::
        
            config = PreprocessingConfig(
                full_activity_windowing=False,
                window_size_sec=1.0,
                step_size_sec=0.25
            )
            gen = WindowGenerator(config)
        
        Generate windows from segments::
        
            windows, count = gen.generate_windows(
                assembly_segments_df,
                phase_segments_df
            )
            print(f"Generated {count} windows")
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the WindowGenerator.
        
        Parameters:
            config (PreprocessingConfig): Configuration instance specifying
                windowing strategy and parameters.
        
        Example:
            >>> config = PreprocessingConfig()
            >>> gen = WindowGenerator(config)
        """
        self.config = config
    
    def generate_windows(
        self,
        activity_segments: pd.DataFrame,
        assembly_segments: pd.DataFrame,
        phase_segments: pd.DataFrame
    ) -> Tuple[List[dict], int]:
        """
        Generate window definitions based on configuration.
        
        Delegates to the appropriate windowing strategy based on the
        configuration's full_activity_windowing setting.
        
        Parameters:
            assembly_segments (pd.DataFrame): DataFrame with columns:
                - start_time: Timestamp when assembly started
                - end_time: Timestamp when assembly ended
                - phase_id: Which phase this assembly belongs to
                - assembly_id: Unique identifier for assembly
            
            phase_segments (pd.DataFrame): DataFrame with columns:
                - phase_id: Unique identifier for phase
                - start_time: Timestamp when phase started
                - end_time: Timestamp when phase ended
        
        Returns:
            tuple[list[dict], int]: A tuple containing:
                - List of window definitions (dicts with window_start, window_end,
                  phase_id, assembly_id)
                - Total number of windows generated
        
        Example:
            >>> windows, count = gen.generate_windows(
            ...     assembly_segments, phase_segments
            ... )
            >>> print(f"Generated {len(windows)} windows")
            >>> print(windows[0])
            {'window_start': Timestamp(...), 'window_end': Timestamp(...),
             'phase_id': 1, 'assembly_id': 1}
        """
        if self.config.full_activity_windowing:
            return self._generate_activity_windows(activity_segments, phase_segments)
        else:
            return self._generate_sliding_windows(assembly_segments, phase_segments)
    
    def _generate_sliding_windows(
        self,
        assembly_segments: pd.DataFrame,
        phase_segments: pd.DataFrame
    ) -> Tuple[List[dict], int]:
        """
        Generate fixed-size overlapping sliding windows.
        
        Creates sliding windows across assembly periods and baseline phases.
        Windows have configurable size and step (overlap) determined by
        window_size_sec and step_size_sec in the configuration.
        
        Windows are only generated:
        - Within assembly time periods
        - At the start of each phase (baseline, before first assembly)
        
        Parameters:
            assembly_segments (pd.DataFrame): Assembly time segments.
            phase_segments (pd.DataFrame): Phase time segments.
        
        Returns:
            tuple[list[dict], int]: (window_definitions, num_windows_generated)
        
        Notes:
            This delegates to window_extraction.generate_windows() which
            implements the actual sliding window algorithm.
        """
        
        return generate_windows(
            assembly_segments=assembly_segments,
            phase_segments=phase_segments,
            window_size_sec=self.config.window_size_sec,
            step_size_sec=self.config.step_size_sec,
            left_macs=self.config.left_macs,
            right_macs=self.config.right_macs
        )
    
    def _generate_activity_windows(
        self,
        activity_segments: pd.DataFrame,
        phase_segments: pd.DataFrame
    ) -> Tuple[List[dict], int]:
        """
        Generate windows spanning complete activities without overlap.
        
        Creates one window per activity, using the activity's start and end
        times as window boundaries. This approach avoids artificial window
        boundaries and preserves temporal coherence within activities.
        
        Parameters:
            assembly_segments (pd.DataFrame): Assembly time segments.
            phase_segments (pd.DataFrame): Phase time segments.
        
        Returns:
            tuple[list[dict], int]: (window_definitions, num_windows_generated)
        
        Notes:
            This delegates to window_extraction.generate_activity_windows()
            which implements the actual activity window algorithm.
        """
        
        return generate_activity_windows(
            activity_segments=activity_segments,
            phase_segments=phase_segments,
            left_macs=self.config.left_macs,
            right_macs=self.config.right_macs
        )
