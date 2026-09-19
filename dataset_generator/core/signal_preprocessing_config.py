"""
Centralized configuration for signal preprocessing pipeline.
All values are defined once as module-level constants, and the PreprocessingConfig
dataclass defaults reference these constants.

Configuration Structure:
    1. Define all constants (LEFT_MACS, WINDOW_SIZE_SEC, etc.)
    2. PreprocessingConfig dataclass uses field(default=CONSTANT) for all defaults
    3. All code imports from this module for configuration

Usage:
    # Option 1: Use defaults (references module constants)
    config = PreprocessingConfig()
    
    # Option 2: Override specific values
    config = PreprocessingConfig(window_size_sec=2.0, separate_motors=True)
    
    # Option 3: Direct access to constants
    from config import LEFT_MACS, WINDOW_SIZE_SEC
    print(LEFT_MACS)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# Device Configuration
# ============================================================
LEFT_MACS = ("DA",)
"""MAC addresses for left watch(es)."""

RIGHT_MACS = ("D5",)
"""MAC addresses for right watch(es)."""

# ============================================================
# Preprocessing Parameters
# ============================================================
CUTOFF = 0.3
"""Low-pass filter cutoff frequency (normalized, 0-1)."""

# ============================================================
# Processing Control
# ============================================================
REPROCESS_ALL_SESSIONS = True
"""If True, force reprocessing of all sessions, ignoring cached outputs."""

# ============================================================
# Windowing Configuration
# ============================================================
FULL_ACTIVITY_WINDOWING = False
"""If True, each window spans a full activity. If False, use sliding windows."""

WINDOW_SIZE_SEC = 5.0
"""Sliding window size in seconds (used only if FULL_ACTIVITY_WINDOWING=False)."""

OVERLAP = 0.5 # Determines overlap
"""Determines overlap between cosecutive sliding windows"""

STEP_SIZE_SEC = WINDOW_SIZE_SEC * (1 - OVERLAP) # Determines step size by overlap
"""Sliding window step size in seconds/overlap (used only if FULL_ACTIVITY_WINDOWING=False)."""

# ============================================================
# Soft Label Configuration
# ============================================================
USE_SOFT_LABELS = False
"""If True, compute per-window soft label distributions."""

# ============================================================
# Learning Effect Parameters
# ============================================================
FIRST_ASSEMBLIES_TO_REMOVE = 2
"""Number of assemblies to remove from first two phases (learning effect)."""

NON_LEARNING_SESSIONS = ["14", "16", "17", "18", "19", "21", "22", "23", "24", "25", "26"]
"""Session codes that are non-learning sessions (no assemblies removed)."""

# ============================================================
# Motor Separation Configuration
# ============================================================
SEPARATE_MOTORS = True
"""If True, create separate datasets for motor1 and motor2."""

# ============================================================
# Activity Grouping Configuration
# ============================================================
ACTIVITY_GROUPING = False
"""If True, group activities according to ACTIVITY_GROUPS mapping."""

ACTIVITY_GROUPS: Dict[str, List[str]] = {
    "Motor housing preparation": ["Pick up motor housing","Insert rotor"], 
    "Wiring": ["Connect wires","Attach power source"],
    "Packaging": ["Place in package","Close package"], 
    "Attach fan": ["Pick up rotor", "Attach fan"],
    "Attach bearings": ["Attach bearings", "Set rotor aside"], 
    "Lower housing half preparation": ["Retrieve lower housing half","Insert stator"],
}
"""
Dictionary mapping group names to activity lists. Empty = no grouping.

Format: {group_name: [list_of_original_activities]}

Example:
    {
        "Assembly": [
            "Pick up motor housing",
            "Insert rotor",
            "Attach brushes",
            "Connect wires",
            "Attach power source",
            "Attach end cap",
            "Tighten cover"
        ],
        "Packaging": [
            "Place in package",
            "Close package"
        ]
    }

When enabled, all activities in a group are remapped to the group name
during preprocessing, reducing the number of classes.
"""

GROUPING_NAME = "1"
"""Name for the grouping scheme (appended to dataset folder name as AG(name) if activity_grouping=True)."""

# ============================================================
# Wavelet Transform
# ============================================================
ADD_WAVELET_FEATURES = False
"""If True, compute and add wavelet features to raw sensor data during preprocessing."""

WAVELET_FAMILY = "morl"
"""Wavelet family to use for CWT (Continuous Wavelet Transform).
Available options: 'morl' (Morlet), 'mexh' (Mexican hat), 'cmor', 'gaus1'-'gaus8', 'cgau1'-'cgau8'.
NOTE: 'db4', 'sym5', etc. are discrete wavelets - use only continuous wavelets for CWT!"""

WAVELET_SCALES = [1, 2, 4, 8]
"""CWT scales for wavelet decomposition. Lower scale = higher frequency localization."""
"""frequency (Hz) = center_frequency / (scale × sampling_period)"""
""" => sampling_period = (center_frequency / frequency (Hz)) × scale"""

# ============================================================
# Advanced: Kernel Discriminant Analysis
# ============================================================
USE_KDA = False
"""If True, apply KDA transformation before feature selection."""

KDA_N_COMPONENTS: Optional[int] = None
"""Number of KDA components (None = n_classes - 1)."""

KDA_GAMMA: Optional[float] = None
"""RBF kernel gamma parameter (None = auto-tune as 1/n_features)."""

# ============================================================
# Feature Selection Configuration
# ============================================================
CUMULATIVE_THRESHOLD = 0.90
"""Cumulative importance threshold for feature selection (0-1)."""

# ============================================================
# Dataset Split Configuration
# ============================================================
TRAIN_FRAC = 0.7
"""Fraction of data for training split."""

VAL_FRAC = 0.15
"""Fraction of data for validation split."""

TEST_FRAC = 0.15
"""Fraction of data for test split."""

# ============================================================
# Dataset Split Robustness
# ============================================================
TOLERANCE = 0.07
"""Max acceptable class distribution deviation for stratified split."""

MAX_ATTEMPTS = 1000
"""Maximum retry attempts for finding suitable stratified split."""

RANDOM_STATE = 42
"""Seed for deterministic random operations."""

SHUFFLE_WITHIN_SPLIT = True
"""Whether to shuffle examples within each split."""


# ============================================================
# PreprocessingConfig Class
# ============================================================

@dataclass
class PreprocessingConfig:
    """
    Centralized configuration for the signal preprocessing pipeline.
    
    This dataclass provides an object-oriented interface to all preprocessing
    configuration. All default values reference the module-level constants defined
    above, ensuring a single source of truth.
    
    To change defaults, edit the module-level constants (LEFT_MACS, WINDOW_SIZE_SEC, etc.)
    at the top of this file. The dataclass will automatically use the updated values.
    
    Attributes:
        left_macs (tuple[str, ...]): MAC addresses for left watch(es).
            Default from: LEFT_MACS
        right_macs (tuple[str, ...]): MAC addresses for right watch(es).
            Default from: RIGHT_MACS
        cutoff (float): Low-pass filter cutoff frequency (normalized, 0-1).
            Default from: CUTOFF
        full_activity_windowing (bool): Windows span complete activities vs sliding.
            Default from: FULL_ACTIVITY_WINDOWING
        window_size_sec (float): Sliding window size in seconds.
            Only used if full_activity_windowing=False.
            Default from: WINDOW_SIZE_SEC
        step_size_sec (float): Sliding window step size in seconds (determines overlap).
            Only used if full_activity_windowing=False.
            Default from: STEP_SIZE_SEC
        first_assemblies_to_remove (int): Assemblies to exclude from first phases.
            Default from: FIRST_ASSEMBLIES_TO_REMOVE
        non_learning_sessions (list[str]): Session codes without learning effect.
            Default from: NON_LEARNING_SESSIONS
        separate_motors (bool): Create separate datasets for each motor.
            Default from: SEPARATE_MOTORS
        activity_grouping (bool): Enable activity grouping/remapping.
            Default from: ACTIVITY_GROUPING
        activity_groups (dict): Group name -> activity list mapping.
            Default from: ACTIVITY_GROUPS
        use_kda (bool): Apply KDA transformation before feature selection.
            Default from: USE_KDA
        kda_n_components (int | None): Number of KDA components.
            Default from: KDA_N_COMPONENTS
        kda_gamma (float | None): RBF kernel gamma parameter.
            Default from: KDA_GAMMA
        train_frac (float): Fraction for training set (0-1).
            Default from: TRAIN_FRAC
        val_frac (float): Fraction for validation set (0-1).
            Default from: VAL_FRAC
        test_frac (float): Fraction for test set (0-1).
            Default from: TEST_FRAC
        tolerance (float): Max deviation in class distribution for splits (0-1).
            Default from: TOLERANCE
        max_attempts (int): Max retries finding stratified split.
            Default from: MAX_ATTEMPTS
        random_state (int): Random seed for reproducibility.
            Default from: RANDOM_STATE
        shuffle_within_split (bool): Shuffle examples within each split.
            Default from: SHUFFLE_WITHIN_SPLIT
    
    Example:
        Create config with all defaults::
        
            config = PreprocessingConfig()
            print(config.window_size_sec)  # 1.0 (from WINDOW_SIZE_SEC)
        
        Override specific values::
        
            config = PreprocessingConfig(
                window_size_sec=2.0,
                separate_motors=True
            )
        
        Change global defaults by editing constants at top of config.py.
        All future instances will use the new values::
        
            # In config.py, change:
            WINDOW_SIZE_SEC = 2.0  # Was 1.0
            
            # Now:
            config = PreprocessingConfig()
            print(config.window_size_sec)  # 2.0 (updated!)
    """
    
    # =====================================================================
    # Device Configuration
    # =====================================================================
    left_macs: Tuple[str, ...] = field(default_factory=lambda: LEFT_MACS)
    """MAC addresses for left watch(es). Default from LEFT_MACS constant."""
    
    right_macs: Tuple[str, ...] = field(default_factory=lambda: RIGHT_MACS)
    """MAC addresses for right watch(es). Default from RIGHT_MACS constant."""
    
    # =====================================================================
    # Preprocessing Parameters
    # =====================================================================
    cutoff: float = field(default=CUTOFF)
    """Low-pass filter cutoff frequency. Default from CUTOFF constant."""
    
    # =====================================================================
    # Windowing Configuration
    # =====================================================================
    full_activity_windowing: bool = field(default=FULL_ACTIVITY_WINDOWING)
    """If True, windows span complete activities. Default from FULL_ACTIVITY_WINDOWING."""
    
    window_size_sec: float = field(default=WINDOW_SIZE_SEC)
    """Sliding window size in seconds. Default from WINDOW_SIZE_SEC constant."""
    
    step_size_sec: float = field(default=STEP_SIZE_SEC)
    """Sliding window step size (overlap). Default from STEP_SIZE_SEC constant."""

    # =====================================================================
    # Soft Labels
    # =====================================================================
    use_soft_labels: bool = field(default=USE_SOFT_LABELS)
    """Compute and propagate per-window label distributions (soft labels)."""
    
    # =====================================================================
    # Learning Effect Removal
    # =====================================================================
    first_assemblies_to_remove: int = field(default=FIRST_ASSEMBLIES_TO_REMOVE)
    """Assemblies to remove from first phases. Default from FIRST_ASSEMBLIES_TO_REMOVE."""
    
    non_learning_sessions: List[str] = field(default_factory=lambda: NON_LEARNING_SESSIONS)
    """Sessions without learning effect. Default from NON_LEARNING_SESSIONS constant."""
    
    # =====================================================================
    # Motor Separation
    # =====================================================================
    separate_motors: bool = field(default=SEPARATE_MOTORS)
    """Create separate datasets per motor. Default from SEPARATE_MOTORS constant."""
    
    # =====================================================================
    # Activity Grouping
    # =====================================================================
    activity_grouping: bool = field(default=ACTIVITY_GROUPING)
    """Enable activity grouping. Default from ACTIVITY_GROUPING constant."""
    
    activity_groups: Dict[str, List[str]] = field(default_factory=lambda: ACTIVITY_GROUPS)
    """Activity group mapping. Default from ACTIVITY_GROUPS constant."""
    
    grouping_name: str = field(default=GROUPING_NAME)
    """Name for grouping scheme (appended as AG(name) to dataset folder if activity_grouping=True). Default from GROUPING_NAME constant."""
    
    # =====================================================================
    # Wavelet Transform
    # =====================================================================
    add_wavelet_features: bool = field(default=ADD_WAVELET_FEATURES)
    """Add wavelet features to raw sensor data. Default from ADD_WAVELET_FEATURES constant."""
    
    wavelet_family: str = field(default=WAVELET_FAMILY)
    """Wavelet family for CWT ('db4', 'sym5', 'coif3', etc.). Default from WAVELET_FAMILY constant."""
    
    wavelet_scales: List[int] = field(default_factory=lambda: WAVELET_SCALES)
    """CWT scales for wavelets. Default from WAVELET_SCALES constant."""
    
    # =====================================================================
    # Advanced: Kernel Discriminant Analysis
    # =====================================================================
    use_kda: bool = field(default=USE_KDA)
    """Apply KDA transformation. Default from USE_KDA constant."""
    
    kda_n_components: Optional[int] = field(default=KDA_N_COMPONENTS)
    """KDA components. Default from KDA_N_COMPONENTS constant."""
    
    kda_gamma: Optional[float] = field(default=KDA_GAMMA)
    """KDA gamma parameter. Default from KDA_GAMMA constant."""
    
    # =====================================================================
    # Feature Selection Configuration
    # =====================================================================
    cumulative_threshold: float = field(default=CUMULATIVE_THRESHOLD)
    """Cumulative importance threshold for feature selection. Default from CUMULATIVE_THRESHOLD constant."""
    
    # =====================================================================
    # Dataset Split Configuration
    # =====================================================================
    train_frac: float = field(default=TRAIN_FRAC)
    """Training set fraction. Default from TRAIN_FRAC constant."""
    
    val_frac: float = field(default=VAL_FRAC)
    """Validation set fraction. Default from VAL_FRAC constant."""
    
    test_frac: float = field(default=TEST_FRAC)
    """Test set fraction. Default from TEST_FRAC constant."""
    
    # =====================================================================
    # Dataset Split Robustness
    # =====================================================================
    tolerance: float = field(default=TOLERANCE)
    """Max class distribution deviation. Default from TOLERANCE constant."""
    
    max_attempts: int = field(default=MAX_ATTEMPTS)
    """Max split attempts. Default from MAX_ATTEMPTS constant."""
    
    random_state: int = field(default=RANDOM_STATE)
    """Random seed. Default from RANDOM_STATE constant."""
    
    shuffle_within_split: bool = field(default=SHUFFLE_WITHIN_SPLIT)
    """Shuffle within splits. Default from SHUFFLE_WITHIN_SPLIT constant."""

    # =====================================================================
    # Processing Control
    # =====================================================================
    reprocess_all_sessions: bool = field(default=REPROCESS_ALL_SESSIONS)
    """Force reprocessing for all sessions, ignoring cached outputs."""
    
    def __post_init__(self):
        """
        Validate configuration after initialization.
        
        Performs sanity checks on configuration parameters to catch
        invalid combinations early. Raises ValueError if validation fails.
        
        Validation checks:
            - Split fractions sum to approximately 1.0
            - Window and step sizes are positive when using sliding windows
            - Step size does not exceed window size
        
        Raises:
            ValueError: If any configuration constraint is violated.
        
        Example:
            >>> config = PreprocessingConfig(train_frac=0.5, val_frac=0.5)
            Traceback (most recent call last):
            ...
            ValueError: Split fractions must sum to 1.0...
        """
        # Validate split fractions sum to 1.0
        total = self.train_frac + self.val_frac + self.test_frac
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total}. "
                f"Current: train={self.train_frac}, val={self.val_frac}, test={self.test_frac}"
            )
        
        # Validate windowing parameters
        if not self.full_activity_windowing:
            if self.window_size_sec <= 0:
                raise ValueError(
                    f"window_size_sec must be positive when using sliding windows, "
                    f"got {self.window_size_sec}"
                )
            if self.step_size_sec <= 0:
                raise ValueError(
                    f"step_size_sec must be positive when using sliding windows, "
                    f"got {self.step_size_sec}"
                )
            if self.step_size_sec > self.window_size_sec:
                raise ValueError(
                    f"step_size_sec ({self.step_size_sec}s) cannot exceed "
                    f"window_size_sec ({self.window_size_sec}s)"
                )
    
    def get_dataset_folder_names(self) -> Tuple[str, str]:
        """
        Generate dynamic dataset folder names based on windowing configuration.
        
        Creates descriptive folder names that encode the windowing strategy
        and parameters. This allows different windowing configurations to
        coexist without overwriting each other.
        
        For sliding windows, naming convention:
            Dataset_features_wsi<size>_ssi<step>
            Dataset_raw_wsi<size>_ssi<step>
        
        For full activity windows, naming convention:
            Dataset_features_fullactivities
            Dataset_raw_fullactivities
        
        If separate_motors is enabled, the motor type (motor_1 or motor_2)
        will be appended to the folder name by the DatasetBuilder.
        
        Returns:
            tuple[str, str]: (features_folder_name, raw_folder_name)
        
        Example:
            >>> config = PreprocessingConfig(window_size_sec=1.0, step_size_sec=0.25)
            >>> feat_name, raw_name = config.get_dataset_folder_names()
            >>> print(feat_name)
            Dataset_features_wsi1_ssi0_25
            
            >>> config = PreprocessingConfig(full_activity_windowing=True)
            >>> feat_name, raw_name = config.get_dataset_folder_names()
            >>> print(feat_name)
            Dataset_features_fullactivities
        """
        if self.full_activity_windowing:
            suffix = "fullactivities"
        else:
            # Convert to int if whole number for cleaner names
            wsi = int(self.window_size_sec) if self.window_size_sec == int(self.window_size_sec) else self.window_size_sec
            ssi = int(self.step_size_sec) if self.step_size_sec == int(self.step_size_sec) else self.step_size_sec
            # Replace decimal point with underscore for filename compatibility
            wsi = str(wsi).replace('.', '_')
            ssi = str(ssi).replace('.', '_')
            suffix = f"wsi{wsi}_ssi{ssi}"
        
        # Append wavelet suffix if enabled (for raw datasets only)
        wavelet_suffix = "_wavelets" if self.add_wavelet_features else ""
        
        # Append activity grouping suffix if enabled
        if self.activity_grouping and self.grouping_name:
            suffix = f"{suffix}_AG{self.grouping_name}"
        
        return f"Dataset_features_{suffix}", f"Dataset_raw{wavelet_suffix}_{suffix}"
    
    def is_non_learning_session(self, session_code: str) -> bool:
        """
        Check if a session is a non-learning session.
        
        Non-learning sessions are identified by session code and do not have
        the first N assemblies removed (no learning effect removal).
        
        Parameters:
            session_code (str): Session identifier to check.
        
        Returns:
            bool: True if the session code is in non_learning_sessions,
                False otherwise.
        
        Example:
            >>> config = PreprocessingConfig(non_learning_sessions=["lab1", "lab2"])
            >>> config.is_non_learning_session("lab1")
            True
            >>> config.is_non_learning_session("1")
            False
        """
        return session_code in self.non_learning_sessions
