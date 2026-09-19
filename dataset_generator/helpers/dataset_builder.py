from __future__ import annotations
from typing import Sequence, Dict, Optional, Tuple
import pandas as pd

from .window_fusion import fuse_feature_windows, fuse_raw_windows
from .splits import grouped_stratified_split_groups
from .signal_preprocessing_utils import setup_logger
from .constants import get_feature_columns


# -------------------------
# Helper: Convert groups -> DataFrame
# -------------------------
def groups_to_df(groups: Dict[str, Dict], data_key: str) -> pd.DataFrame:
    """
    Convert a dictionary of fused window groups into a tabular DataFrame.

    Parameters
    ----------
    groups : dict
        Mapping from window-id -> dict containing fused data and metadata.
        Expected keys in each dict:
            - data_key: dict of <prefix><feature_name> -> value (e.g., "L_mean")
            - label
            - label_encoded
            - window_id
            - session
    data_key : str
        Which key to use for the main data ("features" or "raw_data").

    Returns
    -------
    pd.DataFrame
        Each row corresponds to one window, columns are the prefixed features or raw_data,
        plus label, label_encoded, window_id, session.
    """
    rows = []
    for g in groups.values():
        row = dict(g[data_key])  # copy the feature/raw_data dict
        
        # Ensure lists stay as lists (not converted to Series)
        for col, val in row.items():
            if isinstance(val, list):
                row[col] = val  # Keep as list
        
        row.update({
            "label": g["label"],
            "label_encoded": g["label_encoded"],
            "window_id": g.get("window_id"),
            "session_code": g.get("session")
        })
        
        # Preserve label_soft if present
        if "label_soft" in g and g["label_soft"] is not None:
            row["label_soft"] = g["label_soft"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Force object dtype for list columns and ensure they don't get converted to Series
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert any Series objects back to lists
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, pd.Series) else x)
    
    return df


# -------------------------
# Feature dataset generator
# -------------------------
def generate_feature_dataset_df(
    df: pd.DataFrame,
    log_dir: str,
    left_macs: Sequence[str] = ("DA",),
    right_macs: Sequence[str] = ("D5",),
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    tolerance: float = 0.07,
    max_attempts: int = 1000,
    random_state: int = 42,
    shuffle_within_split: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate train/val/test **feature datasets** directly as pandas DataFrames.

    Each DataFrame contains:
        - Prefixed features: "L_*", "R_*"
        - label, label_encoded, window_id, session_code

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe with per-window per-device feature rows.
    log_dir : str
        Folder to store logs.
    left_macs, right_macs : sequence of str
        MAC addresses for left/right sensors.
    train_frac, val_frac, test_frac : float
        Fractions for splits.
    tolerance : float
        Maximum allowed deviation of label proportions across splits.
    max_attempts : int
        Number of random shuffles to attempt for stratified split.
    random_state : int
        Seed for reproducibility.
    shuffle_within_split : bool
        If True, shuffle the rows within each split.
    verbose : bool
        Whether to print logs.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        DataFrames for train, validation, and test sets.
    """
    logger = setup_logger(log_dir)

    # Fuse windows into groups
    groups, skipped = fuse_feature_windows(df, left_macs, right_macs, logger=logger, verbose=verbose)
    if skipped:
        logger.warning(f"Skipped {len(skipped)} malformed windows.")

    # Stratified split by session/phase/assembly combos
    train_g, val_g, test_g = grouped_stratified_split_groups(
        groups, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        tolerance=tolerance, max_attempts=max_attempts, random_state=random_state,
        logger=logger, verbose=verbose
    )

    # Convert group dicts directly to DataFrames
    train_df = groups_to_df(train_g, data_key="features")
    val_df   = groups_to_df(val_g, data_key="features")
    test_df  = groups_to_df(test_g, data_key="features")
    
    # Filter out rows with NaN values in feature columns
    feature_cols = get_feature_columns(train_df)
    
    train_before = len(train_df)
    train_df = train_df.dropna(subset=feature_cols, how="any").reset_index(drop=True)
    if len(train_df) < train_before:
        logger.warning(f"Removed {train_before - len(train_df)} rows with NaN features from train set")
    
    val_before = len(val_df)
    val_df = val_df.dropna(subset=feature_cols, how="any").reset_index(drop=True)
    if len(val_df) < val_before:
        logger.warning(f"Removed {val_before - len(val_df)} rows with NaN features from val set")
    
    test_before = len(test_df)
    test_df = test_df.dropna(subset=feature_cols, how="any").reset_index(drop=True)
    if len(test_df) < test_before:
        logger.warning(f"Removed {test_before - len(test_df)} rows with NaN features from test set")

    # Optional shuffle
    if shuffle_within_split:
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df   = val_df.sample(frac=1, random_state=random_state + 1).reset_index(drop=True)
        test_df  = test_df.sample(frac=1, random_state=random_state + 2).reset_index(drop=True)

    return train_df, val_df, test_df


# -------------------------
# Raw dataset generator
# -------------------------
def generate_raw_dataset_df(
    df: pd.DataFrame,
    log_dir: str,
    left_macs: Sequence[str] = ("DA",),
    right_macs: Sequence[str] = ("D5",),
    raw_cols: Optional[Sequence[str]] = ("x_motion_smooth", "y_motion_smooth", "z_motion_smooth"),
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    tolerance: float = 0.07,
    max_attempts: int = 1000,
    random_state: int = 42,
    shuffle_within_split: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High-level generator for raw-window datasets as pandas DataFrames.

    Each returned DataFrame contains one row per window, with columns:
      - Prefixed raw signal columns like L_x_motion_smooth, L_y_motion_smooth, R_x_motion_smooth, etc.
        Each cell contains a **NumPy array of raw samples** for that axis and device over the window.
      - 'label_encoded', 'label', 'window_id', 'phase', 'assembly', 'session_code'

    Stratified splitting is performed by unique (session, phase, assembly) combos.
    Only windows containing **all left and all right MACs** are kept.

    Args:
        df: Input dataframe containing raw sensor data per device and window.
        log_dir: Path to log folder.
        left_macs: MAC addresses defining the left sensors.
        right_macs: MAC addresses defining the right sensors.
        train_frac, val_frac, test_frac: Fractions for split.
        tolerance: Maximum allowed deviation of label proportions across splits.
        max_attempts: Maximum attempts to find a split satisfying tolerance.
        random_state: Seed for reproducibility.
        verbose: If True, logs progress and warnings.

    Returns:
        Tuple of (train_df, val_df, test_df) pandas DataFrames.
        Each row contains fused raw arrays for the window, plus metadata.
    """
    logger = setup_logger(log_dir)

    # Auto-detect wavelet columns if present in the dataframe
    wavelet_cols = [col for col in df.columns if col.startswith(('x_wavelet', 'y_wavelet', 'z_wavelet'))]
    
    # Combine raw cols with wavelet cols for fusing
    all_cols_to_fuse = list(raw_cols) + wavelet_cols if raw_cols else wavelet_cols
    
    # Fuse raw windows (with wavelet features if present)
    groups, skipped = fuse_raw_windows(df, left_macs, right_macs, raw_cols=all_cols_to_fuse, logger=logger, verbose=verbose)
    if skipped:
        logger.warning(f"Skipped {len(skipped)} malformed raw windows.")

    # Stratified split by session/phase/assembly combos
    train_g, val_g, test_g = grouped_stratified_split_groups(
        groups, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        tolerance=tolerance, max_attempts=max_attempts, random_state=random_state,
        logger=logger, verbose=verbose
    )

    # Convert group dicts directly to DataFrames
    train_df = groups_to_df(train_g, data_key="raw_data")
    val_df   = groups_to_df(val_g, data_key="raw_data")
    test_df  = groups_to_df(test_g, data_key="raw_data")
    
    # Filter out rows with NaN values in raw data columns
    raw_cols = get_feature_columns(train_df)
    
    train_before = len(train_df)
    train_df = train_df.dropna(subset=raw_cols, how="any").reset_index(drop=True)
    if len(train_df) < train_before:
        logger.warning(f"Removed {train_before - len(train_df)} rows with NaN raw data from train set")
    
    val_before = len(val_df)
    val_df = val_df.dropna(subset=raw_cols, how="any").reset_index(drop=True)
    if len(val_df) < val_before:
        logger.warning(f"Removed {val_before - len(val_df)} rows with NaN raw data from val set")
    
    test_before = len(test_df)
    test_df = test_df.dropna(subset=raw_cols, how="any").reset_index(drop=True)
    if len(test_df) < test_before:
        logger.warning(f"Removed {test_before - len(test_df)} rows with NaN raw data from test set")

    # Optional shuffle
    if shuffle_within_split:
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df   = val_df.sample(frac=1, random_state=random_state + 1).reset_index(drop=True)
        test_df  = test_df.sample(frac=1, random_state=random_state + 2).reset_index(drop=True)

    return train_df, val_df, test_df
