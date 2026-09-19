from __future__ import annotations
from typing import Sequence, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from .constants import get_feature_columns
from .signal_preprocessing_utils import get_logger

# -------------------------
# Fuse windows (feature data)
# -------------------------
def fuse_feature_windows(
    df: pd.DataFrame,
    left_macs: Sequence[str],
    right_macs: Sequence[str],
    logger: Optional[logging.Logger] = None,
    verbose: bool = False
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Fuse per-window *feature* data from left and right sensor devices into a
    single flat, prefix-labeled feature vector.

    For each window (identified by phase_id, assembly_id, window_start, window_end),
    this function extracts the rows belonging to the left and right MAC lists.
    A window is kept only if **all** left MACs and **all** right MACs appear
    exactly once.

    Feature columns are discovered from the dataframe using `get_feature_columns(df)`
    and are then copied into a *single fused feature dict* with automatic prefixes:

        • If only one left MAC:   "L_"
        • If multiple left MACs:  "L1_", "L2_", ...
        • If only one right MAC:  "R_"
        • If multiple right MACs: "R1_", "R2_", ...

    Example (two left MACs, one right MAC):
        L1_rms, L1_mean, L1_std,
        L2_rms, L2_mean, L2_std,
        R_rms,  R_mean,  R_std

    Returned structure per window:
        groups[gid] = {
            "features": { "<prefix><feature_name>": value, ... },
            "feature_columns": ordered list of the fused feature names,
            "label":         class label for the window,
            "label_encoded": encoded label if available,
            "window_id":     optional window identifier,
            "phase":         phase_id,
            "assembly":      assembly_id,
            "session":       optional session_code
        }

    Args:
        df: Input dataframe containing one row per device per window.
        left_macs:  MAC addresses that define the left-side devices.
        right_macs: MAC addresses that define the right-side devices.
        logger: Optional logger instance.
        verbose: If True, logs summary of kept vs. skipped windows.

    Returns:
        groups:  Mapping from window-id string -> fused feature record.
        skipped: List of window-id strings excluded due to missing devices.
    """
    logger = get_logger(logger)
    feature_cols = get_feature_columns(df)

    groups: Dict[str, Dict] = {}
    skipped: List[str] = []

     # Determine prefix templates
    n_left = len(left_macs)
    n_right = len(right_macs)
    left_prefixes = ["L_"] if n_left == 1 else [f"L{i+1}_" for i in range(n_left)]
    right_prefixes = ["R_"] if n_right == 1 else [f"R{i+1}_" for i in range(n_right)]

    # Group the dataframe by window identifiers, in order to get all windows for the different devices
    grouped = df.groupby(
        ["phase_id", "assembly_id", "window_start", "window_end"],
        dropna=False
    )

    for key, group in grouped:
        phase, assembly, ws, we = key
        gid = f"P{phase}_A{assembly}_WS{ws}_WE{we}"

        # Select rows for left and right MACs
        left_rows = group[group["mac"].isin(left_macs)]
        right_rows = group[group["mac"].isin(right_macs)]

        # If there are missing devices for the window, skip the window
        if len(left_rows) != len(left_macs) or len(right_rows) != len(right_macs):
            skipped.append(gid)
            logger.debug(f"Skipping window {gid}: left_rows={len(left_rows)}, right_rows={len(right_rows)}")
            continue

        # Build fused prefixed feature vector
        fused_columns = []
        fused_values = []

        for prefix, (_, row) in zip(left_prefixes, left_rows.iterrows()):
            for c in feature_cols:
                fused_columns.append(f"{prefix}{c}")
                fused_values.append(row[c])
        for prefix, (_, row) in zip(right_prefixes, right_rows.iterrows()):
            for c in feature_cols:
                fused_columns.append(f"{prefix}{c}")
                fused_values.append(row[c])

        meta = left_rows.iloc[0]  # pick metadata from first left row

        groups[gid] = {
            "features": dict(zip(fused_columns, fused_values)),
            "feature_columns": fused_columns,
            "label": meta["label"],
            "label_encoded": meta.get("label_encoded", meta["label"]),
            "label_soft": meta.get("label_soft"),  # Preserve soft labels if present
            "window_id": meta.get("window_id"),
            "phase": phase,
            "assembly": assembly,
            "session": meta.get("session_code", None),
        }

    if verbose:
        logger.info(f"Total windows kept: {len(groups)}; skipped: {len(skipped)}")

    return groups, skipped


# -------------------------
# Fuse windows (raw data)
# -------------------------
def fuse_raw_windows(
    df: pd.DataFrame,
    left_macs: Sequence[str],
    right_macs: Sequence[str],
    raw_cols: Optional[Sequence[str]] = ("x_motion_smooth", "y_motion_smooth", "z_motion_smooth"),
    logger: Optional[logging.Logger] = None,
    verbose: bool = False
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Fuse raw per-window sensor samples from left and right devices.

    For every unique window (phase_id, assembly_id, window_start, window_end),
    this function collects all rows whose `mac` is in `left_macs` or `right_macs`.
    A window is kept only if **all** left MACs and **all** right MACs are present
    exactly once.

    Each device row is reduced to its raw sample columns (e.g. x/y/z arrays).
    These arrays are inserted into a single dictionary with automatically
    assigned prefixes:

        • If only one left MAC:   "L_"
        • If multiple left MACs:  "L1_", "L2_", ...
        • If only one right MAC:  "R_"
        • If multiple right MACs: "R1_", "R2_", ...

    Example output keys for two left MACs and one right MAC:
        L1_x_motion_smooth, L1_y_motion_smooth, L1_z_motion_smooth,
        L2_x_motion_smooth, ...
        R_x_motion_smooth,  R_y_motion_smooth,  R_z_motion_smooth

    Returned structure per window:
        groups[gid] = {
            "raw_data": { "<prefix><col>": np.ndarray, ... },
            "raw_data_columns": list of those keys,
            "label": window label (from first left-row),
            "label_encoded": integer label if present,
            "window_id": optional window identifier,
            "phase": phase_id,
            "assembly": assembly_id,
            "session": optional session_code,
        }

    Args:
        df: Input dataframe containing raw sample columns per row.
        left_macs:  MAC addresses defining the left sensor devices.
        right_macs: MAC addresses defining the right sensor devices.
        raw_cols: Columns containing raw arrays that should be fused.
        logger: Optional logger.
        verbose: If True, prints summary with kept and skipped window counts.

    Returns:
        groups:  Dict mapping window-id string -> fused raw-data record.
        skipped: List of window-ids that were excluded because required
                 devices were missing.
    """
    logger = get_logger(logger)
    groups: Dict[str, Dict] = {}
    skipped: List[str] = []

    # Prefixes for multiple MACs
    n_left, n_right = len(left_macs), len(right_macs)
    left_prefixes  = ["L_"] if n_left == 1 else [f"L{i+1}_" for i in range(n_left)]
    right_prefixes = ["R_"] if n_right == 1 else [f"R{i+1}_" for i in range(n_right)]

    grouped = df.groupby(["phase_id", "assembly_id", "window_start", "window_end"], dropna=False)

    for key, group in grouped:
        phase, assembly, ws, we = key
        gid = f"P{phase}_A{assembly}_WS{ws}_WE{we}"

        left_rows  = group[group["mac"].isin(left_macs)].sort_values("mac").reset_index(drop=True)
        right_rows = group[group["mac"].isin(right_macs)].sort_values("mac").reset_index(drop=True)

        # Skip if any device missing
        if len(left_rows) != n_left or len(right_rows) != n_right:
            skipped.append(gid)
            logger.debug(f"Skipping raw window {gid}: left_rows={len(left_rows)}, right_rows={len(right_rows)}")
            continue

        # Flatten raw columns into prefixed dict
        fused_dict = {}
        for prefix, (_, row) in zip(left_prefixes, left_rows.iterrows()):
            for c in raw_cols:
                fused_dict[f"{prefix}{c}"] = np.asarray(row[c]).tolist()
        for prefix, (_, row) in zip(right_prefixes, right_rows.iterrows()):
            for c in raw_cols:
                fused_dict[f"{prefix}{c}"] = np.asarray(row[c]).tolist()

        meta = left_rows.iloc[0]  # pick metadata from first left row

        groups[gid] = {
            "raw_data": fused_dict,
            "raw_data_columns": list(fused_dict.keys()),
            "label": meta["label"],
            "label_encoded": meta.get("label_encoded", meta["label"]),
            "label_soft": meta.get("label_soft"),  # Preserve soft labels if present
            "window_id": meta.get("window_id"),
            "phase": phase,
            "assembly": assembly,
            "session": meta.get("session_code", None),
        }

    if verbose:
        logger.info(f"[fuse_raw_windows] kept: {len(groups)}, skipped: {len(skipped)}")

    return groups, skipped