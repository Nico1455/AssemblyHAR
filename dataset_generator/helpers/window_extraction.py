import pandas as pd
import numpy as np
from pathlib import Path
from typing import Sequence, List

from .feature_extraction import extract_features
from .constants import EXCLUDED_COLUMNS
from .wavelet_utils import add_wavelet_features_to_window

def generate_windows(
    assembly_segments: pd.DataFrame,
    phase_segments: pd.DataFrame,
    window_size_sec: float,
    step_size_sec: float,
    left_macs: Sequence[str],
    right_macs: Sequence[str],
) -> list[dict]:
    """
    Build a complete list of sliding windows aligned across devices.
    The aligned window boundaries are only created inside assemblies and for the baseline of the first two phases.

    Returns a list of dicts:
        - window_start
        - window_end
        - phase_id
        - assembly_id
    """
    windows = []
    num_device = len(left_macs) + len(right_macs)  # Number of devices used for the experiment

    # ---------------------------------------------------
    # ASSEMBLY WINDOWS
    # ---------------------------------------------------
    for _, row in assembly_segments.iterrows():
        a_start = row["start_time"]
        a_end = row["end_time"]

        current = a_start
        while current + pd.Timedelta(seconds=window_size_sec) <= a_end:
            windows.append({
                "window_start": current,
                "window_end": current + pd.Timedelta(seconds=window_size_sec),
                "phase_id": row["phase_id"],
                "assembly_id": row["assembly_id"],
            })
            current += pd.Timedelta(seconds=step_size_sec)

    # ---------------------------------------------------
    # BASELINE WINDOWS (first two phases only)
    # ---------------------------------------------------
    # Identify the first two phases
    first_two_phases = sorted(phase_segments["phase_id"].unique())[:2]
    baseline_windows = []

    for phase_id in first_two_phases:
        ph = phase_segments[phase_segments["phase_id"] == phase_id].iloc[0]
        # Phase start
        phase_start = ph["start_time"]

        # Assemblies in this phase
        assemblies = assembly_segments[assembly_segments["phase_id"] == phase_id]
        if assemblies.empty:
            continue
        
        # Only take the part before the first assembly
        first_assembly_start = assemblies["start_time"].min()

        # If there is baseline time:
        if phase_start < first_assembly_start:
            current = phase_start
            baseline_end = first_assembly_start

            while current + pd.Timedelta(seconds=window_size_sec) <= baseline_end:
                baseline_windows.append({
                    "window_start": current,
                    "window_end": current + pd.Timedelta(seconds=window_size_sec),
                    "phase_id": phase_id,
                    "assembly_id": None,
                })
                current += pd.Timedelta(seconds=step_size_sec)

    windows.extend(baseline_windows)

    # The total number of windows generated is calculated across all devices used for the experiment
    windows_generated = len(windows)*num_device
    print(f"[WindowGen] Total windows generated: {windows_generated}")
    return windows, windows_generated


def generate_activity_windows(
    activity_segments: pd.DataFrame,
    phase_segments: pd.DataFrame,
    left_macs: Sequence[str],
    right_macs: Sequence[str],
) -> list[dict]:
    """
    Build a list of windows where each window spans an entire activity.
    No overlaps, each activity gets exactly one window.

    Additionally includes baseline windows from the first two phases.

    Returns a list of dicts:
        - window_start
        - window_end
        - phase_id
        - assembly_id
        - activity_id (for activity windows)
        - label (for activity windows)
    """
    windows = []
    num_device = len(left_macs) + len(right_macs)

    # ---------------------------------------------------
    # ACTIVITY WINDOWS (full activity duration)
    # ---------------------------------------------------
    for _, row in activity_segments.iterrows():
        windows.append({
            "window_start": row["start_time"],
            "window_end": row["end_time"],
            "phase_id": row["phase_id"],
            "assembly_id": row["assembly_id"],
            "activity_id": row.get("activity_id"),
            "label": row.get("label"),
        })

    # ---------------------------------------------------
    # BASELINE WINDOWS (first two phases only)
    # ---------------------------------------------------
    # For baseline, we still need to create windows from the phase segments
    first_two_phases = sorted(phase_segments["phase_id"].unique())[:2]
    baseline_windows = []

    for phase_id in first_two_phases:
        ph = phase_segments[phase_segments["phase_id"] == phase_id].iloc[0]
        phase_start = ph["start_time"]

        # Get assemblies in this phase from activity_segments
        activities_in_phase = activity_segments[activity_segments["phase_id"] == phase_id]
        if activities_in_phase.empty:
            continue
        
        # Find the start of the first activity/assembly
        first_activity_start = activities_in_phase["start_time"].min()

        # If there is baseline time before first activity
        if phase_start < first_activity_start:
            baseline_windows.append({
                "window_start": phase_start,
                "window_end": first_activity_start,
                "phase_id": phase_id,
                "assembly_id": None,
                "activity_id": None,
                "label": "Baseline",
            })

    windows.extend(baseline_windows)

    windows_generated = len(windows) * num_device
    print(f"[ActivityWindowGen] Total activity windows generated: {windows_generated}")
    return windows, windows_generated


# ============================================================
# Add Deterministic Window ID
# ============================================================
def add_window_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a reproducible string window identifier based on
    (phase_id, assembly_id, window_start, window_end).

    Baseline windows (assembly_id is None/NaN) are encoded as "AsBl".

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe including a 'window_id' column.
    """
    df = df.copy()

    if pd.api.types.is_datetime64_any_dtype(df["window_start"]):
        ws_str = df["window_start"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    else:
        ws_str = df["window_start"].astype(str)

    if pd.api.types.is_datetime64_any_dtype(df["window_end"]):
        we_str = df["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    else:
        we_str = df["window_end"].astype(str)

    # Handle assembly_id
    assembly_id = df["assembly_id"]

    # assembly_id == NaN → Baseline ("Bl")
    assembly_str = assembly_id.where(~assembly_id.isna(), "Bl").astype(str)

    df["window_id"] = (
        "Ph" + df["phase_id"].astype(str) +
        "_As" + assembly_str +
        "_WS" + ws_str +
        "_WE" + we_str
    )

    return df

def remove_first_assembly_windows(windowed_df: pd.DataFrame, n_assemblies: int) -> pd.DataFrame:
    """
    Remove windows belonging to the first N assemblies from the first two phases (phase_id 1 and 2).

    Parameters
    ----------
    windowed_df : pd.DataFrame
        Windowed data with columns: phase_id, assembly_id, etc.
    n_assemblies : int
        Number of assemblies to remove from each of the first two phases.

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified assembly windows removed from phases 1 and 2.
    """
    result_df = windowed_df.copy()
    
    # Remove first N assemblies from phase 1 and phase 2
    for phase_id in [1, 2]:
        assemblies_to_remove = list(range(1, n_assemblies + 1))  # Range is exclusive on the right side, thats why +1
        mask = (result_df["phase_id"] == phase_id) & (result_df["assembly_id"].isin(assemblies_to_remove))
        result_df = result_df[~mask]  # ~ inverts the boolean mask, keep everything except the mask
        print(f"Removed {n_assemblies} assemblies from phase {phase_id}.")
    
    return result_df

def empty_windows_summary(
        empty_windows: pd.DataFrame,
        results_folder: str,
        num_windows_generated: int,
        features_or_raw: str
    ):
    """
    Create a summary CSV with counts of empty windows by:
    - MAC
    - phase
    - assembly
    - (phase × assembly)
    """
    if len(empty_windows) > 0:
        empty_df = pd.DataFrame(empty_windows)
        summary_rows = []
        # Total count
        total_empty = len(empty_df)
        summary_rows.append({
            "category": "total_empty_windows",
            "value": total_empty
        })
        # Total windows generated
        summary_rows.append({
            "category": "total_windows",
            "value": num_windows_generated
        })
        # Ratio of missing windows
        ratio_missing = total_empty / num_windows_generated if num_windows_generated > 0 else 0.0
        summary_rows.append({
            "category": "ratio_missing_windows",
            "value": ratio_missing
        })
        # Per MAC
        per_mac = (
            empty_df.groupby("mac")
            .size()
            .reset_index(name="count")
        )
        for _, row in per_mac.iterrows():
            summary_rows.append({
                "category": f"Missing windows mac_{row['mac']}",
                "value": row["count"]
            })
        # Per phase
        per_phase = (
            empty_df.groupby("phase_id")
            .size()
            .reset_index(name="count")
        )
        for _, row in per_phase.iterrows():
            summary_rows.append({
                "category": f"Missing windows phase_{row['phase_id']}",
                "value": row["count"]
            })
        # Per assembly
        per_assembly = (
            empty_df.groupby("assembly_id")
            .size()
            .reset_index(name="count")
        )
        for _, row in per_assembly.iterrows():
            summary_rows.append({
                "category": f"Missing windows assembly_{row['assembly_id']}",
                "value": row["count"]
            })
        # Per (phase × assembly)
        per_phase_assembly = (
            empty_df.groupby(["phase_id", "assembly_id"])
            .size()
            .reset_index(name="count")
        )
        for _, row in per_phase_assembly.iterrows():
            summary_rows.append({
                "category": f"Missing windows phase_{row['phase_id']}_assembly_{row['assembly_id']}",
                "value": row["count"]
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = Path(results_folder) / f"empty_windows_summary_{features_or_raw}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved empty window summary to {summary_path.resolve()}")


# ---------------------------------------
# Sliding window extraction with features
# ---------------------------------------
def sliding_window_features(
    df: pd.DataFrame,
    results_folder: str,
    window_defs: list[dict],
    avg_freq: dict[str, float],
    session_code: str,
    num_windows_generated: int,
    first_assemblies_to_remove: int = 0,
    non_learning_session: bool = False,
    use_soft_labels: bool = False
) -> pd.DataFrame:
    """
    Perform sliding window feature extraction for all watches using
    a shared global time axis so that windows align across devices.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed accelerometer data with columns including 'label', 'mac', 'abs_time', etc.
    results_folder : str
        Folder where the label encoder will be stored.
    window_defs : list[dict]
        Complete list of sliding window definitions aligned across devices with keys:
        - window_start, window_end, phase_id, assembly_id
    avg_freq : dict[str, float]
        Mapping from MAC address to average sampling frequency (Hz).
    session_code : str
        Code of the session that is currently being processed.
    num_windows_generated : int
        Total windows generated across all devices.
    first_assemblies_to_remove : int
        Number of first assemblies to remove from phases 1-2.
    use_soft_labels : bool
        If True, compute per-window label distributions (soft labels).

    Returns
    -------
    pd.DataFrame
        DataFrame containing extracted features per window per MAC, including:
        - mac, window_start, window_end
        - label (string activity label via majority voting)
        - label_soft (dict of encoded_label -> probability) if use_soft_labels=True
          Example: {0: 0.6, 1: 0.3, 2: 0.1} where keys are label_encoded values
        - phase_id, assembly_id, session_code
        - All time/frequency/wavelet features from extract_features()
        - label_encoded, window_id
    """
    # 1) Extract features for each MAC using the same windows
    all_features = []
    empty_windows = []

    for mac in df["mac"].unique():
        df_mac = df[df["mac"] == mac].sort_values("abs_time")
        fs = avg_freq.get(mac, 80.0)  # fallback frequency

        for window in window_defs:
            w_start = window["window_start"]
            w_end = window["window_end"]
            w_phase = window["phase_id"]
            w_assembly = window["assembly_id"]

            # Select rows in this window
            w_df = df_mac[(df_mac["abs_time"] >= w_start) & (df_mac["abs_time"] < w_end)]

            # Collect windows with 0 or 1 samples → log them
            if len(w_df) <= 1:
                if len(w_df) == 1:
                    row = w_df.iloc[0]
                    empty_windows.append({
                        "mac": mac,
                        "window_start": w_start,
                        "window_end": w_end,
                        "label": row.get("label", None),
                        "phase_id": w_phase,
                        "assembly_id": w_assembly,
                    })
                else:
                    # truly empty → no rows at all
                    empty_windows.append({
                        "mac": mac,
                        "window_start": w_start,
                        "window_end": w_end,
                        "label": None,
                        "phase_id": w_phase,
                        "assembly_id": w_assembly,
                    })
                continue

            # Derive majority label and optional soft distribution
            labels_series = w_df["label"].dropna()
            label_counts = labels_series.value_counts()
            majority_label = label_counts.idxmax() if not label_counts.empty else None
            label_soft = None
            if use_soft_labels and not label_counts.empty:
                # Use label_encoded to ensure ordering matches model expectations
                encoded_series = w_df["label_encoded"].dropna()
                encoded_counts = encoded_series.value_counts()
                probs = encoded_counts / encoded_counts.sum()
                # Store as {encoded_label: probability} for correct ordering
                label_soft = {int(k): float(v) for k, v in probs.to_dict().items()}

            # Extract features
            feats = extract_features(w_df, fs=fs)
            feats_label = feats.pop("label", majority_label)
            feats_phase = feats.pop("phase_id", w_phase)
            feats_assembly = feats.pop("assembly_id", w_assembly)

            ordered = {
                "mac": mac,
                "window_start": w_start,
                "window_end": w_end,
                "label": majority_label if majority_label is not None else feats_label,
                "phase_id": feats_phase,
                "assembly_id": feats_assembly,
                "session_code": session_code,
                **feats
            }
            if use_soft_labels and label_soft is not None:
                ordered["label_soft"] = label_soft

            all_features.append(ordered)

    feature_df = pd.DataFrame(all_features)

    # 2) Write empty windows to CSV
    if len(empty_windows) > 0:
        empty_df = pd.DataFrame(empty_windows)
        out_path = Path(results_folder) / "empty_windows_features.csv"
        empty_df.to_csv(out_path, index=False)
        print(f"[Feature Windowing] Saved empty window log to {out_path.resolve()}")
    else:
        print("[Feature Windowing] No empty windows detected.")

    # 3) Create the empty windows summary
    empty_windows_summary(pd.DataFrame(empty_windows), results_folder, num_windows_generated, features_or_raw = "features")

    # 4) Add an unique window_id to each window 
    feature_df = add_window_id(feature_df)

    # 5) Remove first N assemblies from the first two phases if it is a learning session
    if not non_learning_session and first_assemblies_to_remove > 0:
        feature_df = remove_first_assembly_windows(feature_df, first_assemblies_to_remove)

    return feature_df


# ---------------------------------------
# Sliding window extraction with raw data
# ---------------------------------------
def sliding_window_raw(
    df: pd.DataFrame,
    results_folder: str,
    window_defs: list[dict],
    session_code: str,
    num_windows_generated: int,
    first_assemblies_to_remove: int = 0,
    non_learning_session: bool = False,
    use_soft_labels: bool = False,
    add_wavelet_features: bool = False,
    wavelet_family: str = "morl",
    wavelet_scales: List[int] | None = None,
    use_pywt: bool = False
) -> pd.DataFrame:
    """
    Create sliding windows containing RAW ACCELEROMETER SAMPLES
    (instead of feature vectors).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed accelerometer data with columns including 'label'.
    window_defs : list[dict]
        List of dicts with:
            - window_start
            - window_end
            - phase_id
            - assembly_id
    session_code : str
        Session identifier.
    results_folder : str
        Folder to store empty window logs.
    num_windows_generated : int
        Total windows generated across all devices.
    first_assemblies_to_remove : int
        Number of first assemblies to remove from phases 1-2.
    use_soft_labels : bool
        If True, include label probability distributions.
    add_wavelet_features : bool
        If True, compute and add CWT wavelet features to raw data.
    wavelet_family : str
        Wavelet family for CWT ('morl', 'mexh', 'db4', etc.).
    wavelet_scales : List[int] | None
        CWT scales. If None, defaults to [1, 2, 4, 8].

    Returns
    -------
    pd.DataFrame
        Row per window per MAC:
            - mac
            - window_start
            - window_end
            - phase_id
            - assembly_id
            - session_code
            - label (string activity label from majority voting)
            - label_soft (dict of encoded_label -> probability) if use_soft_labels=True
              Example: {0: 0.6, 1: 0.3, 2: 0.1} where keys are label_encoded values
            - x_motion_smooth
            - y_motion_smooth
            - z_motion_smooth
            - [Wavelet features] if add_wavelet_features=True
            - window_id
    """
    
    if wavelet_scales is None:
        wavelet_scales = [1, 2, 4, 8]
    
    windows_raw = []
    empty_windows = []

    # Create sliding windows for each device
    for mac in df["mac"].unique():
        df_mac = df[df["mac"] == mac].sort_values("abs_time")

        # Use the provided window definitions to have consistent windowing across devices
        for w in window_defs:
            w_start = w["window_start"]
            w_end = w["window_end"]
            phase_id = w["phase_id"]
            assembly_id = w["assembly_id"]

            # Select rows in the current window
            w_df = df_mac[(df_mac["abs_time"] >= w_start) & (df_mac["abs_time"] < w_end)]

            # Handle empty or nearly-empty windows
            if len(w_df) <= 1:
                empty_windows.append({
                    "mac": mac,
                    "window_start": w_start,
                    "window_end": w_end,
                    "phase_id": phase_id,
                    "assembly_id": assembly_id,
                })
                continue

            # Derive majority label and optional soft distribution
            labels_series = w_df["label"].dropna()
            label_counts = labels_series.value_counts()
            majority_label = label_counts.idxmax() if not label_counts.empty else None
            label_soft = None
            if use_soft_labels and not label_counts.empty:
                # Use label_encoded to ensure ordering matches model expectations
                encoded_series = w_df["label_encoded"].dropna()
                encoded_counts = encoded_series.value_counts()
                probs = encoded_counts / encoded_counts.sum()
                # Store as {encoded_label: probability} for correct ordering
                label_soft = {int(k): float(v) for k, v in probs.to_dict().items()}

            # Extract smoothed motion signals as arrays
            x_motion_smooth = w_df["x_motion_smooth"].values.tolist()
            y_motion_smooth = w_df["y_motion_smooth"].values.tolist()
            z_motion_smooth = w_df["z_motion_smooth"].values.tolist()

            record = {
                "mac": mac,
                "window_start": w_start,
                "window_end": w_end,
                "phase_id": phase_id,
                "assembly_id": assembly_id,
                "session_code": session_code,
                "label": majority_label,
                "x_motion_smooth": x_motion_smooth,
                "y_motion_smooth": y_motion_smooth,
                "z_motion_smooth": z_motion_smooth,
            }
            
            # Add wavelet features if enabled
            if add_wavelet_features:
                window_array = np.column_stack([x_motion_smooth, y_motion_smooth, z_motion_smooth])
                combined_array = add_wavelet_features_to_window(
                    window_array,
                    wavelet=wavelet_family,
                    scales=wavelet_scales,
                    normalize=True
                )
                # Split into wavelet feature columns
                # combined_array shape: (time_steps, 3 raw + 3 * len(scales) wavelet features)
                n_raw_sensors = 3
                n_scales = len(wavelet_scales)
                
                # Extract wavelet features (skip the first 3 columns which are raw data)
                for scale_idx, scale in enumerate(wavelet_scales):
                    for sensor_idx, sensor_name in enumerate(['x', 'y', 'z']):
                        # Wavelet feature column index in combined_array
                        col_idx = n_raw_sensors + sensor_idx * n_scales + scale_idx
                        wavelet_col_data = combined_array[:, col_idx].tolist()
                        record[f"{sensor_name}_wavelet_scale{scale}"] = wavelet_col_data
            
            if use_soft_labels and label_soft is not None:
                record["label_soft"] = label_soft

            windows_raw.append(record)

    # Log empty windows
    if len(empty_windows) > 0:
        empty_df = pd.DataFrame(empty_windows)
        out_path = Path(results_folder) / "empty_windows_raw.csv"
        empty_df.to_csv(out_path, index=False)
        print(f"[Raw Windowing] Saved empty window log to {out_path.resolve()}")
    else:
        print("[Raw Windowing] No empty windows detected.")

    # Create the empty windows summary
    empty_windows_summary(pd.DataFrame(empty_windows), results_folder, num_windows_generated, features_or_raw = "raw")

    # Add an unique window_id to each window 
    windows_raw = pd.DataFrame(windows_raw)    
    windows_raw = add_window_id(windows_raw)

    # Remove first N assemblies from the first two phases if it is a learning session
    if not non_learning_session and first_assemblies_to_remove > 0:
        windows_raw = remove_first_assembly_windows(windows_raw, first_assemblies_to_remove)

    return windows_raw