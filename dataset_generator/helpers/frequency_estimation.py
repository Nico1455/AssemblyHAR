import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .signal_preprocessing_utils import save_text, save_dataframe


def timestamp_outlier_detection(df: pd.DataFrame, results_folder: Path = None) -> pd.DataFrame:
    """
    Detects and removes timestamp outliers from accelerometer data.

    Args:
        df: Input dataframe with columns 'mac', 'abs_time', and 'timestamp'.
        results_folder: Path to save output summary. If None, summary is only printed.

    Returns:
        Cleaned dataframe with outliers removed.
    """
    if "abs_time" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Dataframe must contain 'abs_time' and 'timestamp' columns.")

    df = df.copy()
    df["abs_time"] = pd.to_datetime(df["abs_time"], errors="coerce")
    df["raw_ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["abs_time", "raw_ts"]).reset_index(drop=True)

    BACKWARD_THRESHOLD_SECONDS = -1.0
    RAW_TS_BACKWARD_THRESHOLD = 0

    # Keep original file order
    df["_file_index"] = np.arange(len(df))
    df = df.sort_values("_file_index").reset_index(drop=True)

    # Compute deltas
    df["prev_abs_time"] = df.groupby("mac")["abs_time"].shift(1)
    df["delta_file_s"] = (df["abs_time"] - df["prev_abs_time"]).dt.total_seconds()

    df["prev_raw_ts"] = df.groupby("mac")["raw_ts"].shift(1)
    df["raw_ts_delta"] = df["raw_ts"] - df["prev_raw_ts"]

    # Detect anomalies
    mask_outliers = (df["delta_file_s"] < BACKWARD_THRESHOLD_SECONDS) | (df["raw_ts_delta"] < RAW_TS_BACKWARD_THRESHOLD)
    num_outliers = mask_outliers.sum()

    summary = []
    summary.append("=== Timestamp Outlier Detection ===")
    
    if num_outliers > 0:
        summary.append(f"Detected {num_outliers} timestamp anomalies.")
        df = df.loc[~mask_outliers].reset_index(drop=True)
        summary.append(f"Removed {num_outliers} rows.")
    else:
        summary.append("No timestamp anomalies detected.")

    # Print summary
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Save summary to file if results_folder is provided
    if results_folder:
        save_text(summary_text, results_folder, "timestamp_outlier_detection")

    # Drop helper columns
    df = df.drop(columns=["_file_index", "prev_abs_time", "delta_file_s", "prev_raw_ts", "raw_ts_delta"], errors="ignore")
    return df


def accelerometer_outlier_detection(df: pd.DataFrame, expected_fs: float = 80.0, tolerance: float = 0.2, max_accel_g: float = 4.0, results_folder: Path = None) -> pd.DataFrame:
    """
    Detects accelerometer data quality issues and removes invalid data.
    
    Performs the following checks:
    - Checks for missing or invalid accelerometer values (x, y, z)
    - Checks for unrealistic acceleration values (exceeding max_accel_g)
    - Detects seconds with abnormal sample counts (reported but not removed)
    
    Args:
        df: Input dataframe with columns 'mac', 'abs_time', 'x', 'y', 'z'.
        expected_fs: Expected sampling frequency in Hz (default: 80.0).
        tolerance: Tolerance for frequency deviation (default: 0.2 = 20%).
        max_accel_g: Maximum acceleration in g (default: 8.0). Values exceeding ±max_accel_g are considered unrealistic.
        results_folder: Path to save output summary. If None, summary is only printed.
    
    Returns:
        Cleaned dataframe with invalid acceleration data removed.
    """
    if "abs_time" not in df.columns:
        raise ValueError("Dataframe must contain 'abs_time' column.")
    
    accel_cols = ['x', 'y', 'z']
    missing_cols = [col for col in accel_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataframe must contain accelerometer columns: {missing_cols}")
    
    df = df.copy()
    df["abs_time"] = pd.to_datetime(df["abs_time"], errors="coerce")
    df = df.dropna(subset=["abs_time"]).reset_index(drop=True)
    
    initial_count = len(df)
    summary = []
    summary.append("=== Accelerometer Data Quality Check ===")
    summary.append(f"Initial sample count: {initial_count:,}")
    
    # 1. Check for missing or invalid accelerometer values
    df[accel_cols] = df[accel_cols].apply(pd.to_numeric, errors='coerce')
    missing_mask = df[accel_cols].isna().any(axis=1)
    num_missing = missing_mask.sum()
    if num_missing > 0:
        summary.append(f"Removing {num_missing:,} rows with missing/invalid accelerometer values")
        df = df.loc[~missing_mask].reset_index(drop=True)
    else:
        summary.append("No missing/invalid accelerometer values detected.")
    
    # 2. Check for unrealistic acceleration values (both positive and negative)
    # Check for values exceeding max_accel_g or -max_accel_g
    extreme_mask = (
        (df[accel_cols[0]] > max_accel_g) | (df[accel_cols[0]] < -max_accel_g) |
        (df[accel_cols[1]] > max_accel_g) | (df[accel_cols[1]] < -max_accel_g) |
        (df[accel_cols[2]] > max_accel_g) | (df[accel_cols[2]] < -max_accel_g)
    )
    num_extreme = extreme_mask.sum()
    if num_extreme > 0:
        summary.append(f"Removing {num_extreme:,} rows with unrealistic acceleration values (> ±{max_accel_g}g)")
        df = df.loc[~extreme_mask].reset_index(drop=True)
    else:
        summary.append(f"No unrealistic acceleration values detected (threshold: ±{max_accel_g}g).")
    
    # 3. Detect (but do not remove) seconds with abnormal sample counts
    df["second"] = df["abs_time"].dt.floor("s") # Converts all timestamps within the same second to identical values
    # Create new dataframe showing how many samples each watch collected in each second
    samples_per_second = df.groupby(["mac", "second"]).size().reset_index(name="sample_count")
    
    # Define acceptable range
    min_samples = int(expected_fs * (1 - tolerance))
    max_samples = int(expected_fs * (1 + tolerance))
    
    outlier_seconds = samples_per_second[
        (samples_per_second["sample_count"] < min_samples) | 
        (samples_per_second["sample_count"] > max_samples)
    ]
    
    num_outlier_seconds = len(outlier_seconds)
    summary.append("\n=== Sampling Frequency Analysis ===")
    if num_outlier_seconds > 0:
        summary.append(f"Detected {num_outlier_seconds} seconds with abnormal sample counts (not removed):")
        summary.append(f"  Expected range: {min_samples}-{max_samples} samples/second")
        summary.append(f"  Too low (<{min_samples}): {(outlier_seconds['sample_count'] < min_samples).sum()} seconds")
        summary.append(f"  Too high (>{max_samples}): {(outlier_seconds['sample_count'] > max_samples).sum()} seconds")
    else:
        summary.append(f"No frequency outliers detected (acceptable range: {min_samples}-{max_samples} samples/second)")
    
    df = df.drop(columns=["second"], errors="ignore")
    
    final_count = len(df)
    total_removed = initial_count - final_count
    summary.append(f"\nTotal samples removed: {total_removed:,} ({total_removed/initial_count*100:.2f}%)")
    summary.append(f"Final sample count: {final_count:,}")
    
    # Print and save summary
    summary_text = "\n".join(summary)
    print(summary_text)
    
    if results_folder:
        save_text(summary_text, results_folder, "accelerometer_outlier_detection")
    
    return df


def compute_acceleration_statistics(df: pd.DataFrame, results_folder: Path = None, verbose: bool = True) -> dict:
    """
    Computes statistics for acceleration data and saves them.
    
    Calculates statistics for each axis (x, y, z) and magnitude including:
    - Min, max, range
    - Mean, median, standard deviation, variance
    - Quartiles (25th, 50th, 75th percentiles)
    - Interquartile range (IQR)
    - Skewness and kurtosis
    - 5th and 95th percentiles
    
    Args:
        df: Input dataframe with columns 'x', 'y', 'z'.
        results_folder: Path to save output statistics. If None, stats are only printed (This is used).
        verbose: Whether to print the formatted summary text.
    
    Returns:
        Dictionary containing all computed statistics.
    """
    accel_cols = ['x', 'y', 'z']
    missing_cols = [col for col in accel_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataframe must contain accelerometer columns: {missing_cols}")
    
    # Ensure numeric data
    df[accel_cols] = df[accel_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=accel_cols)
    
    # Compute magnitude
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    stats = {}
    summary = []
    summary.append("=== Acceleration Statistics ===\n")
    
    for axis in accel_cols + ['magnitude']:
        data = df[axis]
        
        axis_stats = {
            'count': len(data),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'variance': data.var(),
            'q25': data.quantile(0.25),
            'q50': data.quantile(0.50),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'p5': data.quantile(0.05),
            'p95': data.quantile(0.95),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
        
        stats[axis] = axis_stats
        
        # Format output
        axis_label = axis.upper() if axis in accel_cols else 'Magnitude'
        summary.append(f"--- {axis_label} Axis ---")
        summary.append(f"  Count:           {axis_stats['count']:,}")
        summary.append(f"  Min:             {axis_stats['min']:.4f} g")
        summary.append(f"  Max:             {axis_stats['max']:.4f} g")
        summary.append(f"  Range:           {axis_stats['range']:.4f} g")
        summary.append(f"  Mean:            {axis_stats['mean']:.4f} g")
        summary.append(f"  Median:          {axis_stats['median']:.4f} g")
        summary.append(f"  Std Dev:         {axis_stats['std']:.4f} g")
        summary.append(f"  Variance:        {axis_stats['variance']:.4f} g²")
        summary.append(f"  25th Percentile: {axis_stats['q25']:.4f} g")
        summary.append(f"  50th Percentile: {axis_stats['q50']:.4f} g")
        summary.append(f"  75th Percentile: {axis_stats['q75']:.4f} g")
        summary.append(f"  IQR:             {axis_stats['iqr']:.4f} g")
        summary.append(f"  5th Percentile:  {axis_stats['p5']:.4f} g")
        summary.append(f"  95th Percentile: {axis_stats['p95']:.4f} g")
        summary.append(f"  Skewness:        {axis_stats['skewness']:.4f}")
        summary.append(f"  Kurtosis:        {axis_stats['kurtosis']:.4f}")
        summary.append("")
    
    # Print and save summary
    summary_text = "\n".join(summary)
    if verbose:
        print(summary_text)
    
    if results_folder:
        save_text(summary_text, results_folder, "acceleration_statistics")
        stats_df = pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(columns={"index": "axis"})
        save_dataframe(stats_df, results_folder, "acceleration_statistics_table")
    
    return stats


def estimate_fs(clean_df: pd.DataFrame, results_folder: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Estimates sampling frequency per watch and prepares data for plotting.

    Args:
        clean_df: Dataframe after outlier removal.

    Returns:
        freq_df: Dataframe with columns ['mac', 'second', 'samples_per_second'].
    """
    df = clean_df.copy()
    df["second"] = df["abs_time"].dt.floor("s")

    # Compute samples per second
    freq_df = df.groupby(["mac", "second"]).size().reset_index(name="samples_per_second")

    # Compute average frequency per watch
    avg_freq = freq_df.groupby("mac")["samples_per_second"].mean()

    # Build summary text
    summary = []

    summary.append("=== Average Sampling Frequency per Watch (Hz, per-second method) ===")
    for mac, freq in avg_freq.items():
        summary.append(f"{mac}: {freq:.2f} Hz")

    summary.append("\n=== Sampling Frequency per Watch (total duration method) ===")
    for mac, subdf in df.groupby("mac"):
        total_samples = len(subdf)
        duration_sec = (subdf["abs_time"].max() - subdf["abs_time"].min()).total_seconds()
        if duration_sec > 0:
            freq_total = total_samples / duration_sec
            summary.append(f"{mac}: {freq_total:.2f} Hz ({total_samples:,} samples, {duration_sec:.1f} sec)")
        else:
            summary.append(f"{mac}: Duration too short or invalid.")

    total_duration_min = (df["second"].max() - df["second"].min()).total_seconds() / 60
    summary.append(f"\nTotal samples after cleaning: {len(df):,}")
    summary.append(f"Total duration: {total_duration_min:.2f} minutes")

    # Convert summary to text
    summary_text = "\n".join(summary)
    print(summary_text)

    # Save summary text
    save_text(summary_text, results_folder, "avg_frequency_summary")

    return freq_df, avg_freq



def plot_frequency(freq_df: pd.DataFrame, results_folder: Path, rolling_window: int = 10):
    """
    Plots rolling sampling frequency per watch over time.

    Args:
        freq_df: Dataframe with columns ['mac', 'second', 'samples_per_second'].
        rolling_window: Window size (in seconds) for rolling average.
    """
    plt.figure(figsize=(12, 6))

    for mac, subdf in freq_df.groupby("mac"):
        subdf = subdf.sort_values("second").copy()
        subdf["rolling_freq"] = subdf["samples_per_second"].rolling(window=rolling_window, center=True).mean()
        plt.plot(subdf["second"], subdf["rolling_freq"], label=f"{mac} ({rolling_window}s avg)")

    plt.xlabel("Time")
    plt.ylabel("Samples per second (Hz)")
    plt.title("Sampling Frequency Over Time")
    plt.legend(title="Watch MAC", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the resuls subfolder
    outfile = results_folder / "sampling_frequency_plot.png"
    plt.savefig(outfile, dpi=300)
    #plt.show()
    plt.close()