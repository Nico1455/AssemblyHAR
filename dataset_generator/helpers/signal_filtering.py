import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

_SERIF_FONT_STACK = [
    "Computer Modern Roman",
    "Latin Modern Roman",
    "CMU Serif",
    "TeX Gyre Termes",
    "Times New Roman",
    "Times",
    "DejaVu Serif",
]


# -----------------------
# Filtering utilities
# -----------------------
def lowpass_filter(x, cutoff, fs, order=3):
    """Low-pass Butterworth filter to estimate gravity."""
    if len(x) < 8:
        return x

    nyquist = 0.5 * fs
    wn = min(max(cutoff / nyquist, 1e-6), 0.99)  # protect filter
    b, a = butter(order, wn, btype="low", analog=False)

    try:
        return filtfilt(b, a, x, method="pad")
    except Exception:
        return filtfilt(b, a, x, method="gust")


def median_smooth(x, kernel=3):
    """Remove spikes with median filter."""
    if len(x) < kernel:
        return x
    if kernel % 2 == 0:
        kernel += 1
    return medfilt(x, kernel_size=kernel)


def moving_average(x, window):
    """Moving average smoothing."""
    if len(x) < window:
        return x
    w = np.ones(window) / window
    return np.convolve(x, w, mode="same")


# -----------------------
# Main filtering function
# -----------------------
def filter_signal(df: pd.DataFrame, avg_freq: pd.Series, cutoff: float = 0.25) -> pd.DataFrame:
    """
    Apply median smoothing, gravity removal, and moving-average smoothing
    to labeled accelerometer data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns: x, y, z, mac, abs_time
    avg_freq : pd.Series
        Average sampling frequency per watch (from main)
    cutoff : float
        Low-pass cutoff frequency (Hz)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with additional columns:
        *_gravity, *_motion, *_motion_smooth
    """
    df_filtered = df.copy()

    for mac in df_filtered["mac"].unique():
        mask = df_filtered["mac"] == mac
        fs = avg_freq.get(mac, 80.0)  # fallback if missing

        smooth_win = max(3, int(round(fs * 0.10)))  # ~0.25 s smoothing

        for axis in ["x", "y", "z"]:
            raw = df_filtered.loc[mask, axis].to_numpy()

            # Light median filter to remove spikes
            raw_med = median_smooth(raw, kernel=3)

            # Gravity estimate via low-pass filter
            grav = lowpass_filter(raw_med, cutoff=cutoff, fs=fs)
            df_filtered.loc[mask, f"{axis}_gravity"] = grav

            # Motion = raw - gravity
            motion = raw_med - grav
            df_filtered.loc[mask, f"{axis}_motion"] = motion

            # Moving average smoothing
            smooth = moving_average(motion, window=smooth_win)
            df_filtered.loc[mask, f"{axis}_motion_smooth"] = smooth

    return df_filtered


# -----------------------
# Plotting function
# -----------------------
def plot_filtered_signal(
    df_filtered: pd.DataFrame,
    results_folder: Path,
    label: str | None = None,
    assembly_id: int | None = None,
    phase_id: int | None = None,
    activity_id: int | None = None,
    max_samples: int = 3000,
    export_formats: tuple[str, ...] = ("png", "pdf", "svg")
):
    """
    Generate and save filtered accelerometer plots for each watch with all axes
    in one figure. Allows filtering by activity, assembly, or phase.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered accelerometer data with *_gravity, *_motion, *_motion_smooth columns.
    results_folder : Path
        Folder where plots will be saved.
    label : str, optional
        Filter to a specific activity label.
    assembly_id : int, optional
        Filter to a specific assembly ID.
    phase_id : int, optional
        Filter to a specific phase ID.
    max_samples : int, default 3000
        Maximum number of samples to plot for readability.
    export_formats : tuple[str, ...], default ("png", "pdf", "svg", "pgf")
        File formats to save. PGF is disabled because it requires a TeX installation.
    """
    results_folder.mkdir(exist_ok=True, parents=True)

    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": _SERIF_FONT_STACK,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 16,
        "mathtext.fontset": "dejavuserif",
        "pgf.rcfonts": False,
        # "pgf.texsystem": "pdflatex",  # Disabled: requires pdflatex/TeX on the system.
    }):
        for mac in df_filtered["mac"].unique():
            subset = df_filtered[df_filtered["mac"] == mac].copy()
            mask = pd.Series(True, index=subset.index)

            if label is not None:
                mask &= subset["label"] == label
            if assembly_id is not None:
                mask &= subset["assembly_id"] == assembly_id
            if phase_id is not None:
                mask &= subset["phase_id"] == phase_id
            if activity_id is not None and "activity_id" in subset.columns:
                mask &= subset["activity_id"] == activity_id

            subset = subset[mask]

            if subset.empty:
                print(f"No data found for device {mac} with the selected filters.")
                continue  # skip this MAC

            if len(subset) > max_samples:
                subset = subset.iloc[:max_samples]

            subset = subset.sort_values("abs_time")
            time = pd.to_datetime(subset["abs_time"])

            # Save one separate figure per axis so each signal can be inspected independently.
            for axis in ["x", "y", "z"]:
                fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.plot(time, subset[axis], label="Original Signal", linewidth=0.8, alpha=0.7)
                ax.plot(time, subset[f"{axis}_gravity"], label="Gravity (Low-pass)", linewidth=1.2)
                ax.plot(time, subset[f"{axis}_motion"], label="Motion (Raw–Gravity)", linewidth=0.9, alpha=0.8)
                ax.plot(time, subset[f"{axis}_motion_smooth"], label="Smoothed Motion", linewidth=1.4)
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{axis.upper()} (g)")
                ax.legend(loc="upper right")
                ax.grid(True, linestyle="--", alpha=0.5)
                fig.tight_layout()

                for export_format in export_formats:
                    outfile = results_folder / f"{mac}_{axis}_filtered_plot.{export_format}"
                    save_kwargs = {"bbox_inches": "tight"}
                    if export_format == "png":
                        save_kwargs["dpi"] = 300
                    fig.savefig(outfile, **save_kwargs)
                plt.close(fig)

