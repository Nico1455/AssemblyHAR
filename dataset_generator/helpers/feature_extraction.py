import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import welch, correlate
import pywt
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Wavelet features
# -----------------------
def wavelet_features(signal, wavelet='db4', level=3):
    """Extract energy, entropy, mean, std, rms for each wavelet coefficient."""
    features = {}
    if len(signal) < 4:
        return features
    
    # Automatically reduce level if signal is too short to avoid boundary effects
    # Rule of thumb: signal length should be >= 2^(level+1)
    max_safe_level = int(np.floor(np.log2(len(signal)))) - 1
    actual_level = min(level, max_safe_level)
    
    coeffs = pywt.wavedec(signal, wavelet, level=actual_level)
    for i, c in enumerate(coeffs):
        prefix = 'A' if i == 0 else f'D{i}'
        energy = np.sum(c**2) / len(c)
        P = c**2 / (np.sum(c**2) + 1e-12)
        entropy = -np.sum(P * np.log2(P + 1e-12))
        features[f'{prefix}_energy'] = energy
        features[f'{prefix}_entropy'] = entropy
        features[f'{prefix}_mean'] = np.mean(c)
        features[f'{prefix}_std'] = np.std(c)
        features[f'{prefix}_rms'] = np.sqrt(np.mean(c**2))
    return features

# -----------------------
# Features per window
# -----------------------
def extract_features(window_df, fs=80):
    """Extract time, frequency, and wavelet features for a window."""
    features = {}
    axes = ['x_motion_smooth', 'y_motion_smooth', 'z_motion_smooth']
    x, y, z = [window_df[a].values for a in axes]
    dm = np.sqrt(x**2 + y**2 + z**2)
    axis_dict = dict(zip(axes + ['dm'], [x, y, z, dm]))

    for name, arr in axis_dict.items():
        features[f'{name}_mean'] = np.mean(arr)
        features[f'{name}_std'] = np.std(arr)
        features[f'{name}_rms'] = np.sqrt(np.mean(arr**2))
        features[f'{name}_range'] = np.ptp(arr)
        features[f'{name}_iqr'] = np.percentile(arr, 75) - np.percentile(arr, 25)
        features[f'{name}_skew'] = skew(arr)
        features[f'{name}_kurtosis'] = kurtosis(arr)

        # Frequency domain
        f, Pxx = welch(arr, fs=fs, nperseg=min(len(arr), 256))
        if len(Pxx) > 0:
            df = f[1] - f[0] if len(f) > 1 else 0.0
            avg_power = np.sum(Pxx) * df
            window_energy = avg_power * (len(arr) / fs)

            features[f'{name}_power'] = avg_power
            features[f'{name}_energy'] = window_energy

            Pxx_norm = Pxx / (np.sum(Pxx) + 1e-12)
            features[f'{name}_spectral_entropy'] = -np.sum(Pxx_norm*np.log2(Pxx_norm + 1e-12))
            features[f'{name}_peak_power_freq'] = f[np.argmax(Pxx)]

        # Wavelet
        w_feats = wavelet_features(arr, wavelet='db4', level=3)
        for k, v in w_feats.items():
            features[f'{name}_{k}'] = v

    # Cross-axis correlations
    features['corr_xy'] = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    features['corr_xz'] = np.corrcoef(x, z)[0, 1] if len(x) > 1 else 0
    features['corr_yz'] = np.corrcoef(y, z)[0, 1] if len(y) > 1 else 0

    # Max cross-correlation
    features['xcorr_xy'] = np.max(np.abs(correlate(x, y, mode='full'))) if len(x) > 1 else 0
    features['xcorr_xz'] = np.max(np.abs(correlate(x, z, mode='full'))) if len(x) > 1 else 0
    features['xcorr_yz'] = np.max(np.abs(correlate(y, z, mode='full'))) if len(y) > 1 else 0

    # SMA (Signal Magnitude Area)
    features['sma'] = np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / len(x)

    # Window label, phase, assembly
    features['label'] = Counter(window_df['label'].dropna()).most_common(1)[0][0] \
                        if not window_df['label'].dropna().empty else 'Unknown'
    features['phase_id'] = Counter(window_df['phase_id'].dropna()).most_common(1)[0][0] \
                           if not window_df['phase_id'].dropna().empty else None
    features['assembly_id'] = Counter(window_df['assembly_id'].dropna()).most_common(1)[0][0] \
                              if not window_df['assembly_id'].dropna().empty else None

    return features

# -----------------------
# Visualization functions
# -----------------------
def visualize_features(
    features_df: pd.DataFrame,
    features_to_plot: list[str] | None = None,
    max_activities: int = 6,
    results_folder: Path | None = None
) -> None:
    """
    Plot boxplots of selected features per activity label.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing extracted features with a 'label' column.
    features_to_plot : list[str] | None
        List of feature column names to plot. If None, selects a default subset of numeric columns.
    max_activities : int
        Maximum number of distinct activity labels to include in the plots.
    results_folder : Path | None
        Base folder where plots will be saved. A subfolder 'feature_visualizations'
        will be created inside it.

    Returns
    -------
    None
    """
    if features_to_plot is None:
        numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        features_to_plot = numeric_cols[2:8]

    labels = features_df["label"].unique()[:max_activities]
    df_plot = features_df[features_df["label"].isin(labels)]

    # Create visualization subfolder
    if results_folder:
        plot_folder = results_folder / "feature_visualizations"
        plot_folder.mkdir(exist_ok=True, parents=True)
    else:
        plot_folder = None

    # Generate boxplots
    for feature in features_to_plot:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="label", y=feature, data=df_plot)
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {feature} per activity")
        plt.tight_layout()

        if plot_folder:
            outfile = plot_folder / f"{feature}_boxplot.png"
            plt.savefig(outfile, dpi=300)
            plt.close()
        else:
            plt.show()


def scatter_features(
    features_df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    max_activities: int = 6,
    results_folder: Path | None = None
) -> None:
    """
    Plot scatter plot of two features colored by activity label.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing extracted features with a 'label' column.
    feature_x : str
        Column name for the x-axis.
    feature_y : str
        Column name for the y-axis.
    max_activities : int
        Maximum number of distinct activity labels to include in the plot.
    results_folder : Path | None
        Base folder where plots will be saved. A subfolder 'feature_visualizations'
        will be created inside it.

    Returns
    -------
    None
    """
    labels = features_df["label"].unique()[:max_activities]
    df_plot = features_df[features_df["label"].isin(labels)]

    # Create visualization subfolder
    if results_folder:
        plot_folder = results_folder / "feature_visualizations"
        plot_folder.mkdir(exist_ok=True, parents=True)
    else:
        plot_folder = None

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=feature_x,
        y=feature_y,
        hue="label",
        data=df_plot,
        palette="tab10",
        alpha=0.7
    )
    plt.title(f"Scatter plot: {feature_x} vs {feature_y}")
    plt.tight_layout()

    if plot_folder:
        outfile = plot_folder / f"{feature_x}_vs_{feature_y}_scatter.png"
        plt.savefig(outfile, dpi=300)
        plt.close()
    else:
        plt.show()
