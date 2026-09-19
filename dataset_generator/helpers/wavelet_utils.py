"""
Wavelet transform utilities for signal processing.

This module provides functions for computing continuous wavelet transforms (CWT)
on sensor time series data, enabling time-frequency analysis for activity recognition.

The implementation uses per-timestep wavelet coefficients, so each timestep gets
its own set of wavelet features (unlike windowed DWT which would produce identical
features for all timesteps in a window).
"""

from __future__ import annotations
import numpy as np
import pywt
from typing import List

def compute_cwt_features(
    signal: np.ndarray,
    wavelet: str = "morl",
    scales: List[int] | None = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute Continuous Wavelet Transform (CWT) features for a 1D signal using PyWavelets.
    
    Each timestep gets its own set of wavelet coefficients, preserving temporal
    resolution. This differs from DWT which would downsample and produce identical
    values across timesteps.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values (one sensor channel over time).
        Shape: (time_steps,)
    
    wavelet : str
        Wavelet family: 'morl' (Morlet), 'mexh' (Mexican hat), etc.
        Must be a continuous wavelet for use with pywt.cwt().
        Default: 'morl' (generally good for time-frequency analysis)
    
    scales : List[int] | None
        CWT scales corresponding to frequencies. Lower scale = higher frequency.
        If None, uses [1, 2, 4, 8].
        Example: scales=[1, 2, 4, 8] gives 4 frequency bands per timestep.
    
    normalize : bool
        If True, normalize CWT magnitude to [0, 1] per scale band for stability.
    
    Returns
    -------
    np.ndarray
        CWT coefficients. Shape: (len(scales), time_steps)
        Each row is one frequency band; each column is one timestep.
        Each timestep has distinct wavelet features from other timesteps.
    
    Example
    -------
    >>> signal = np.array([1, 2, 3, 2, 1, 0, -1, -2, -1, 0])  # 10 timesteps
    >>> scales = [1, 2, 4]  # 3 frequency bands
    >>> cwt_coeff = compute_cwt_features(signal, scales=scales)
    >>> cwt_coeff.shape
    (3, 10)  # 3 scales, 10 timesteps - each timestep has 3 wavelet features
    """
    if scales is None:
        scales = [1, 2, 4, 8]
    
    # Convert scales to numpy array
    scales = np.array(scales, dtype=np.float64)
    
    # Convert signal to numpy array for compatibility
    signal = np.asarray(signal, dtype=np.float64)
    
    # Compute CWT using PyWavelets
    # Uses method='fft' for efficiency
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, method='fft')
    
    # Take absolute value (magnitude)
    coefficients = np.abs(coefficients)
    
    if normalize:
        # Normalize each scale band independently to [0, 1]
        for i in range(coefficients.shape[0]):
            min_val = coefficients[i].min()
            max_val = coefficients[i].max()
            if max_val > min_val:
                coefficients[i] = (coefficients[i] - min_val) / (max_val - min_val)
    
    return coefficients


def add_wavelet_features_to_window(
    window_data: np.ndarray,
    wavelet: str = "morl",
    scales: List[int] | None = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Add wavelet features to raw sensor window data.
    
    For each sensor channel, compute CWT features and concatenate with raw data.
    Each timestep retains distinct wavelet features.
    
    Parameters
    ----------
    window_data : np.ndarray
        Raw sensor data for one window. 
        Shape: (time_steps, n_sensors) e.g., (128, 6) for 128 timesteps, 6 sensors
    
    wavelet : str
        Wavelet family to use (e.g., 'morl', 'mexh', 'db4').
    
    scales : List[int] | None
        CWT scales. If None, uses [1, 2, 4, 8].
    
    normalize : bool
        Whether to normalize wavelet coefficients.
    
    Returns
    -------
    np.ndarray
        Window data with wavelet features appended.
        Shape: (time_steps, n_sensors + n_sensors * len(scales))
        
        Example for (128, 6) input with 4 scales:
        - Original: 6 sensor channels
        - Added: 6 sensors × 4 scales = 24 wavelet channels
        - Output: (128, 30)
        
        Each timestep now has 30 features: 6 raw + 24 wavelet.
        Each timestep has DIFFERENT wavelet values (unlike repeated DWT values).
    
    Example
    -------
    >>> window = np.random.randn(128, 6)  # 128 timesteps, 6 sensors
    >>> scales = [1, 2, 4]
    >>> window_with_wavelets = add_wavelet_features_to_window(
    ...     window, scales=scales
    ... )
    >>> window_with_wavelets.shape
    (128, 24)  # 6 sensors + 6 * 3 scales = 24 features per timestep
    """
    if scales is None:
        scales = [1, 2, 4, 8]
    
    _ , n_sensors = window_data.shape
    wavelet_features_list = []
    
    # Compute CWT for each sensor channel
    for sensor_idx in range(n_sensors):
        signal = window_data[:, sensor_idx]
        cwt_coeff = compute_cwt_features(signal, wavelet=wavelet, scales=scales, normalize=normalize)
        # cwt_coeff shape: (len(scales), time_steps)
        # Transpose to (time_steps, len(scales)) to match window_data orientation
        wavelet_features_list.append(cwt_coeff.T)
    
    # Concatenate all wavelet features: (time_steps, n_sensors * len(scales))
    all_wavelet_features = np.hstack(wavelet_features_list)
    
    # Concatenate raw data with wavelet features
    combined = np.hstack([window_data, all_wavelet_features])
    
    return combined
