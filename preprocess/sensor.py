"""Utility functions for processing raw sensor time-series."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["resample_signal", "normalize_signal", "window_signal"]


def resample_signal(df: pd.DataFrame, time_col: str, freq: str, method: str = "mean") -> pd.DataFrame:
    """Resample a time-indexed DataFrame to a new frequency.

    Parameters
    ----------
    df : DataFrame
        Input data containing a time column.
    time_col : str
        Name of the column representing timestamps.
    freq : str
        Target pandas frequency string (e.g., ``'1S'`` for 1-second).
    method : str, optional
        Aggregation method: ``'mean'`` (default) or ``'nearest'``.

    Returns
    -------
    DataFrame
        Resampled DataFrame with the same columns as ``df``.
    """
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    if method == "nearest":
        resampled = df.resample(freq).nearest()
    else:
        resampled = df.resample(freq).mean()
    return resampled.reset_index()


def normalize_signal(arr: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize a sensor signal.

    Parameters
    ----------
    arr : ndarray
        Input array of shape (T, F) or (T,).
    method : str, optional
        ``'zscore'`` (default) for zero-mean/unit-var or ``'minmax'``.

    Returns
    -------
    ndarray
        Normalized array with the same shape as ``arr``.
    """
    arr = np.asarray(arr, dtype=float)
    if method == "minmax":
        min_ = arr.min(axis=0, keepdims=True)
        max_ = arr.max(axis=0, keepdims=True)
        denom = max_ - min_
        denom[denom == 0] = 1.0
        return (arr - min_) / denom
    else:  # z-score
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (arr - mean) / std


def window_signal(arr: np.ndarray, window_size: int, step: int | None = None) -> np.ndarray:
    """Create overlapping windows from a time-series array.

    Parameters
    ----------
    arr : ndarray
        Array of shape (T, F) or (T,).
    window_size : int
        Number of time steps per window.
    step : int, optional
        Step size between windows. Defaults to ``window_size`` (non-overlapping).

    Returns
    -------
    ndarray
        Array of shape (N, window_size, F) containing windows. ``N`` may be zero
        if the sequence is shorter than ``window_size``.
    """
    arr = np.asarray(arr)
    if step is None:
        step = window_size
    n_steps = arr.shape[0]
    windows = [arr[start:start + window_size] for start in range(0, n_steps - window_size + 1, step)]
    if not windows:
        return np.empty((0, window_size) + arr.shape[1:])
    return np.stack(windows)
