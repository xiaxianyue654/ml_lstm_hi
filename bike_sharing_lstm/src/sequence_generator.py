"""Sequence generation helpers for autoregressive LSTM training and inference."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _combine_feature_and_target_windows(
    feature_window: np.ndarray,
    target_window: np.ndarray,
) -> np.ndarray:
    """Concatenate exogenous features with the target-history input channel."""
    target_channel = target_window.reshape(-1, 1)
    return np.concatenate([feature_window, target_channel], axis=1).astype(np.float32)


def create_sequences(
    feature_frame: pd.DataFrame,
    target_array: np.ndarray,
    lookback: int,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create one-step-ahead autoregressive sequences for supervised learning."""
    feature_values = feature_frame.to_numpy(dtype=np.float32)
    actual_start = lookback if start_index is None else start_index
    actual_end = len(feature_values) if end_index is None else end_index

    sequences = []
    labels = []

    for target_idx in range(actual_start, actual_end):
        feature_window = feature_values[target_idx - lookback : target_idx]
        target_window = target_array[target_idx - lookback : target_idx]
        sequences.append(_combine_feature_and_target_windows(feature_window, target_window))
        labels.append(target_array[target_idx])

    x_array = np.asarray(sequences, dtype=np.float32)
    y_array = np.asarray(labels, dtype=np.float32)
    return x_array, y_array


def create_inference_sequence(
    feature_history: pd.DataFrame,
    target_history: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Build one inference input tensor from the latest feature and target history."""
    feature_window = feature_history.tail(lookback).to_numpy(dtype=np.float32)
    target_window = np.asarray(target_history[-lookback:], dtype=np.float32)
    sequence = _combine_feature_and_target_windows(feature_window, target_window)
    return sequence[np.newaxis, :, :]
