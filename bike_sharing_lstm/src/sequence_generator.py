"""Sequence generation helpers for LSTM training and inference."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def create_sequences(
    feature_frame: pd.DataFrame,
    target_array: Optional[np.ndarray],
    lookback: int,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Create sliding-window sequences for supervised learning."""
    feature_values = feature_frame.to_numpy(dtype=np.float32)
    actual_start = lookback if start_index is None else start_index
    actual_end = len(feature_values) if end_index is None else end_index

    sequences = []
    labels = []

    for target_idx in range(actual_start, actual_end):
        sequences.append(feature_values[target_idx - lookback : target_idx])
        if target_array is not None:
            labels.append(target_array[target_idx])

    x_array = np.asarray(sequences, dtype=np.float32)
    if target_array is None:
        return x_array, None

    y_array = np.asarray(labels, dtype=np.float32)
    return x_array, y_array


def create_test_sequences(
    history_features: pd.DataFrame,
    future_features: pd.DataFrame,
    lookback: int,
) -> np.ndarray:
    """Create test windows using train history followed by test features."""
    combined_features = pd.concat([history_features, future_features], axis=0, ignore_index=True)
    x_test, _ = create_sequences(
        feature_frame=combined_features,
        target_array=None,
        lookback=lookback,
        start_index=len(history_features),
        end_index=len(combined_features),
    )
    return x_test

