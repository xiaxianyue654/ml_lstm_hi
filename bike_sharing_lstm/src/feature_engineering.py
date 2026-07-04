"""Feature engineering and preprocessing utilities."""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logger = logging.getLogger(__name__)

CATEGORICAL_COLUMNS = ["season", "weathersit"]
CONTINUOUS_COLUMNS = ["temp", "atemp", "hum", "windspeed"]
PASSTHROUGH_COLUMNS = ["yr", "holiday", "workingday"]
CYCLIC_COLUMNS = ["hr_sin", "hr_cos", "weekday_sin", "weekday_cos", "mnth_sin", "mnth_cos"]


@dataclass
class FeatureArtifacts:
    """Saved preprocessing objects required for inference."""

    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    category_encoder: OneHotEncoder
    feature_columns: List[str]
    continuous_columns: List[str]
    categorical_columns: List[str]
    passthrough_columns: List[str]
    cyclic_columns: List[str]


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create a compatible OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def add_cyclic_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings for hour, weekday, and month."""
    frame = dataframe.copy()
    frame["hr_sin"] = np.sin(2 * math.pi * frame["hr"] / 24.0)
    frame["hr_cos"] = np.cos(2 * math.pi * frame["hr"] / 24.0)
    frame["weekday_sin"] = np.sin(2 * math.pi * frame["weekday"] / 7.0)
    frame["weekday_cos"] = np.cos(2 * math.pi * frame["weekday"] / 7.0)
    frame["mnth_sin"] = np.sin(2 * math.pi * frame["mnth"] / 12.0)
    frame["mnth_cos"] = np.cos(2 * math.pi * frame["mnth"] / 12.0)
    return frame


def _assemble_feature_frame(
    dataframe: pd.DataFrame,
    artifacts: FeatureArtifacts,
) -> pd.DataFrame:
    """Transform a raw dataframe into the model input feature frame."""
    frame = add_cyclic_time_features(dataframe)

    continuous_array = artifacts.feature_scaler.transform(frame[artifacts.continuous_columns])
    continuous_df = pd.DataFrame(
        continuous_array,
        columns=artifacts.continuous_columns,
        index=frame.index,
    )

    categorical_array = artifacts.category_encoder.transform(frame[artifacts.categorical_columns])
    categorical_df = pd.DataFrame(
        categorical_array,
        columns=artifacts.category_encoder.get_feature_names_out(artifacts.categorical_columns),
        index=frame.index,
    )

    passthrough_df = frame[artifacts.passthrough_columns + artifacts.cyclic_columns].copy()
    passthrough_df = passthrough_df.astype(np.float32)

    feature_frame = pd.concat([passthrough_df, continuous_df, categorical_df], axis=1)
    feature_frame = feature_frame.reindex(columns=artifacts.feature_columns)
    feature_frame = feature_frame.astype(np.float32)
    return feature_frame


def fit_transform_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, FeatureArtifacts]:
    """Fit preprocessing on the training set and transform both splits."""
    train_frame = add_cyclic_time_features(train_df)
    test_frame = add_cyclic_time_features(test_df)

    feature_scaler = StandardScaler()
    train_continuous = feature_scaler.fit_transform(train_frame[CONTINUOUS_COLUMNS])
    test_continuous = feature_scaler.transform(test_frame[CONTINUOUS_COLUMNS])

    category_encoder = _build_one_hot_encoder()
    train_categorical = category_encoder.fit_transform(train_frame[CATEGORICAL_COLUMNS])
    test_categorical = category_encoder.transform(test_frame[CATEGORICAL_COLUMNS])
    category_feature_names = list(category_encoder.get_feature_names_out(CATEGORICAL_COLUMNS))

    target_scaler = StandardScaler()
    target_array = target_scaler.fit_transform(train_frame[["cnt"]]).astype(np.float32).ravel()

    train_passthrough = train_frame[PASSTHROUGH_COLUMNS + CYCLIC_COLUMNS].astype(np.float32).reset_index(drop=True)
    test_passthrough = test_frame[PASSTHROUGH_COLUMNS + CYCLIC_COLUMNS].astype(np.float32).reset_index(drop=True)

    train_continuous_df = pd.DataFrame(train_continuous, columns=CONTINUOUS_COLUMNS)
    test_continuous_df = pd.DataFrame(test_continuous, columns=CONTINUOUS_COLUMNS)
    train_categorical_df = pd.DataFrame(train_categorical, columns=category_feature_names)
    test_categorical_df = pd.DataFrame(test_categorical, columns=category_feature_names)

    train_features = pd.concat([train_passthrough, train_continuous_df, train_categorical_df], axis=1)
    test_features = pd.concat([test_passthrough, test_continuous_df, test_categorical_df], axis=1)

    feature_columns = train_features.columns.tolist()
    artifacts = FeatureArtifacts(
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        category_encoder=category_encoder,
        feature_columns=feature_columns,
        continuous_columns=CONTINUOUS_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        passthrough_columns=PASSTHROUGH_COLUMNS,
        cyclic_columns=CYCLIC_COLUMNS,
    )

    logger.info(
        "Feature engineering complete: train_features=%s, test_features=%s",
        train_features.shape,
        test_features.shape,
    )
    return train_features.astype(np.float32), test_features.astype(np.float32), target_array, artifacts


def transform_features(
    dataframe: pd.DataFrame,
    artifacts: FeatureArtifacts,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Transform a dataframe using previously fitted preprocessing artifacts."""
    feature_frame = _assemble_feature_frame(dataframe, artifacts)
    target_array: Optional[np.ndarray] = None

    if "cnt" in dataframe.columns:
        target_array = artifacts.target_scaler.transform(dataframe[["cnt"]]).astype(np.float32).ravel()

    return feature_frame, target_array


def inverse_transform_target(values: np.ndarray, artifacts: FeatureArtifacts) -> np.ndarray:
    """Convert standardized predictions back to the original cnt scale."""
    restored = artifacts.target_scaler.inverse_transform(values.reshape(-1, 1)).ravel()
    return restored


def save_feature_artifacts(artifacts: FeatureArtifacts, filepath: Path) -> None:
    """Persist preprocessing artifacts to disk."""
    with filepath.open("wb") as file_obj:
        pickle.dump(artifacts, file_obj)
    logger.info("Saved preprocessing artifacts to %s", filepath)


def load_feature_artifacts(filepath: Path) -> FeatureArtifacts:
    """Load preprocessing artifacts from disk."""
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {filepath}")

    with filepath.open("rb") as file_obj:
        artifacts = pickle.load(file_obj)

    if not isinstance(artifacts, FeatureArtifacts):
        raise TypeError(f"Unexpected preprocessor object stored in {filepath}")

    return artifacts

