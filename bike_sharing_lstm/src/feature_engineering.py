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

CONTINUOUS_COLUMNS = ["temp", "atemp", "hum", "windspeed", "temp_hum", "temp_windspeed"]
CATEGORICAL_COLUMNS = ["season", "weathersit", "part_of_day"]
PASSTHROUGH_COLUMNS = [
    "yr",
    "holiday",
    "workingday",
    "is_weekend",
    "is_rush_hour",
    "daytime_flag",
    "rush_working",
    "bad_weather_flag",
    "rush_bad_weather",
    "weekend_daytime",
]
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


def add_time_context_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add stable time-context features derived from known timestamps."""
    frame = dataframe.copy()
    frame["is_weekend"] = frame["weekday"].isin([0, 6]).astype(np.int8)
    frame["is_rush_hour"] = frame["hr"].isin([7, 8, 17, 18]).astype(np.int8)
    frame["part_of_day"] = np.select(
        [frame["hr"] < 6, frame["hr"] < 12, frame["hr"] < 18],
        [0, 1, 2],
        default=3,
    ).astype(np.int8)
    frame["daytime_flag"] = frame["part_of_day"].isin([1, 2]).astype(np.int8)
    return frame


def add_interaction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add low-risk interaction features that preserve the current LSTM pipeline."""
    frame = dataframe.copy()
    frame["temp_hum"] = frame["temp"] * frame["hum"]
    frame["temp_windspeed"] = frame["temp"] * frame["windspeed"]
    frame["rush_working"] = (frame["is_rush_hour"] * frame["workingday"]).astype(np.int8)
    frame["bad_weather_flag"] = (frame["weathersit"] >= 3).astype(np.int8)
    frame["rush_bad_weather"] = (frame["is_rush_hour"] * frame["bad_weather_flag"]).astype(np.int8)
    frame["weekend_daytime"] = (frame["is_weekend"] * frame["daytime_flag"]).astype(np.int8)
    return frame


def build_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build stable exogenous features only."""
    frame = dataframe.copy()
    frame = add_cyclic_time_features(frame)
    frame = add_time_context_features(frame)
    frame = add_interaction_features(frame)
    return frame


def transform_target_series(target_series: pd.Series | np.ndarray) -> np.ndarray:
    """Apply the log-domain target transform before scaling."""
    target_array = np.asarray(target_series, dtype=np.float32).reshape(-1, 1)
    return np.log1p(target_array)


def _assemble_feature_frame(
    dataframe: pd.DataFrame,
    artifacts: FeatureArtifacts,
) -> pd.DataFrame:
    """Transform a raw dataframe into the model input feature frame."""
    frame = build_feature_frame(dataframe)

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

    passthrough_df = frame[artifacts.passthrough_columns + artifacts.cyclic_columns].copy().astype(np.float32)
    feature_frame = pd.concat([passthrough_df, continuous_df, categorical_df], axis=1)
    feature_frame = feature_frame.reindex(columns=artifacts.feature_columns)
    return feature_frame.astype(np.float32)


def fit_transform_features(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, FeatureArtifacts]:
    """Fit preprocessing on the training set and return transformed features."""
    train_frame = build_feature_frame(train_df)

    feature_scaler = StandardScaler()
    train_continuous = feature_scaler.fit_transform(train_frame[CONTINUOUS_COLUMNS])

    category_encoder = _build_one_hot_encoder()
    train_categorical = category_encoder.fit_transform(train_frame[CATEGORICAL_COLUMNS])
    category_feature_names = list(category_encoder.get_feature_names_out(CATEGORICAL_COLUMNS))

    target_scaler = StandardScaler()
    target_log = transform_target_series(train_df["cnt"])
    target_array = target_scaler.fit_transform(target_log).astype(np.float32).ravel()

    train_passthrough = train_frame[PASSTHROUGH_COLUMNS + CYCLIC_COLUMNS].astype(np.float32).reset_index(drop=True)
    train_continuous_df = pd.DataFrame(train_continuous, columns=CONTINUOUS_COLUMNS)
    train_categorical_df = pd.DataFrame(train_categorical, columns=category_feature_names)
    train_features = pd.concat([train_passthrough, train_continuous_df, train_categorical_df], axis=1)

    artifacts = FeatureArtifacts(
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        category_encoder=category_encoder,
        feature_columns=train_features.columns.tolist(),
        continuous_columns=CONTINUOUS_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        passthrough_columns=PASSTHROUGH_COLUMNS,
        cyclic_columns=CYCLIC_COLUMNS,
    )

    logger.info("Feature engineering complete: train_features=%s", train_features.shape)
    logger.info("Feature columns (%d): %s...", len(train_features.columns), train_features.columns[:10].tolist())
    logger.info("Contains lag features: %s", any("lag" in column for column in train_features.columns))
    logger.info("Contains rolling features: %s", any("roll" in column for column in train_features.columns))
    logger.info("Contains time context: %s", any("weekend" in column for column in train_features.columns))
    logger.info("Contains interaction: %s", any("temp_hum" in column for column in train_features.columns))
    logger.info("Target transform: log1p(cnt) + StandardScaler")

    return train_features.astype(np.float32), target_array, artifacts


def transform_features(
    dataframe: pd.DataFrame,
    artifacts: FeatureArtifacts,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Transform a dataframe using saved preprocessing artifacts."""
    feature_frame = _assemble_feature_frame(dataframe, artifacts)
    target_array: Optional[np.ndarray] = None

    if "cnt" in dataframe.columns:
        target_log = transform_target_series(dataframe["cnt"])
        target_array = artifacts.target_scaler.transform(target_log).astype(np.float32).ravel()

    return feature_frame, target_array


def inverse_transform_target(values: np.ndarray, artifacts: FeatureArtifacts) -> np.ndarray:
    """Convert scaled log-domain predictions back to the original cnt scale."""
    log_values = artifacts.target_scaler.inverse_transform(values.reshape(-1, 1))
    restored = np.expm1(log_values)
    return np.clip(restored, 0.0, None).ravel()


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
