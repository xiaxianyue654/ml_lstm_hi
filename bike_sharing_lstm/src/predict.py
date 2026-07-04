"""Prediction pipeline and submission generation."""

from __future__ import annotations

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import (
    H5_MODEL_PATH,
    LOOKBACK,
    MODEL_PATH,
    OUTPUT_DIR,
    PREDICTION_PLOT_PATH,
    PREDICTION_PLOT_TAIL,
    PREPROCESSOR_PATH,
    SUBMISSION_PATH,
    ensure_project_dirs,
)
from .data_loader import load_train_test_data
from .feature_engineering import inverse_transform_target, load_feature_artifacts, transform_features
from .sequence_generator import create_test_sequences


logger = logging.getLogger(__name__)


def _build_timestamps(dataframe: pd.DataFrame) -> pd.Series:
    """Combine date and hour into an hourly timestamp."""
    return dataframe["dteday"] + pd.to_timedelta(dataframe["hr"], unit="h")


def _plot_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, predictions: np.ndarray) -> None:
    """Save a forecast visualization with recent train values and test predictions."""
    train_tail = train_df.tail(PREDICTION_PLOT_TAIL).copy()
    train_tail["timestamp"] = _build_timestamps(train_tail)

    forecast_df = test_df.copy()
    forecast_df["timestamp"] = _build_timestamps(forecast_df)
    forecast_df["predicted_cnt"] = predictions

    plt.figure(figsize=(12, 5))
    plt.plot(train_tail["timestamp"], train_tail["cnt"], label="train_cnt_recent")
    plt.plot(forecast_df["timestamp"], forecast_df["predicted_cnt"], label="test_forecast")
    plt.xlabel("Time")
    plt.ylabel("Bike Count")
    plt.title("Bike Sharing Forecast Preview")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=150)
    plt.close()
    logger.info("Saved prediction visualization to %s", PREDICTION_PLOT_PATH)


def load_trained_model() -> tf.keras.Model:
    """Load the trained model, preferring .keras and falling back to legacy .h5."""
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Loaded model from %s", MODEL_PATH)
        return model

    if H5_MODEL_PATH.exists():
        model = tf.keras.models.load_model(
            H5_MODEL_PATH,
            custom_objects={
                "mse": tf.keras.losses.MeanSquaredError(),
                "mae": tf.keras.metrics.MeanAbsoluteError(),
            },
            compile=False,
        )
        logger.info("Loaded legacy H5 model from %s", H5_MODEL_PATH)
        return model

    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH} or {H5_MODEL_PATH}")


def generate_submission() -> pd.DataFrame:
    """Load the trained model, run inference, and create submission.csv."""
    ensure_project_dirs()

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    train_df, test_df = load_train_test_data()
    artifacts = load_feature_artifacts(PREPROCESSOR_PATH)

    train_features, _ = transform_features(train_df, artifacts)
    test_features, _ = transform_features(test_df, artifacts)

    x_test = create_test_sequences(
        history_features=train_features,
        future_features=test_features,
        lookback=LOOKBACK,
    )
    logger.info("Prediction input shape: %s", x_test.shape)

    model = load_trained_model()
    pred_scaled = model.predict(x_test, verbose=0).reshape(-1)
    predictions = inverse_transform_target(pred_scaled, artifacts)
    predictions = np.clip(predictions, 0.0, None)

    submission = pd.DataFrame({"ID": test_df["ID"].to_numpy(), "cnt": predictions})
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Saved submission to %s", SUBMISSION_PATH)

    _plot_predictions(train_df, test_df, predictions)
    return submission
