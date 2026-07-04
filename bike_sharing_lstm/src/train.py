"""Training pipeline for the bike sharing LSTM model."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .config import (
    BATCH_SIZE,
    EPOCHS,
    H5_MODEL_PATH,
    HISTORY_PATH,
    LOOKBACK,
    MODEL_PATH,
    PATIENCE,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    TRAINING_PLOT_PATH,
    TRAIN_RATIO,
    ensure_project_dirs,
)
from .data_loader import load_train_test_data
from .feature_engineering import (
    fit_transform_features,
    inverse_transform_target,
    save_feature_artifacts,
)
from .model import build_lstm_model
from .sequence_generator import create_sequences


logger = logging.getLogger(__name__)


class EpochMetricsCallback(Callback):
    """Compute business-scale metrics on train/validation data after each epoch."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        artifacts,
    ) -> None:
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.artifacts = artifacts
        self.epoch_metrics: list[dict[str, float]] = []

    def _compute_metrics(self, y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> tuple[float, float]:
        """Compute RMSE and MAE after restoring the original cnt scale."""
        y_true = inverse_transform_target(y_true_scaled, self.artifacts)
        y_pred = inverse_transform_target(y_pred_scaled, self.artifacts)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        return rmse, mae

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Evaluate and log metrics at the end of each epoch."""
        logs = logs or {}

        train_pred_scaled = self.model.predict(self.x_train, verbose=0).reshape(-1)
        val_pred_scaled = self.model.predict(self.x_val, verbose=0).reshape(-1)

        train_rmse, train_mae = self._compute_metrics(self.y_train, train_pred_scaled)
        val_rmse, val_mae = self._compute_metrics(self.y_val, val_pred_scaled)

        metrics = {
            "epoch": float(epoch + 1),
            "loss": float(logs.get("loss", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_mae": train_mae,
            "val_mae": val_mae,
        }
        self.epoch_metrics.append(metrics)

        logger.info(
            (
                "Epoch %03d | train_loss=%.6f | val_loss=%.6f | "
                "train_rmse=%.4f | val_rmse=%.4f | train_mae=%.4f | val_mae=%.4f"
            ),
            epoch + 1,
            metrics["loss"],
            metrics["val_loss"],
            train_rmse,
            val_rmse,
            train_mae,
            val_mae,
        )


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_runtime() -> str:
    """Detect GPU availability and enable memory growth when possible."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("TensorFlow runtime: CPU")
        return "CPU"

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            logger.debug("Could not enable memory growth for GPU: %s", gpu)

    runtime_desc = f"GPU x{len(gpus)}"
    logger.info("TensorFlow runtime: %s", runtime_desc)
    return runtime_desc


def _plot_training_history(history_df: pd.DataFrame, plot_path: Path) -> None:
    """Save a line plot of training and validation loss."""
    plt.figure(figsize=(10, 5))
    plt.plot(history_df.index + 1, history_df["loss"], label="train_loss")
    plt.plot(history_df.index + 1, history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Saved training curve to %s", plot_path)


def _save_trained_models(model: tf.keras.Model) -> None:
    """Persist the trained model in modern and compatibility formats."""
    model.save(MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

    model.save(H5_MODEL_PATH)
    logger.info("Saved compatibility model to %s", H5_MODEL_PATH)


def _build_history_dataframe(
    history: tf.keras.callbacks.History,
    metrics_callback: EpochMetricsCallback,
) -> pd.DataFrame:
    """Combine Keras history with business-scale metrics for export."""
    history_df = pd.DataFrame(history.history)
    metrics_df = pd.DataFrame(metrics_callback.epoch_metrics)

    if "epoch" not in history_df.columns:
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

    metric_columns = ["train_rmse", "val_rmse", "train_mae", "val_mae"]
    for column in metric_columns:
        if column in metrics_df.columns:
            history_df[column] = metrics_df[column].values

    return history_df


def train_model() -> Dict[str, float]:
    """Run the end-to-end model training workflow."""
    ensure_project_dirs()
    set_random_seed()
    configure_runtime()

    train_df, test_df = load_train_test_data()
    train_features, _, target_array, artifacts = fit_transform_features(train_df, test_df)

    split_index = int(len(train_features) * TRAIN_RATIO)
    split_index = max(split_index, LOOKBACK + 1)
    if split_index >= len(train_features):
        raise ValueError("Validation split is empty. Adjust TRAIN_RATIO or LOOKBACK.")

    x_train, y_train = create_sequences(
        feature_frame=train_features,
        target_array=target_array,
        lookback=LOOKBACK,
        start_index=LOOKBACK,
        end_index=split_index,
    )
    x_val, y_val = create_sequences(
        feature_frame=train_features,
        target_array=target_array,
        lookback=LOOKBACK,
        start_index=split_index,
        end_index=len(train_features),
    )

    logger.info(
        "Sequence shapes: x_train=%s, y_train=%s, x_val=%s, y_val=%s",
        x_train.shape,
        y_train.shape,
        x_val.shape,
        y_val.shape,
    )

    model = build_lstm_model((LOOKBACK, x_train.shape[-1]))
    metrics_callback = EpochMetricsCallback(x_train, y_train, x_val, y_val, artifacts)
    callbacks = [
        metrics_callback,
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, PATIENCE // 2), min_lr=1e-5),
        ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    history_df = _build_history_dataframe(history, metrics_callback)
    history_df.to_csv(HISTORY_PATH, index=False)
    logger.info("Saved training history to %s", HISTORY_PATH)
    _plot_training_history(history_df, TRAINING_PLOT_PATH)

    _save_trained_models(model)
    save_feature_artifacts(artifacts, PREPROCESSOR_PATH)

    best_row = history_df.loc[history_df["val_loss"].idxmin()]
    best_epoch = int(best_row["epoch"])
    logger.info(
        (
            "Best epoch %03d | train_loss=%.6f | val_loss=%.6f | "
            "train_rmse=%.4f | val_rmse=%.4f | train_mae=%.4f | val_mae=%.4f"
        ),
        best_epoch,
        float(best_row["loss"]),
        float(best_row["val_loss"]),
        float(best_row["train_rmse"]),
        float(best_row["val_rmse"]),
        float(best_row["train_mae"]),
        float(best_row["val_mae"]),
    )
    return {
        "best_epoch": float(best_epoch),
        "val_loss": float(best_row["val_loss"]),
        "val_rmse": float(best_row["val_rmse"]),
        "val_mae": float(best_row["val_mae"]),
    }
