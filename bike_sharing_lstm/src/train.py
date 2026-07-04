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
    EARLY_STOP_MIN_DELTA,
    EPOCHS,
    FINAL_HISTORY_PATH,
    H5_MODEL_PATH,
    HISTORY_PATH,
    LOOKBACK,
    LR_REDUCE_PATIENCE,
    MODEL_PATH,
    N_SPLITS,
    PATIENCE,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    TRAINING_PLOT_PATH,
    VALIDATION_MODE,
    ensure_project_dirs,
)
from .data_loader import load_train_test_data
from .feature_engineering import (
    fit_transform_features,
    inverse_transform_target,
    save_feature_artifacts,
    transform_features,
)
from .model import build_lstm_model
from .sequence_generator import create_sequences


logger = logging.getLogger(__name__)


class EpochMetricsCallback(Callback):
    """Compute business-scale metrics on train/validation data after each epoch."""

    def __init__(
        self,
        fold_id: int | str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        artifacts,
    ) -> None:
        super().__init__()
        self.fold_id = fold_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.artifacts = artifacts
        self.epoch_metrics: list[dict[str, float | int | str]] = []

    def _compute_metrics(self, y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> tuple[float, float]:
        """Compute RMSE and MAE on the original cnt scale."""
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

        logs["train_rmse"] = train_rmse
        logs["val_rmse"] = val_rmse
        logs["train_mae"] = train_mae
        logs["val_mae"] = val_mae

        metrics = {
            "fold": self.fold_id,
            "epoch": int(epoch + 1),
            "loss": float(logs.get("loss", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "learning_rate": float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
        }
        self.epoch_metrics.append(metrics)

        logger.info(
            (
                "Fold %s | Epoch %03d | train_loss=%.6f | val_loss=%.6f | "
                "train_rmse=%.4f | val_rmse=%.4f | train_mae=%.4f | val_mae=%.4f"
            ),
            self.fold_id,
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
    if history_df.empty:
        return

    plot_df = history_df.copy()
    if "fold" in plot_df.columns:
        plot_df = plot_df[plot_df["fold"] != "final"]

    plt.figure(figsize=(10, 5))
    for fold_id, fold_df in plot_df.groupby("fold"):
        plt.plot(fold_df["epoch"], fold_df["val_loss"], label=f"fold_{fold_id}_val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Walk-Forward Validation Loss")
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


def _create_walk_forward_folds(
    total_rows: int,
    n_splits: int,
) -> list[tuple[int, int]]:
    """Create forward-only validation windows covering the latter portion of the series."""
    validation_start = max(LOOKBACK * 2, int(total_rows * 0.5))
    fold_edges = np.linspace(validation_start, total_rows, n_splits + 1, dtype=int)

    folds: list[tuple[int, int]] = []
    for fold_id in range(n_splits):
        start_idx = int(fold_edges[fold_id])
        end_idx = int(fold_edges[fold_id + 1])
        if end_idx <= start_idx:
            continue
        folds.append((start_idx, end_idx))

    if not folds:
        raise ValueError("Failed to create walk-forward folds. Check dataset size and split settings.")

    return folds


def _prepare_fold_datasets(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
) -> tuple[object, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare per-fold transformed data and autoregressive sequence blocks."""
    train_features, train_target, artifacts = fit_transform_features(train_raw)
    combined_raw = pd.concat([train_raw, val_raw], ignore_index=True)
    combined_features, combined_target = transform_features(combined_raw, artifacts)
    if combined_target is None:
        raise ValueError("Validation target array is required for training.")

    train_feature_count = len(train_features)
    x_train, y_train = create_sequences(train_features, train_target, LOOKBACK)
    x_val, y_val = create_sequences(
        combined_features,
        combined_target,
        LOOKBACK,
        start_index=train_feature_count,
        end_index=len(combined_features),
    )

    return artifacts, train_features, train_target, x_train, y_train, x_val, y_val


def _train_single_fold(
    fold_id: int,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train one walk-forward fold and return the full epoch history plus fold summary."""
    artifacts, train_features, train_target, x_train, y_train, x_val, y_val = _prepare_fold_datasets(train_raw, val_raw)

    logger.info(
        "Fold %d data | train_rows=%d | val_rows=%d | feature_dim=%d | x_train=%s | x_val=%s",
        fold_id,
        len(train_raw),
        len(val_raw),
        train_features.shape[1],
        x_train.shape,
        x_val.shape,
    )

    model = build_lstm_model((LOOKBACK, x_train.shape[-1]))
    metrics_callback = EpochMetricsCallback(
        fold_id=fold_id,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        artifacts=artifacts,
    )

    checkpoint_path = MODEL_PATH.with_name(f"lstm_model_fold_{fold_id}.keras")
    callbacks = [
        metrics_callback,
        EarlyStopping(
            monitor="val_rmse",
            patience=PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            mode="min",
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_rmse",
            factor=0.5,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-5,
            mode="min",
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_rmse",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    history_df = pd.DataFrame(metrics_callback.epoch_metrics)
    best_row = history_df.loc[history_df["val_rmse"].idxmin()]
    summary = {
        "fold_id": float(fold_id),
        "best_epoch": float(best_row["epoch"]),
        "best_val_rmse": float(best_row["val_rmse"]),
        "best_val_mae": float(best_row["val_mae"]),
    }

    logger.info(
        "Fold %d best | epoch=%d | val_rmse=%.4f | val_mae=%.4f",
        fold_id,
        int(best_row["epoch"]),
        float(best_row["val_rmse"]),
        float(best_row["val_mae"]),
    )
    return history_df, summary


def _train_final_model(train_df: pd.DataFrame, final_epochs: int) -> dict[str, float]:
    """Train the final submission model on the full training set."""
    train_features, target_array, artifacts = fit_transform_features(train_df)
    x_train, y_train = create_sequences(train_features, target_array, LOOKBACK)

    logger.info(
        "Final training | epochs=%d | feature_dim=%d | x_train=%s",
        final_epochs,
        train_features.shape[1],
        x_train.shape,
    )

    model = build_lstm_model((LOOKBACK, x_train.shape[-1]))
    history = model.fit(
        x_train,
        y_train,
        epochs=final_epochs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.insert(0, "fold", "final")
    history_df.to_csv(FINAL_HISTORY_PATH, index=False)
    logger.info("Saved final training history to %s", FINAL_HISTORY_PATH)

    _save_trained_models(model)
    save_feature_artifacts(artifacts, PREPROCESSOR_PATH)

    train_pred_scaled = model.predict(x_train, verbose=0).reshape(-1)
    train_rmse, train_mae = EpochMetricsCallback(
        fold_id="final",
        x_train=x_train,
        y_train=y_train,
        x_val=x_train,
        y_val=y_train,
        artifacts=artifacts,
    )._compute_metrics(y_train, train_pred_scaled)

    logger.info(
        "Final model fit | epochs=%d | train_rmse=%.4f | train_mae=%.4f",
        final_epochs,
        train_rmse,
        train_mae,
    )
    return {"final_epochs": float(final_epochs), "train_rmse": train_rmse, "train_mae": train_mae}


def train_model() -> Dict[str, float]:
    """Run the end-to-end walk-forward training workflow for the LSTM only."""
    ensure_project_dirs()
    set_random_seed()
    configure_runtime()

    if VALIDATION_MODE != "walk_forward":
        raise ValueError(f"Unsupported validation mode: {VALIDATION_MODE}")

    train_df, _ = load_train_test_data()
    fold_ranges = _create_walk_forward_folds(len(train_df), N_SPLITS)
    logger.info("Using walk-forward folds: %s", fold_ranges)

    fold_histories: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, float]] = []

    for fold_id, (val_start, val_end) in enumerate(fold_ranges, start=1):
        train_raw = train_df.iloc[:val_start].reset_index(drop=True)
        val_raw = train_df.iloc[val_start:val_end].reset_index(drop=True)
        history_df, summary = _train_single_fold(fold_id, train_raw, val_raw)
        fold_histories.append(history_df)
        fold_summaries.append(summary)

    combined_history_df = pd.concat(fold_histories, ignore_index=True)
    combined_history_df.to_csv(HISTORY_PATH, index=False)
    logger.info("Saved walk-forward training history to %s", HISTORY_PATH)
    _plot_training_history(combined_history_df, TRAINING_PLOT_PATH)

    avg_val_rmse = float(np.mean([summary["best_val_rmse"] for summary in fold_summaries]))
    avg_val_mae = float(np.mean([summary["best_val_mae"] for summary in fold_summaries]))
    avg_best_epoch = float(np.mean([summary["best_epoch"] for summary in fold_summaries]))
    final_epochs = max(5, int(round(avg_best_epoch)))

    logger.info(
        "Walk-forward summary | avg_val_rmse=%.4f | avg_val_mae=%.4f | avg_best_epoch=%.2f",
        avg_val_rmse,
        avg_val_mae,
        avg_best_epoch,
    )

    final_train_metrics = _train_final_model(train_df, final_epochs)

    return {
        "avg_val_rmse": avg_val_rmse,
        "avg_val_mae": avg_val_mae,
        "avg_best_epoch": avg_best_epoch,
        "final_epochs": final_train_metrics["final_epochs"],
        "final_train_rmse": final_train_metrics["train_rmse"],
        "final_train_mae": final_train_metrics["train_mae"],
    }
