import math
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


LOOKBACK = 24
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
SEED = 42


@dataclass
class StandardStats:
    mean: pd.Series
    std: pd.Series


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_standardize(frame: pd.DataFrame, stats: StandardStats) -> pd.DataFrame:
    return (frame - stats.mean) / stats.std.replace(0, 1.0)


def add_cyclic_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dteday"] = pd.to_datetime(frame["dteday"])

    frame["sin_hr"] = np.sin(2 * math.pi * frame["hr"] / 24.0)
    frame["cos_hr"] = np.cos(2 * math.pi * frame["hr"] / 24.0)
    frame["sin_wd"] = np.sin(2 * math.pi * frame["weekday"] / 7.0)
    frame["cos_wd"] = np.cos(2 * math.pi * frame["weekday"] / 7.0)
    frame["sin_mn"] = np.sin(2 * math.pi * frame["mnth"] / 12.0)
    frame["cos_mn"] = np.cos(2 * math.pi * frame["mnth"] / 12.0)
    return frame


def preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float, float]:
    train_df = add_cyclic_features(train_df).sort_values(["dteday", "hr", "ID"]).reset_index(drop=True)
    test_df = add_cyclic_features(test_df).sort_values(["dteday", "hr", "ID"]).reset_index(drop=True)

    train_target = train_df["cnt"].astype("float32")
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    if target_std == 0:
        target_std = 1.0
    y_scaled = ((train_target - target_mean) / target_std).to_numpy(dtype=np.float32)

    combined = pd.concat(
        [
            train_df.assign(_split="train"),
            test_df.assign(_split="test"),
        ],
        ignore_index=True,
        sort=False,
    )

    categorical_cols = ["season", "weathersit"]
    combined = pd.get_dummies(combined, columns=categorical_cols, prefix=categorical_cols, drop_first=False)

    continuous_cols = ["temp", "atemp", "hum", "windspeed"]
    continuous_stats = StandardStats(
        mean=combined.loc[combined["_split"] == "train", continuous_cols].mean(),
        std=combined.loc[combined["_split"] == "train", continuous_cols].std().replace(0, 1.0),
    )
    combined.loc[:, continuous_cols] = safe_standardize(combined[continuous_cols], continuous_stats)

    drop_cols = ["ID", "dteday", "cnt", "_split", "hr", "weekday", "mnth"]
    feature_cols = [col for col in combined.columns if col not in drop_cols]

    train_features = combined.loc[combined["_split"] == "train", feature_cols].reset_index(drop=True)
    test_features = combined.loc[combined["_split"] == "test", feature_cols].reset_index(drop=True)

    return train_features, test_features, y_scaled, target_mean, target_std


def create_sequence_block(
    features: np.ndarray,
    targets: np.ndarray | None,
    lookback: int,
    start_idx: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    xs = []
    ys = []

    for target_idx in range(start_idx, end_idx):
        xs.append(features[target_idx - lookback : target_idx])
        if targets is not None:
            ys.append(targets[target_idx])

    x_array = np.asarray(xs, dtype=np.float32)
    if targets is None:
        return x_array, None
    y_array = np.asarray(ys, dtype=np.float32)
    return x_array, y_array


def build_model(lookback: int, num_features: int) -> tf.keras.Model:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, dropout=0.2, input_shape=(lookback, num_features)),
            LSTM(32, return_sequences=False, dropout=0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def main() -> None:
    set_seed()

    train_path = os.path.join("data-bike", "train.csv")
    test_path = os.path.join("data-bike", "test.csv")
    submission_path = "submission.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if len(train_df) <= LOOKBACK:
        raise ValueError(f"Training rows must be greater than lookback={LOOKBACK}.")

    train_features_df, test_features_df, y_scaled, y_mean, y_std = preprocess_features(train_df, test_df)
    train_features = train_features_df.to_numpy(dtype=np.float32)
    test_features = test_features_df.to_numpy(dtype=np.float32)

    split_index = int(len(train_features) * TRAIN_RATIO)
    split_index = max(split_index, LOOKBACK + 1)
    if split_index >= len(train_features):
        raise ValueError("Training/validation split leaves no validation samples.")

    x_train, y_train = create_sequence_block(
        features=train_features,
        targets=y_scaled,
        lookback=LOOKBACK,
        start_idx=LOOKBACK,
        end_idx=split_index,
    )
    x_val, y_val = create_sequence_block(
        features=train_features,
        targets=y_scaled,
        lookback=LOOKBACK,
        start_idx=split_index,
        end_idx=len(train_features),
    )

    model = build_model(LOOKBACK, train_features.shape[1])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    val_pred_scaled = model.predict(x_val, verbose=0).reshape(-1)
    val_pred = val_pred_scaled * y_std + y_mean
    val_true = y_val * y_std + y_mean
    val_rmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))
    val_mae = float(np.mean(np.abs(val_pred - val_true)))
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")

    # The model only uses past feature windows, so test windows can be built
    # directly from the concatenated train/test feature timeline.
    combined_features = np.vstack([train_features, test_features]).astype(np.float32)
    test_start = len(train_features)
    x_test, _ = create_sequence_block(
        features=combined_features,
        targets=None,
        lookback=LOOKBACK,
        start_idx=test_start,
        end_idx=len(combined_features),
    )

    test_pred_scaled = model.predict(x_test, verbose=0).reshape(-1)
    test_pred = test_pred_scaled * y_std + y_mean
    test_pred = np.clip(test_pred, 0.0, None)

    submission = pd.DataFrame(
        {
            "ID": test_df["ID"].to_numpy(),
            "cnt": test_pred,
        }
    )
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to: {submission_path}")


if __name__ == "__main__":
    main()
