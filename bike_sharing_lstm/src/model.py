"""Model definition for bike sharing demand forecasting."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from .config import DENSE_UNITS, DROPOUT, LEARNING_RATE, LSTM_UNITS, WEIGHT_DECAY


@tf.keras.utils.register_keras_serializable()
def build_lstm_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """Build and compile the two-layer LSTM model."""
    regularizer = l2(WEIGHT_DECAY)
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(
                LSTM_UNITS[0],
                return_sequences=True,
                dropout=DROPOUT,
                kernel_regularizer=regularizer,
            ),
            LSTM(
                LSTM_UNITS[1],
                return_sequences=False,
                dropout=DROPOUT,
                kernel_regularizer=regularizer,
            ),
            Dense(DENSE_UNITS, activation="relu", kernel_regularizer=regularizer),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    return model
