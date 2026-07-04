"""Data loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import TEST_DATA_PATH, TRAIN_DATA_PATH


logger = logging.getLogger(__name__)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file, parse the date column, and sort by time."""
    try:
        dataframe = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Data file not found: {csv_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV file: {csv_path}") from exc

    required_columns = {"ID", "dteday", "hr"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing_str}")

    try:
        dataframe["dteday"] = pd.to_datetime(dataframe["dteday"])
    except Exception as exc:
        raise ValueError(f"Failed to convert dteday to datetime in {csv_path.name}") from exc

    dataframe = dataframe.sort_values(["dteday", "hr", "ID"]).reset_index(drop=True)
    logger.info("Loaded %s with shape %s", csv_path.name, dataframe.shape)
    return dataframe


def load_train_test_data(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load sorted train and test datasets."""
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)
    return train_df, test_df

