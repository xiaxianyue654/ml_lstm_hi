"""Compare two CSV files and compute the MSE of the cnt column.

Usage:
    python compare_mse.py
    python compare_mse.py --pred submission.csv --truth "test(1).csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PRED_PATH = Path(__file__).resolve().parent / "submission.csv"
DEFAULT_TRUTH_PATH = Path(__file__).resolve().parent / "test(1).csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute cnt MSE between two CSV files.")
    parser.add_argument(
        "--pred",
        type=Path,
        default=DEFAULT_PRED_PATH,
        help="Prediction CSV path. Default: submission.csv in the current folder.",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=DEFAULT_TRUTH_PATH,
        help="Ground-truth CSV path. Default: test(1).csv in the current folder.",
    )
    return parser.parse_args()


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if "cnt" not in dataframe.columns:
        raise ValueError(f"Missing required column 'cnt' in: {csv_path}")
    return dataframe


def align_series(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align prediction and truth cnt values by ID when possible, otherwise by row order."""
    if "ID" in pred_df.columns and "ID" in truth_df.columns:
        merged_df = truth_df[["ID", "cnt"]].merge(
            pred_df[["ID", "cnt"]],
            on="ID",
            how="inner",
            suffixes=("_true", "_pred"),
        )
        if len(merged_df) != len(truth_df) or len(merged_df) != len(pred_df):
            raise ValueError("ID alignment failed: the two files do not contain the same ID set.")
        y_true = merged_df["cnt_true"].to_numpy(dtype=np.float64)
        y_pred = merged_df["cnt_pred"].to_numpy(dtype=np.float64)
        return y_true, y_pred

    if len(pred_df) != len(truth_df):
        raise ValueError("Row counts do not match, and ID columns are unavailable for alignment.")

    y_true = truth_df["cnt"].to_numpy(dtype=np.float64)
    y_pred = pred_df["cnt"].to_numpy(dtype=np.float64)
    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute MSE and RMSE."""
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse


def main() -> None:
    """Run the comparison."""
    args = parse_args()
    pred_df = load_csv(args.pred)
    truth_df = load_csv(args.truth)
    y_true, y_pred = align_series(pred_df, truth_df)
    mse, rmse = compute_metrics(y_true, y_pred)

    print(f"Prediction file: {args.pred}")
    print(f"Ground truth file: {args.truth}")
    print(f"Samples compared: {len(y_true)}")
    print(f"cnt MSE: {mse:.6f}")
    print(f"cnt RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
