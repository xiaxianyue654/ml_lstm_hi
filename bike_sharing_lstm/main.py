"""Command-line entrypoint for the bike sharing LSTM project."""

from __future__ import annotations

import argparse
import logging

from src.config import OUTPUT_DIR, SUBMISSION_PATH, ensure_project_dirs
from src.predict import generate_submission
from src.train import train_model


def configure_logging() -> None:
    """Initialize application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bike sharing demand forecasting with LSTM")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "all"],
        default="all",
        help="Run only training, only prediction, or the full pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested workflow."""
    configure_logging()
    ensure_project_dirs()
    args = parse_args()

    if args.mode in {"train", "all"}:
        train_metrics = train_model()
        logging.getLogger(__name__).info("Training complete: %s", train_metrics)

    if args.mode in {"predict", "all"}:
        submission = generate_submission()
        logging.getLogger(__name__).info(
            "Prediction complete. Submission shape=%s, path=%s",
            submission.shape,
            SUBMISSION_PATH,
        )

    logging.getLogger(__name__).info("Outputs available under %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

