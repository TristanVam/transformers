"""Command-line entry point for training the hybrid PatchTST model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.training.config import ExperimentConfig
from src.training.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stochastic PatchTST model")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON file overriding default experiment parameters.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    return ExperimentConfig.from_json(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
