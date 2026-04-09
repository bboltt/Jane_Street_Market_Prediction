"""Command line entry point for training and managing model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .service import JaneStreetModelService


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the training CLI."""
    parser = argparse.ArgumentParser(description="Jane Street model management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and persist baseline model")
    train_parser.add_argument("--data", required=True, help="CSV path containing features and target")
    train_parser.add_argument("--target", default="action", help="Target column name")
    train_parser.add_argument("--out", default="artifacts/baseline_model.joblib", help="Output model path")

    return parser


def main() -> int:
    """Run CLI command and return process exit code."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        frame = pd.read_csv(args.data)
        if args.target not in frame.columns:
            raise ValueError(f"Target column '{args.target}' not found in dataset.")

        target = frame[args.target]
        features = frame.drop(columns=[args.target])

        service = JaneStreetModelService()
        service.train(features, target)
        output = service.save(Path(args.out))
        print(f"Model saved to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
