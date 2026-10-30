"""Train the topic classifier and save it to the model/ folder.

Usage (from the project folder):

    python train_model.py

Optional flags:

    python train_model.py --data data/training_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import text_classifier as tc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the TF-IDF + logistic regression topic classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=tc.DATA_PATH,
        help="Path to the labelled CSV file (default: data/training_data.csv).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=tc.MODEL_PATH,
        help="Where to save the trained pipeline (default: model/classifier.joblib).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=tc.METRICS_PATH,
        help="Where to save the metrics JSON (default: model/metrics.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("Training the topic classifier ...")
    pipeline, metrics = tc.train_model(args.data)
    model_path, metrics_path = tc.save_artifacts(pipeline, metrics, args.model, args.metrics)

    print("\nTraining complete.")
    print(f"  Examples        : {metrics['train_size']}")
    print(f"  Labels          : {', '.join(metrics['labels'])}")
    print(
        f"  Cross-validated accuracy: {metrics['cross_validation_accuracy']:.1%} "
        f"({metrics['cross_validation_folds']} folds)"
    )
    print(f"  Training accuracy        : {metrics['training_accuracy']:.1%}")
    print("  Strongest words per label:")
    for label, words in metrics["top_features_per_label"].items():
        print(f"    - {label}: {', '.join(words)}")
    print(f"\nSaved model   : {model_path}")
    print(f"Saved metrics : {metrics_path}")
    print("\nNext step: python -m streamlit run app.py")
    print("Note: confidence is a model score, not a factual truth check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
