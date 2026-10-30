"""Train and use the TF-IDF + logistic regression topic classifier.

The model predicts a *topic label* (Science, Technology, Health, Geography,
Finance) and a probability for each label. Those probabilities are the model's
confidence, which is **not** evidence that a statement is true. Factual checks
live in verification.py and never touch this model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "training_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

REQUIRED_COLUMNS = ["text", "label"]
MIN_SAMPLES_PER_CLASS = 5


@dataclass
class Prediction:
    """One model prediction, kept clearly separate from factual verification."""

    label: str
    confidence: float
    probabilities: dict[str, float]

    @property
    def runner_up(self) -> tuple[str, float]:
        ordered = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)
        return ordered[1] if len(ordered) > 1 else ordered[0]


def load_training_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read and validate the labelled sample dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [column.strip() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing)}")

    frame["text"] = frame["text"].str.strip()
    frame["label"] = frame["label"].str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Training data is empty after removing blank rows.")
    if frame["label"].nunique() < 2:
        raise ValueError("Training data needs at least two different labels.")
    return frame


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + logistic regression pipeline used by the project."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _top_features(pipeline: Pipeline, per_class: int = 5) -> dict[str, list[str]]:
    """Report the strongest learned words per label, for transparency."""
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    names = vectorizer.get_feature_names_out()
    summary: dict[str, list[str]] = {}
    for index, label in enumerate(classifier.classes_):
        weights = classifier.coef_[index]
        best = weights.argsort()[-per_class:][::-1]
        summary[str(label)] = [str(names[i]) for i in best]
    return summary


def train_model(
    data_path: str | Path = DATA_PATH,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict]:
    """Train the classifier and return the fitted pipeline plus a metrics dict."""
    frame = load_training_data(data_path)
    labels = sorted(frame["label"].unique().tolist())
    smallest_class = int(frame["label"].value_counts().min())
    if smallest_class < MIN_SAMPLES_PER_CLASS:
        raise ValueError(
            f"Every label needs at least {MIN_SAMPLES_PER_CLASS} examples; "
            f"the smallest class has {smallest_class}."
        )

    pipeline = build_pipeline()
    pipeline.fit(frame["text"], frame["label"])

    folds = max(2, min(cv_folds, smallest_class))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, frame["text"], frame["label"], cv=splitter)

    train_accuracy = float(pipeline.score(frame["text"], frame["label"]))
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(Path(data_path).name),
        "train_size": int(len(frame)),
        "labels": labels,
        "samples_per_label": {
            label: int(count) for label, count in frame["label"].value_counts().sort_index().items()
        },
        "cross_validation_folds": folds,
        "cross_validation_accuracy": round(float(scores.mean()), 4),
        "cross_validation_scores": [round(float(score), 4) for score in scores],
        "training_accuracy": round(train_accuracy, 4),
        "top_features_per_label": _top_features(pipeline),
        "note": (
            "Accuracy is measured on a tiny synthetic sample dataset. "
            "It describes topic-classification skill only, not factual reliability."
        ),
    }
    return pipeline, metrics


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
) -> tuple[Path, Path]:
    """Persist the fitted pipeline and its metrics next to each other."""
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model_path, metrics_path


def load_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    """Load a pipeline saved by :func:`save_artifacts`."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run 'python train_model.py' first."
        )
    return joblib.load(model_path)


def load_metrics(metrics_path: str | Path = METRICS_PATH) -> dict:
    """Load saved metrics, returning an empty dict when they are unavailable."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def predict_with_confidence(pipeline: Pipeline, text: str) -> Prediction:
    """Predict the topic label and the model's confidence for one text."""
    if not text or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    probabilities = pipeline.predict_proba([text])[0]
    labels = [str(label) for label in pipeline.classes_]
    mapping = {label: float(probability) for label, probability in zip(labels, probabilities)}
    best_index = int(probabilities.argmax())
    return Prediction(
        label=labels[best_index],
        confidence=float(probabilities[best_index]),
        probabilities=mapping,
    )


def ensure_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
) -> tuple[Pipeline, dict, bool]:
    """Load the saved model, training it first if the artifact is missing.

    Returns ``(pipeline, metrics, trained_now)``.
    """
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    if model_path.exists():
        return load_model(model_path), load_metrics(metrics_path), False

    pipeline, metrics = train_model(data_path)
    save_artifacts(pipeline, metrics, model_path, metrics_path)
    return pipeline, metrics, True
