"""Tests for the TF-IDF + logistic regression topic classifier."""

import json

import pytest

import text_classifier as tc


@pytest.fixture(scope="module")
def training_data():
    return tc.load_training_data()


@pytest.fixture(scope="module")
def trained():
    return tc.train_model()


def test_dataset_has_expected_shape(training_data):
    assert len(training_data) >= 100
    labels = sorted(training_data["label"].unique())
    assert labels == ["Finance", "Geography", "Health", "Science", "Technology"]
    counts = training_data["label"].value_counts()
    assert counts.min() >= tc.MIN_SAMPLES_PER_CLASS
    assert training_data["text"].str.len().gt(0).all()


def test_missing_training_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("body,class\nhello,Science\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tc.load_training_data(path)


def test_missing_training_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tc.load_training_data(tmp_path / "absent.csv")


def test_training_metrics_are_recorded(trained):
    pipeline, metrics = trained
    assert hasattr(pipeline, "predict_proba")
    assert metrics["train_size"] >= 100
    assert 0.0 <= metrics["cross_validation_accuracy"] <= 1.0
    assert metrics["cross_validation_folds"] >= 2
    assert metrics["cross_validation_accuracy"] > 0.7
    assert set(metrics["top_features_per_label"]) == set(metrics["labels"])
    assert metrics["note"]


def test_prediction_has_valid_probability_distribution(trained):
    pipeline, _ = trained
    prediction = tc.predict_with_confidence(
        pipeline, "The investor reviewed the quarterly earnings and the stock portfolio."
    )
    assert prediction.label in prediction.probabilities
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.confidence == pytest.approx(max(prediction.probabilities.values()))
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert len(prediction.probabilities) == 5


def test_prediction_learns_clear_examples(trained):
    pipeline, _ = trained
    sample = "The doctor checked the patient's blood pressure after the vaccination."
    assert tc.predict_with_confidence(pipeline, sample).label == "Health"


def test_empty_prediction_text_raises(trained):
    pipeline, _ = trained
    with pytest.raises(ValueError):
        tc.predict_with_confidence(pipeline, "   ")


def test_save_and_load_roundtrip(tmp_path, trained):
    pipeline, metrics = trained
    model_path, metrics_path = tc.save_artifacts(
        pipeline,
        metrics,
        tmp_path / "classifier.joblib",
        tmp_path / "metrics.json",
    )
    assert model_path.exists() and metrics_path.exists()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["labels"] == metrics["labels"]

    reloaded = tc.load_model(model_path)
    text = "The server returned a JSON response over HTTP."
    assert reloaded.predict([text]).tolist() == pipeline.predict([text]).tolist()
    assert tc.load_metrics(metrics_path)["train_size"] == metrics["train_size"]


def test_ensure_model_trains_when_artifact_is_missing(tmp_path):
    model_path = tmp_path / "nested" / "classifier.joblib"
    metrics_path = tmp_path / "nested" / "metrics.json"
    pipeline, metrics, trained_now = tc.ensure_model(
        model_path=model_path,
        metrics_path=metrics_path,
    )
    assert trained_now is True
    assert model_path.exists() and metrics_path.exists()
    assert metrics["labels"]

    _, _, trained_again = tc.ensure_model(model_path=model_path, metrics_path=metrics_path)
    assert trained_again is False


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tc.load_model(tmp_path / "missing.joblib")
