"""Tests for the engine layer that joins the classifier and the verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import engine as eng
import text_classifier as tc
import verification as vf


@pytest.fixture(scope="module")
def loaded_engine() -> eng.AnalysisEngine:
    return eng.load_engine()


def test_knowledge_base_is_found():
    path = eng.find_knowledge_base_path()
    assert path.exists()
    assert path.name == "knowledge_base.csv"


def test_missing_knowledge_base_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eng,
        "KNOWLEDGE_BASE_CANDIDATES",
        (tmp_path / "knowledge_base.csv", tmp_path / "data" / "knowledge_base.csv"),
    )
    with pytest.raises(FileNotFoundError) as error:
        eng.find_knowledge_base_path()
    assert "Knowledge base not found" in str(error.value)


def test_root_level_knowledge_base_is_accepted(monkeypatch, tmp_path):
    copied = tmp_path / "knowledge_base.csv"
    copied.write_text(
        vf.KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(
        eng,
        "KNOWLEDGE_BASE_CANDIDATES",
        (copied, eng.BASE_DIR / "data" / "knowledge_base.csv"),
    )
    assert eng.find_knowledge_base_path() == copied


def test_engine_loads_model_and_reference_data(loaded_engine: eng.AnalysisEngine):
    assert loaded_engine.pipeline is not None
    assert len(loaded_engine.labels) == 5
    assert not loaded_engine.knowledge_base.empty
    assert loaded_engine.knowledge_base_path is not None


def test_analyse_keeps_the_two_signals_separate(loaded_engine: eng.AnalysisEngine):
    analysis = loaded_engine.analyse(
        "Water boils at 90 degrees Celsius at sea level."
    )

    assert analysis.topic in loaded_engine.labels
    assert 0.0 < analysis.confidence <= 1.0
    assert analysis.verdict == vf.CONTRADICTED
    assert analysis.has_reference_match

    # The verdict must be exactly what the verifier produces on its own.
    direct = vf.verify_text(
        "Water boils at 90 degrees Celsius at sea level.",
        loaded_engine.knowledge_base,
    )
    assert analysis.verification == direct


def test_analyse_returns_insufficient_evidence_for_unknown_text(
    loaded_engine: eng.AnalysisEngine,
):
    analysis = loaded_engine.analyse("My favourite colour is blue.")
    assert analysis.verdict == vf.INSUFFICIENT
    assert not analysis.has_reference_match


def test_blank_text_is_rejected(loaded_engine: eng.AnalysisEngine):
    with pytest.raises(ValueError):
        loaded_engine.analyse("   ")


def test_reference_row_lookup(loaded_engine: eng.AnalysisEngine):
    analysis = loaded_engine.analyse("The capital of France is Paris.")
    row = loaded_engine.reference_row(analysis.verification.matched_id)
    assert row is not None
    assert row["topic"] == "Capital of France"
    assert loaded_engine.reference_row(None) is None
    assert loaded_engine.reference_row("does-not-exist") is None


def test_analysis_to_dict_is_json_serialisable(loaded_engine: eng.AnalysisEngine):
    payload = json.loads(json.dumps(loaded_engine.analyse("The capital of France is Paris.").to_dict()))
    assert payload["predicted_topic"] in loaded_engine.labels
    assert payload["verification"]["verdict"] == vf.SUPPORTED


def test_engine_does_not_retrain_when_artifacts_exist():
    # The repository ships model/classifier.joblib, so loading must not retrain.
    assert tc.MODEL_PATH.exists()
    engine = eng.load_engine()
    assert engine.trained_now is False
