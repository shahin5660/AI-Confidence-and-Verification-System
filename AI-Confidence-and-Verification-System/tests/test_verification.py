"""Tests for the rule-based reference verification."""

import pandas as pd
import pytest

import verification as vf


@pytest.fixture(scope="module")
def knowledge_base() -> pd.DataFrame:
    return vf.load_knowledge_base()


def verdict(text: str, knowledge_base: pd.DataFrame) -> str:
    return vf.verify_text(text, knowledge_base).verdict


def test_knowledge_base_loads_with_required_columns(knowledge_base):
    for column in vf.REQUIRED_COLUMNS:
        assert column in knowledge_base.columns
    assert len(knowledge_base) >= 10
    assert not knowledge_base["id"].duplicated().any()
    assert (knowledge_base["statement"].str.len() > 0).all()
    assert (knowledge_base["source"].str.len() > 0).all()


def test_knowledge_base_rejects_missing_columns(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"id": ["x"], "topic": ["t"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        vf.load_knowledge_base(path)


def test_knowledge_base_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.csv"
    row = {
        "id": "kb-x",
        "topic": "t",
        "statement": "s",
        "keywords": "a",
        "expected_value": "1",
        "unit": "",
        "alternatives": "",
        "source": "src",
    }
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        vf.load_knowledge_base(path)


def test_missing_knowledge_base_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vf.load_knowledge_base(tmp_path / "absent.csv")


def test_supported_numeric_claim(knowledge_base):
    result = vf.verify_text("Water boils at 100 degrees Celsius at sea level.", knowledge_base)
    assert result.verdict == vf.SUPPORTED
    assert result.matched_id == "kb-001"
    assert result.topic == "Boiling point of water"
    assert result.source


def test_contradicted_numeric_claim(knowledge_base):
    result = vf.verify_text("Water boils at 90 degrees Celsius at sea level.", knowledge_base)
    assert result.verdict == vf.CONTRADICTED
    assert "90" in result.reason


def test_negated_numeric_claim_is_contradicted(knowledge_base):
    result = vf.verify_text("Water does not boil at 100 degrees Celsius.", knowledge_base)
    assert result.verdict == vf.CONTRADICTED


def test_supported_text_claim(knowledge_base):
    result = vf.verify_text("The capital of France is Paris.", knowledge_base)
    assert result.verdict == vf.SUPPORTED
    assert result.matched_id == "kb-003"


def test_contradicted_text_claim_via_alternative(knowledge_base):
    result = vf.verify_text("The capital of France is Berlin.", knowledge_base)
    assert result.verdict == vf.CONTRADICTED
    assert "Berlin" in result.reason


def test_negated_text_claim_is_contradicted(knowledge_base):
    result = vf.verify_text("The capital of France is not Paris.", knowledge_base)
    assert result.verdict == vf.CONTRADICTED


def test_topic_match_without_comparable_number_is_insufficient(knowledge_base):
    result = vf.verify_text("Water boils when it is heated.", knowledge_base)
    assert result.verdict == vf.INSUFFICIENT


def test_unit_that_cannot_be_compared_is_insufficient(knowledge_base):
    result = vf.verify_text("The height of Mount Everest is 8849 feet.", knowledge_base)
    assert result.verdict == vf.INSUFFICIENT


def test_number_in_different_temperature_scale_is_insufficient(knowledge_base):
    result = vf.verify_text("Water boils at 212 degrees Fahrenheit.", knowledge_base)
    assert result.verdict == vf.INSUFFICIENT


def test_sleep_guidance_is_supported(knowledge_base):
    result = vf.verify_text("Adults should get at least 7 hours of sleep per night.", knowledge_base)
    assert result.verdict == vf.SUPPORTED
    assert result.matched_id == "kb-014"


def test_wrong_sleep_number_is_contradicted(knowledge_base):
    result = vf.verify_text("Adults need 5 hours of sleep per night.", knowledge_base)
    assert result.verdict == vf.CONTRADICTED


def test_unrelated_text_is_insufficient(knowledge_base):
    result = vf.verify_text("My favourite colour is blue.", knowledge_base)
    assert result.verdict == vf.INSUFFICIENT
    assert result.matched_id is None


def test_empty_text_is_insufficient(knowledge_base):
    assert verdict("", knowledge_base) == vf.INSUFFICIENT
    assert verdict("    ", knowledge_base) == vf.INSUFFICIENT


def test_https_port_is_not_confused_with_http(knowledge_base):
    assert verdict("The default port for HTTPS is 443.", knowledge_base) == vf.SUPPORTED
    assert verdict("The default port for HTTP is 8080.", knowledge_base) == vf.CONTRADICTED


def test_text_mentions_two_reference_topics_prefers_more_specific_row(knowledge_base):
    result = vf.verify_text(
        "The capital of France is Paris and the capital of Japan is Tokyo.",
        knowledge_base,
    )
    assert result.verdict == vf.SUPPORTED
    assert result.matched_id == "kb-003"


def test_verification_does_not_use_the_classifier(knowledge_base):
    """A sentence with no reference match stays Insufficient, however confident a model is."""
    result = vf.verify_text("The stock market rallied after the earnings report.", knowledge_base)
    assert result.verdict == vf.INSUFFICIENT
