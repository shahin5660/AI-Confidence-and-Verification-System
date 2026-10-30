"""Smoke tests: run the Streamlit app headlessly and inspect the rendered UI."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run(text: str) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    app.text_area[0].set_value(text)
    next(button for button in app.button if button.label == "Analyse statement").click()
    app.run()
    assert not app.exception, app.exception
    return app


def _markdown_text(app: AppTest) -> str:
    return " ".join(block.value for block in app.markdown)


def test_app_loads_without_errors():
    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    assert not app.exception
    assert any("AI Confidence and Verification System" in title.value for title in app.title)


def test_contradicted_claim_shows_verdict():
    app = _run("Water boils at 90 degrees Celsius at sea level.")
    assert "Contradicted" in _markdown_text(app)


def test_supported_claim_shows_verdict():
    app = _run("The capital of France is Paris.")
    assert "Supported" in _markdown_text(app)


def test_unverifiable_claim_shows_insufficient_evidence():
    app = _run("My favourite colour is blue.")
    assert "Insufficient Evidence" in _markdown_text(app)


def test_two_result_cards_and_probability_chart_are_rendered():
    app = _run("The capital of France is Paris.")
    rendered = _markdown_text(app)

    # Card titles for both independent signals.
    assert "🤖 Model confidence" in rendered
    assert "📚 Reference verification" in rendered

    # A probability chart plus a confidence metric.
    assert len(app.get("vega_lite_chart")) >= 1
    metric_labels = [metric.label for metric in app.metric]
    assert "Predicted topic" in metric_labels
    assert "Confidence" in metric_labels


def test_expanders_for_reference_facts_and_project_information():
    app = _run("The capital of France is Paris.")
    labels = [expander.label for expander in app.get("expander")]
    assert any("Reference facts" in label for label in labels)
    assert any("Project information" in label for label in labels)


def test_matching_reference_record_is_shown():
    app = _run("The capital of France is Paris.")
    rendered = _markdown_text(app)
    assert "Matching reference record" in rendered
    assert "kb-003" in rendered


def test_example_selector_fills_the_text_area():
    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    app.selectbox[0].set_value("Water boils at 90 degrees Celsius at sea level.")
    app.run()
    assert app.text_area[0].value == "Water boils at 90 degrees Celsius at sea level."
    assert not app.exception
