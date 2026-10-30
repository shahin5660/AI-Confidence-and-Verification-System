"""AI Confidence and Verification System - Streamlit front end.

Run with:

    python -m streamlit run app.py

Two deliberately separate signals are shown for every text:

1. Model confidence - how strongly the TF-IDF + logistic regression model
   favours a topic label. This is a statistical score, not a truth score.
2. Reference verification - a rule-based lookup in the small, transparent
   knowledge base (`knowledge_base.csv`). It can only return Supported,
   Contradicted, or Insufficient Evidence.

`engine.py` loads the model and the reference data and produces both signals;
this file only draws them.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import engine as eng
import text_classifier as tc
import verification as vf

st.set_page_config(
    page_title="AI Confidence and Verification System",
    page_icon="🔎",
    layout="wide",
)

VERDICT_STYLES = {
    vf.SUPPORTED: {
        "icon": "✅",
        "text": "#0f5132",
        "background": "#d1e7dd",
        "border": "#a3cfbb",
    },
    vf.CONTRADICTED: {
        "icon": "❌",
        "text": "#842029",
        "background": "#f8d7da",
        "border": "#f1aeb5",
    },
    vf.INSUFFICIENT: {
        "icon": "⚠️",
        "text": "#664d03",
        "background": "#fff3cd",
        "border": "#ffe69c",
    },
}

EXAMPLES = [
    "Water boils at 90 degrees Celsius at sea level.",
    "The capital of France is Paris.",
    "Mount Everest rises 8849 metres above sea level.",
    "The capital of Japan is Osaka.",
    "Quantum tunnelling explains how stars fuse hydrogen.",
    "The default port for HTTP is 80.",
]

CARD_CSS = """
<style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.75rem;
    }
    .aicv-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 0.15rem;
    }
    .aicv-card-subtitle {
        color: #6c757d;
        font-size: 0.85rem;
        margin-bottom: 0.9rem;
    }
    .aicv-record {
        border: 1px solid #dee2e6;
        border-radius: 0.6rem;
        padding: 0.8rem 0.9rem;
        margin-top: 0.6rem;
        background: rgba(127, 127, 127, 0.05);
    }
    .aicv-record-id {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.8rem;
        color: #6c757d;
    }
</style>
"""


@st.cache_resource(show_spinner="Loading the classifier and reference knowledge base ...")
def get_engine() -> tuple[eng.AnalysisEngine | None, str | None]:
    """Load the engine once per session, reporting problems in the UI."""
    try:
        return eng.load_engine(), None
    except Exception as error:  # surfaced in the UI instead of crashing the app
        return None, str(error)


def verdict_style(verdict: str) -> dict:
    return VERDICT_STYLES.get(
        verdict,
        {"icon": "•", "text": "#212529", "background": "#e9ecef", "border": "#ced4da"},
    )


def verdict_badge(verdict: str) -> None:
    """Render the verification verdict as a coloured badge."""
    style = verdict_style(verdict)
    st.markdown(
        f"""
        <div style="background:{style['background']};border:1px solid {style['border']};
                    color:{style['text']};padding:0.8rem 1.1rem;border-radius:0.6rem;
                    font-weight:700;font-size:1.15rem;text-align:center;">
            {style['icon']} {verdict}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="aicv-card-title">{title}</div>'
        f'<div class="aicv-card-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def render_confidence_card(analysis: eng.Analysis) -> None:
    """Predicted topic, confidence, probability chart and runner-up."""
    prediction = analysis.prediction
    topic_column, confidence_column = st.columns(2)
    topic_column.metric("Predicted topic", prediction.label)
    confidence_column.metric(
        "Confidence",
        f"{prediction.confidence:.1%}",
        help="The model's probability for the winning label. Not a truth score.",
    )
    st.progress(min(max(prediction.confidence, 0.0), 1.0))

    chart_data = pd.DataFrame(
        {
            "Topic": list(prediction.probabilities.keys()),
            "Probability": list(prediction.probabilities.values()),
        }
    ).sort_values("Probability", ascending=False, ignore_index=True)

    st.markdown("**Probability per topic**")
    st.bar_chart(
        chart_data.set_index("Topic"),
        y="Probability",
        color="#2f6feb",
        height=230,
    )

    runner_up_label, runner_up_score = prediction.runner_up
    st.caption(
        f"Model certainty only. Runner-up: **{runner_up_label}** at {runner_up_score:.1%}. "
        "A high score means the wording looks like the training examples for that topic - "
        "it says nothing about whether the sentence is true."
    )

    with st.expander("All class probabilities"):
        table = chart_data.copy()
        table["Probability"] = table["Probability"].map(lambda value: f"{value:.1%}")
        st.dataframe(table, hide_index=True)


def render_record(analysis: eng.Analysis, engine: eng.AnalysisEngine) -> None:
    """Show the exact reference row behind the verdict."""
    result = analysis.verification
    if not result.matched_id:
        st.caption("No reference row was used for this verdict.")
        return

    row = engine.reference_row(result.matched_id)
    unit_label = vf.UNIT_LABELS.get(str(row["unit"]).strip().lower(), str(row["unit"])) if row is not None else ""
    expected = str(row["expected_value"]) if row is not None else ""
    alternatives = str(row["alternatives"]) if row is not None else ""

    st.markdown("**Matching reference record**")
    st.markdown(
        f"""
        <div class="aicv-record">
            <div class="aicv-record-id">row {result.matched_id} &middot; {result.topic}</div>
            <div style="margin-top:0.35rem;">{result.statement}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    details = [
        ("Expected value", f"{expected} {unit_label}".strip() or "—"),
        ("Alternatives treated as wrong", alternatives.replace(";", ", ") or "—"),
        ("Source", result.source or "—"),
        (
            "Matched keywords",
            ", ".join(result.matched_keywords) if result.matched_keywords else "—",
        ),
    ]
    st.dataframe(pd.DataFrame(details, columns=["Field", "Value"]).astype(str), hide_index=True)


def render_verification_card(analysis: eng.Analysis, engine: eng.AnalysisEngine) -> None:
    """Verdict, plain-language explanation, and the matching reference record."""
    result = analysis.verification
    st.markdown("**Verification status**")
    verdict_badge(result.verdict)

    st.markdown("**Explanation**")
    st.write(result.reason)

    render_record(analysis, engine)

    st.caption(
        "Verdicts come only from the local rules in `verification.py` plus "
        "`knowledge_base.csv`. The classifier's output is never used here."
    )


def render_reference_facts(engine: eng.AnalysisEngine, analysis: eng.Analysis | None) -> None:
    """Expandable view of the whole reference knowledge base."""
    knowledge_base = engine.knowledge_base
    matched_id = analysis.verification.matched_id if analysis else None
    with st.expander(f"📚 Reference facts ({len(knowledge_base)} rows)", expanded=bool(matched_id)):
        if matched_id:
            st.success(f"This analysis matched reference row **{matched_id}**.")
        else:
            st.info(
                "No reference row matched the last analysed text, so every verdict "
                "would be Insufficient Evidence."
            )
        st.dataframe(
            knowledge_base[
                ["id", "topic", "statement", "expected_value", "unit", "alternatives", "source"]
            ].astype(str),
            hide_index=True,
        )
        st.caption(
            "The app can only verify claims covered here. A row applies only when all of "
            "its keywords appear in the text; numeric claims must include a comparable "
            "number and unit (for example '100 °C')."
        )


def render_project_information(engine: eng.AnalysisEngine) -> None:
    """Expandable project and model documentation."""
    metrics = engine.metrics
    with st.expander("ℹ️ Project information and model details"):
        st.markdown(
            """
**Why two boxes?** A topic classifier and a fact checker answer different questions.
The model says *"this reads like a Science sentence, and I am 84% sure"*. It cannot
say whether the sentence is true. Verification is a separate, readable rule set over
a tiny reference file, so every verdict can be traced to a row and a source.
            """
        )

        st.markdown("**Pipeline**")
        st.markdown(
            """
1. Text becomes TF-IDF features (single words and word pairs, English stop words removed).
2. Logistic regression predicts one of five topics plus a probability per topic.
3. Independently, `verification.py` matches the text against `knowledge_base.csv`.
4. Only those rules may return **Supported**, **Contradicted**, or **Insufficient Evidence**.
            """
        )

        st.markdown("**Model metrics**")
        if metrics:
            st.dataframe(
                pd.DataFrame(
                    [
                        ("Cross-validated accuracy", f"{metrics.get('cross_validation_accuracy', 0):.1%}"),
                        ("Training accuracy", f"{metrics.get('training_accuracy', 0):.1%}"),
                        ("Training rows", metrics.get("train_size", "n/a")),
                        ("Cross-validation folds", metrics.get("cross_validation_folds", "n/a")),
                        ("Labels", ", ".join(metrics.get("labels", []))),
                        ("Trained at (UTC)", metrics.get("trained_at", "unknown")),
                    ],
                    columns=["Metric", "Value"],
                ).astype(str),
                hide_index=True,
            )
            top_features = metrics.get("top_features_per_label") or {}
            if top_features:
                st.markdown("**Strongest learned words per topic**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            (label, ", ".join(words))
                            for label, words in sorted(top_features.items())
                        ],
                        columns=["Topic", "Top features"],
                    ),
                    hide_index=True,
                )
            st.caption(
                "Accuracy is measured on a small synthetic sample dataset. It describes "
                "topic-classification skill only, never factual reliability."
            )
        else:
            st.warning("Model metrics are unavailable. Run `python train_model.py`.")

        st.markdown("**Files**")
        st.markdown(
            """
| File | Role |
| --- | --- |
| `app.py` | Streamlit layout and rendering |
| `engine.py` | Loads the model + reference data, returns both signals |
| `text_classifier.py` | TF-IDF + logistic regression training and prediction |
| `verification.py` | Transparent rules for Supported / Contradicted / Insufficient Evidence |
| `train_model.py` | Command-line training script |
| `knowledge_base.csv` | Reference facts (shipped as `data/knowledge_base.csv`) |
| `data/training_data.csv` | 180 labelled sentences across five topics |
            """
        )

        st.markdown("**Limitations**")
        st.markdown(
            """
- The training set is tiny and synthetic; the model does not generalise far.
- The knowledge base covers a handful of claims and is not a source of truth.
- Verification understands only the patterns the rules implement: keyword presence,
  comparable numbers in a recognised unit, listed alternatives, and negations.
- Neither component should be used for real-world fact checking.
            """
        )


def render_sidebar(engine: eng.AnalysisEngine) -> None:
    """Model status and retraining controls."""
    metrics = engine.metrics
    with st.sidebar:
        st.header("Model status")
        if engine.trained_now:
            st.success("No saved model was found, so it was trained automatically.")

        if metrics:
            st.metric(
                "Cross-validated accuracy",
                f"{metrics.get('cross_validation_accuracy', 0):.1%}",
                help="Topic-classification skill on the sample dataset only.",
            )
            st.write(f"**Training rows:** {metrics.get('train_size', 'n/a')}")
            labels = metrics.get("labels", [])
            st.caption("Labels: " + ", ".join(labels) if labels else "Labels: not available")
            st.caption(f"Trained at: {metrics.get('trained_at', 'unknown')}")
        else:
            st.warning("Model metrics are unavailable.")

        st.caption(
            f"Reference rows: {len(engine.knowledge_base)}"
            + (f" ({engine.knowledge_base_path.name})" if engine.knowledge_base_path else "")
        )

        if st.button("Retrain model"):
            try:
                pipeline, new_metrics = tc.train_model()
                tc.save_artifacts(pipeline, new_metrics)
                engine.pipeline = pipeline
                engine.metrics = new_metrics
                engine.trained_now = False
                st.success("Model retrained.")
                st.rerun()
            except Exception as error:
                st.error(f"Retraining failed: {error}")

        st.divider()
        st.caption(
            "Educational demo. No API keys, no paid services, no internet connection "
            "needed. All paths are relative to the project folder."
        )


def render_results(analysis: eng.Analysis, engine: eng.AnalysisEngine) -> None:
    """Draw the two result cards and the supporting expanders."""
    st.divider()
    st.subheader("2. Two separate results")
    confidence_column, verification_column = st.columns(2, gap="large")

    with confidence_column:
        with st.container(border=True):
            card_header(
                "🤖 Model confidence",
                "What topic does the model think this is, and how sure is it?",
            )
            render_confidence_card(analysis)

    with verification_column:
        with st.container(border=True):
            card_header(
                "📚 Reference verification",
                "What does the local reference knowledge base actually say?",
            )
            render_verification_card(analysis, engine)

    st.divider()
    with st.expander("Why these two results are independent", expanded=True):
        st.markdown(
            """
- The classifier only reads **words**. It learned from a small labelled dataset and
  outputs probabilities over the five topic labels.
- The verifier only reads **rules plus `knowledge_base.csv`**. It never sees the
  classifier's output, and the classifier never sees the verifier's verdict.
- A verdict of **Supported** means "this reference row agrees with the text", not
  "this statement is proven true".
- **Insufficient Evidence** is the honest default: no matching reference row, no
  comparable number, an unknown unit, or an ambiguous match all lead here.
            """
        )

    render_reference_facts(engine, analysis)
    render_project_information(engine)

    st.caption(
        "Educational demo. The knowledge base is intentionally small, the training data is "
        "synthetic, and neither component should be used for real-world fact checking."
    )


def render_welcome(engine: eng.AnalysisEngine) -> None:
    """Explain the two upcoming cards before anything has been analysed."""
    st.subheader("How to read the results")
    confidence_column, verification_column = st.columns(2, gap="large")
    with confidence_column:
        with st.container(border=True):
            card_header("🤖 Model confidence", "A statistical guess about the topic")
            st.markdown(
                f"""
- **Predicted topic** - one of {", ".join(engine.labels) or "the training labels"}.
- **Confidence** - the model's probability for that label.
- **Probability chart** - how the score is spread across all topics.
- It never checks facts, so 99% confidence is not evidence of truth.
                """
            )
    with verification_column:
        with st.container(border=True):
            card_header("📚 Reference verification", "A rule-based lookup in a tiny local file")
            st.markdown(
                """
- **Supported** - a reference row agrees with the claim.
- **Contradicted** - it disagrees, or the claim is explicitly negated.
- **Insufficient Evidence** - the honest default when nothing justifies a verdict.
- The model's opinion is never used in this box.
                """
            )


def main() -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    model, load_error = get_engine()

    st.title("🔎 AI Confidence and Verification System")
    st.markdown(
        "Enter a short statement. The app shows **two independent signals**: what the "
        "classifier thinks the topic is (with a confidence score), and whether the small "
        "reference knowledge base supports, contradicts, or cannot judge the claim."
    )

    st.warning(
        "**Confidence is not truth.** A 99% confident prediction can still be wrong, and a "
        "topic label never proves a fact. Verification is a separate rule-based check "
        "against a tiny local reference file.",
        icon="⚠️",
    )

    if model is None:
        st.error("The classifier or knowledge base could not be loaded.")
        st.code(load_error or "Unknown error", language="text")
        st.info(
            "Make sure `data/training_data.csv` and `knowledge_base.csv` (or "
            "`data/knowledge_base.csv`) exist, then run `python train_model.py` in the "
            "project folder and reload this page."
        )
        st.stop()

    render_sidebar(model)

    with st.container(border=True):
        st.subheader("1. Enter a statement")
        example = st.selectbox(
            "Or pick an example to try:",
            ["(choose an example)"] + EXAMPLES,
            index=0,
        )
        if example != "(choose an example)" and st.session_state.get("applied_example") != example:
            st.session_state["claim_text"] = example
            st.session_state["applied_example"] = example

        claim = st.text_area(
            "Statement to analyse",
            key="claim_text",
            height=110,
            placeholder="e.g. Water boils at 100 degrees Celsius at sea level.",
        )

        analyse_clicked = st.button(
            "Analyse statement", type="primary", key="analyse_statement"
        )

    if not analyse_clicked:
        st.caption("Press **Analyse statement** to see the model prediction and the verification verdict.")
        previous = st.session_state.get("last_analysis")
        if previous is None:
            render_welcome(model)
            st.divider()
            render_reference_facts(model, None)
            render_project_information(model)
        else:
            st.caption("Showing the previous result. Press **Analyse statement** to refresh it.")
            render_results(previous, model)
        return

    if not claim or not claim.strip():
        st.warning("Please type a statement first.")
        return

    analysis = model.analyse(claim)
    st.session_state["last_analysis"] = analysis
    render_results(analysis, model)


if __name__ == "__main__":
    main()
