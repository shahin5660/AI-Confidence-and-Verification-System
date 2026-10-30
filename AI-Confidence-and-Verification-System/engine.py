"""Single entry point that joins the classifier and the reference verifier.

The Streamlit UI (`app.py`) should not have to know how the model is trained or
where the knowledge base lives. It asks this module for one analysis and gets
back two clearly separated results:

* **model confidence** - from TF-IDF + logistic regression (`text_classifier.py`)
* **reference verification** - rule based, over `knowledge_base.csv`
  (`verification.py`)

Nothing here changes either component's behaviour: this is glue, plus the file
lookup for the reference data, so the two halves can never be confused with one
another.

Command-line use (handy for a quick check without Streamlit)::

    python engine.py "Water boils at 90 degrees Celsius at sea level."
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import text_classifier as tc
import verification as vf

BASE_DIR = Path(__file__).resolve().parent

# The reference data is shipped in data/knowledge_base.csv. A copy placed in the
# project root is also accepted, so either layout works on a fresh checkout.
KNOWLEDGE_BASE_CANDIDATES = (
    BASE_DIR / "knowledge_base.csv",
    BASE_DIR / "data" / "knowledge_base.csv",
)


def find_knowledge_base_path() -> Path:
    """Return the first existing knowledge-base file, or raise a clear error."""
    for candidate in KNOWLEDGE_BASE_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(path) for path in KNOWLEDGE_BASE_CANDIDATES)
    raise FileNotFoundError(
        "Knowledge base not found. Looked for:\n  " + searched
    )


def load_knowledge_base(path: str | Path | None = None) -> pd.DataFrame:
    """Load the reference knowledge base from an explicit or discovered path."""
    resolved = Path(path) if path is not None else find_knowledge_base_path()
    return vf.load_knowledge_base(resolved)


@dataclass
class Analysis:
    """Both signals for one statement, kept side by side and never merged."""

    text: str
    prediction: tc.Prediction
    verification: vf.VerificationResult

    @property
    def topic(self) -> str:
        return self.prediction.label

    @property
    def confidence(self) -> float:
        return self.prediction.confidence

    @property
    def verdict(self) -> str:
        return self.verification.verdict

    @property
    def has_reference_match(self) -> bool:
        return bool(self.verification.matched_id)

    def to_dict(self) -> dict:
        """Plain-data view, used by the CLI and useful for logging or export."""
        return {
            "text": self.text,
            "predicted_topic": self.prediction.label,
            "confidence": round(self.prediction.confidence, 4),
            "probabilities": {
                label: round(value, 4)
                for label, value in sorted(
                    self.prediction.probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "verification": {
                "verdict": self.verification.verdict,
                "reason": self.verification.reason,
                "matched_id": self.verification.matched_id,
                "topic": self.verification.topic,
                "statement": self.verification.statement,
                "source": self.verification.source,
                "matched_keywords": list(self.verification.matched_keywords),
            },
        }


class AnalysisEngine:
    """A loaded classifier plus a loaded knowledge base."""

    def __init__(
        self,
        pipeline: tc.Pipeline,
        knowledge_base: pd.DataFrame,
        metrics: dict | None = None,
        trained_now: bool = False,
        knowledge_base_path: Path | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.knowledge_base = knowledge_base
        self.metrics = metrics or {}
        self.trained_now = trained_now
        self.knowledge_base_path = knowledge_base_path

    def analyse(self, text: str) -> Analysis:
        """Run both signals on one statement.

        Raises ``ValueError`` for blank text; the UI checks first and shows a
        friendly warning instead.
        """
        if not text or not text.strip():
            raise ValueError("Text must be a non-empty string.")
        clean = text.strip()
        return Analysis(
            text=clean,
            prediction=tc.predict_with_confidence(self.pipeline, clean),
            verification=vf.verify_text(clean, self.knowledge_base),
        )

    @property
    def labels(self) -> list[str]:
        return list(getattr(self.pipeline, "classes_", []))

    def reference_row(self, row_id: str | None) -> pd.Series | None:
        """Look up one reference row for display, or None when it is absent."""
        if not row_id:
            return None
        matches = self.knowledge_base[self.knowledge_base["id"] == row_id]
        return None if matches.empty else matches.iloc[0]


def load_engine(
    data_path: str | Path = tc.DATA_PATH,
    model_path: str | Path = tc.MODEL_PATH,
    metrics_path: str | Path = tc.METRICS_PATH,
    knowledge_base_path: str | Path | None = None,
) -> AnalysisEngine:
    """Load the model (training it if needed) and the reference knowledge base."""
    pipeline, metrics, trained_now = tc.ensure_model(data_path, model_path, metrics_path)
    resolved_kb = Path(knowledge_base_path) if knowledge_base_path else find_knowledge_base_path()
    knowledge_base = load_knowledge_base(resolved_kb)
    return AnalysisEngine(
        pipeline=pipeline,
        knowledge_base=knowledge_base,
        metrics=metrics,
        trained_now=trained_now,
        knowledge_base_path=resolved_kb,
    )


def analyse(text: str, engine: AnalysisEngine) -> Analysis:
    """Convenience wrapper: ``engine.analyse`` as a free function."""
    return engine.analyse(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="statement to analyse")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the analysis as JSON instead of a short summary",
    )
    args = parser.parse_args()

    analysis = load_engine().analyse(args.text)
    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
        return

    print(f"Model confidence : {analysis.topic} ({analysis.confidence:.1%})")
    print(f"Runner-up        : {analysis.prediction.runner_up[0]} "
          f"({analysis.prediction.runner_up[1]:.1%})")
    print(f"Verification     : {analysis.verdict}")
    print(f"Explanation      : {analysis.verification.reason}")
    if analysis.has_reference_match:
        print(f"Reference row    : {analysis.verification.matched_id} "
              f"({analysis.verification.topic})")
        print(f"Source           : {analysis.verification.source}")


if __name__ == "__main__":
    main()
