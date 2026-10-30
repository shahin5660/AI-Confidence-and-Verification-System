"""Transparent reference verification for the AI Confidence and Verification System.

This module answers a deliberately narrow question:

    "Does the small reference knowledge base in data/knowledge_base.csv
     support, contradict, or say nothing about the claim in this text?"

It is rule based and fully traceable. It never uses the machine-learning model,
so a *confident* prediction can never turn into a *Supported* verdict.

Rules (all visible in the code below):
1. A knowledge-base row is a candidate only if *every* keyword of the row
   appears in the text.
2. If several rows match equally strongly, the result is Insufficient Evidence,
   because the evidence is ambiguous.
3. For numeric rows, the text must contain a comparable number (same unit
   family, or no unit at all). A number that differs from the reference is a
   Contradicted verdict; a matching number is Supported.
4. For text rows, the expected value (e.g. "paris") means Supported, and a
   listed alternative (e.g. "berlin") means Contradicted.
5. A negated claim ("water does not boil at 100 °C") is Contradicted.
6. Anything else is Insufficient Evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.csv"

SUPPORTED = "Supported"
CONTRADICTED = "Contradicted"
INSUFFICIENT = "Insufficient Evidence"

REQUIRED_COLUMNS = [
    "id",
    "topic",
    "statement",
    "keywords",
    "expected_value",
    "unit",
    "alternatives",
    "source",
]

# Human-friendly labels used in explanations shown by the app.
UNIT_LABELS = {
    "": "",
    "c": "°C",
    "f": "°F",
    "m": "m",
    "km": "km",
    "m/s": "m/s",
    "km/s": "km/s",
    "hour": "hours",
    "percent": "%",
}

# Spellings accepted for each unit family in free text.
UNIT_TOKENS = {
    "c": ["degrees celsius", "degree celsius", "degrees centigrade", "degree centigrade", "celsius", "°c", "degrees c", "degree c"],
    "f": ["degrees fahrenheit", "degree fahrenheit", "fahrenheit", "°f", "degrees f", "degree f"],
    "m": ["metres", "meters", "metre", "meter", "m"],
    "km": ["kilometres", "kilometers", "kilometre", "kilometer", "km"],
    "m/s": ["metres per second", "meters per second", "metre per second", "meter per second", "m/s"],
    "km/s": ["kilometres per second", "kilometers per second", "kilometre per second", "kilometer per second", "km/s"],
    "hour": ["hours", "hour", "hrs", "hr", "h"],
    "percent": ["percent", "per cent", "%"],
}

# Knowledge-base unit strings mapped onto the families above.
UNIT_ALIASES = {
    "": "",
    "c": "c",
    "°c": "c",
    "celsius": "c",
    "f": "f",
    "°f": "f",
    "m": "m",
    "km": "km",
    "m/s": "m/s",
    "km/s": "km/s",
    "hour": "hour",
    "hours": "hour",
    "h": "hour",
    "percent": "percent",
    "%": "percent",
}

NUMBER_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")

NEGATION_RE = re.compile(
    r"\b(?:not|never|cannot|neither|nor|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|won't|can't|hasn't|haven't)\b"
)


@dataclass
class VerificationResult:
    """The outcome of checking one piece of text against the reference data."""

    verdict: str
    reason: str
    matched_id: str | None = None
    topic: str | None = None
    statement: str | None = None
    source: str | None = None
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class NumberMention:
    """A number found in the text, with an optional nearby unit family."""

    value: float
    start: int
    end: int
    unit_family: str | None


def load_knowledge_base(path: str | Path = KNOWLEDGE_BASE_PATH) -> pd.DataFrame:
    """Load and validate the reference knowledge base."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Knowledge base is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Knowledge base is empty.")
    if frame["id"].duplicated().any():
        duplicates = ", ".join(sorted(frame.loc[frame["id"].duplicated(), "id"]))
        raise ValueError(f"Knowledge base ids must be unique: {duplicates}")
    return frame


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #
def _stem(token: str) -> str:
    """Very small suffix stripper so 'boils', 'boiling' and 'boiled' match."""
    for suffix in ("ing", "es", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)]
            break
    if token.endswith("s") and len(token) >= 4:
        token = token[:-1]
    if token.endswith("e") and len(token) >= 5:
        token = token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    """Normalise text into lowercase stemmed words."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [_stem(token) for token in cleaned.split() if token]


def _contains_phrase(haystack: list[str], phrase: list[str]) -> bool:
    """True when `phrase` appears as consecutive words inside `haystack`."""
    if not phrase:
        return False
    size = len(phrase)
    return any(haystack[i : i + size] == phrase for i in range(len(haystack) - size + 1))


def _normalise_text(text: str) -> str:
    """Lowercase text with predictable punctuation for numeric/negation checks."""
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_negated(text: str, position: int, window: int = 40) -> bool:
    """Look for a negation word shortly before a claim in the text."""
    start = max(0, position - window)
    return bool(NEGATION_RE.search(text[start:position]))


def _phrase_position(text: str, phrase: str) -> int | None:
    """Character offset of a whole-word phrase, or None when it is absent."""
    words = [re.escape(word) for word in phrase.split() if word]
    if not words:
        return None
    pattern = r"\b" + r"\s+".join(words) + r"\b"
    match = re.search(pattern, text)
    return match.start() if match else None


def _unit_family_from_label(label: str) -> str:
    """Map a knowledge-base unit string to a unit family."""
    label = label.strip().lower()
    return UNIT_ALIASES.get(label, "")


def _unit_patterns() -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for family, tokens in UNIT_TOKENS.items():
        parts = []
        for token in sorted(tokens, key=len, reverse=True):
            escaped = re.escape(token)
            if token[0].isalnum():
                parts.append(rf"(?<![\w]){escaped}(?![\w])")
            else:
                parts.append(escaped)
        patterns[family] = re.compile(r"^\s*(?:" + "|".join(parts) + r")", re.IGNORECASE)
    return patterns


_UNIT_PATTERNS = _unit_patterns()


def _detect_unit(text: str, end: int) -> str | None:
    """Return the unit family written directly after a number, if any."""
    segment = text[end : end + 24]
    best: tuple[int, str] | None = None
    for family, pattern in _UNIT_PATTERNS.items():
        match = pattern.match(segment)
        if match:
            length = len(match.group(0).strip())
            if best is None or length > best[0]:
                best = (length, family)
    return best[1] if best else None


def _extract_numbers(text: str) -> list[NumberMention]:
    """Find numeric values in the text and try to attach a unit family."""
    mentions: list[NumberMention] = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - guarded by the regex
            continue
        mentions.append(
            NumberMention(
                value=value,
                start=match.start(1),
                end=match.end(1),
                unit_family=_detect_unit(text, match.end(1)),
            )
        )
    return mentions


def _format_value(value: str, unit_label: str) -> str:
    """Format a reference value for a human-readable explanation."""
    label = UNIT_LABELS.get(unit_label, unit_label)
    return f"{value} {label}".strip()


def _unit_compatible(entry_family: str, number_family: str | None) -> bool:
    """Can this number be compared with the reference value?

    An entry without a unit (for example a country count) only compares bare
    numbers. An entry with a unit only compares numbers that carry the same,
    recognised unit. Anything else is skipped rather than guessed, so an
    unfamiliar or different unit can never produce a false verdict.
    """
    if entry_family == "":
        return number_family is None
    return number_family == entry_family


def _numeric_tolerance(expected: float) -> float:
    """Allow small rounding differences such as 8849 m vs 8850 m."""
    return max(0.5, abs(expected) * 0.02)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _insufficient(reason: str, row: pd.Series | None = None) -> VerificationResult:
    result = VerificationResult(verdict=INSUFFICIENT, reason=reason)
    if row is not None:
        result.matched_id = row["id"]
        result.topic = row["topic"]
        result.statement = row["statement"]
        result.source = row["source"]
    return result


def _check_numeric(
    text: str,
    row: pd.Series,
    matched_keywords: list[str],
) -> VerificationResult:
    raw_text = _normalise_text(text)
    expected = float(row["expected_value"])
    entry_family = _unit_family_from_label(row["unit"])

    comparable = [
        mention
        for mention in _extract_numbers(raw_text)
        if _unit_compatible(entry_family, mention.unit_family)
    ]
    if not comparable:
        return _insufficient(
            "The text does not state a number that can be compared with this reference row.",
            row,
        )

    tolerance = _numeric_tolerance(expected)
    matching = [mention for mention in comparable if abs(mention.value - expected) <= tolerance]
    reference_text = _format_value(row["expected_value"], entry_family)

    if matching:
        if any(not _is_negated(raw_text, mention.start) for mention in matching):
            return VerificationResult(
                verdict=SUPPORTED,
                reason=f"The text states {reference_text}, which matches the reference value.",
                matched_id=row["id"],
                topic=row["topic"],
                statement=row["statement"],
                source=row["source"],
                matched_keywords=matched_keywords,
            )
        return VerificationResult(
            verdict=CONTRADICTED,
            reason=f"The text negates the reference value of {reference_text}.",
            matched_id=row["id"],
            topic=row["topic"],
            statement=row["statement"],
            source=row["source"],
            matched_keywords=matched_keywords,
        )

    stated = comparable[0]
    stated_text = _format_value(f"{stated.value:g}", stated.unit_family or entry_family)
    return VerificationResult(
        verdict=CONTRADICTED,
        reason=(
            f"The text states {stated_text}, while this reference row gives "
            f"{reference_text}."
        ),
        matched_id=row["id"],
        topic=row["topic"],
        statement=row["statement"],
        source=row["source"],
        matched_keywords=matched_keywords,
    )


def _check_text_value(
    text: str,
    row: pd.Series,
    matched_keywords: list[str],
) -> VerificationResult:
    raw_text = _normalise_text(text)
    expected = row["expected_value"].strip()
    alternatives = [item.strip() for item in row["alternatives"].split(";") if item.strip()]

    expected_position = _phrase_position(raw_text, expected.lower())
    if expected_position is not None:
        if _is_negated(raw_text, expected_position):
            return VerificationResult(
                verdict=CONTRADICTED,
                reason=f"The text explicitly denies that {expected} is correct.",
                matched_id=row["id"],
                topic=row["topic"],
                statement=row["statement"],
                source=row["source"],
                matched_keywords=matched_keywords,
            )
        return VerificationResult(
            verdict=SUPPORTED,
            reason=f"The text states {expected}, which matches the reference row.",
            matched_id=row["id"],
            topic=row["topic"],
            statement=row["statement"],
            source=row["source"],
            matched_keywords=matched_keywords,
        )

    for alternative in alternatives:
        position = _phrase_position(raw_text, alternative.lower())
        if position is not None and not _is_negated(raw_text, position):
            return VerificationResult(
                verdict=CONTRADICTED,
                reason=(
                    f"The text states {alternative}, but this reference row gives {expected}."
                ),
                matched_id=row["id"],
                topic=row["topic"],
                statement=row["statement"],
                source=row["source"],
                matched_keywords=matched_keywords,
            )

    return _insufficient(
        "The text matches the topic but does not state a value this row can check.",
        row,
    )


def verify_text(text: str, knowledge_base: pd.DataFrame) -> VerificationResult:
    """Check a piece of text against the reference knowledge base."""
    if not text or not text.strip():
        return _insufficient("No text was supplied, so there is nothing to verify.")

    text_tokens = _tokens(text)
    if not text_tokens:
        return _insufficient("The text contains no words that can be matched to the reference data.")

    matches: list[tuple[int, pd.Series, list[str]]] = []
    for _, row in knowledge_base.iterrows():
        keywords = [keyword.strip() for keyword in str(row["keywords"]).split(";") if keyword.strip()]
        if not keywords:
            continue
        if all(_contains_phrase(text_tokens, _tokens(keyword)) for keyword in keywords):
            specificity = sum(len(keyword) for keyword in keywords)
            matches.append((specificity, row, keywords))

    if not matches:
        return _insufficient(
            "No reference row matched the topic of this text, so no verdict is possible."
        )

    matches.sort(key=lambda item: item[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        tied = ", ".join(row["id"] for _, row, _ in matches[:2])
        return _insufficient(
            f"The text matches more than one reference row equally well ({tied}), "
            "so the verdict is ambiguous."
        )

    _, row, keywords = matches[0]
    expected_value = str(row["expected_value"]).strip()
    if not expected_value:
        return _insufficient(
            "This reference row has no expected value configured, so it cannot produce a verdict.",
            row,
        )

    if _is_numeric_value(expected_value):
        return _check_numeric(text, row, keywords)
    return _check_text_value(text, row, keywords)


def _is_numeric_value(value: str) -> bool:
    """True when the expected value is a plain number such as 100 or 37.5."""
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", value))
