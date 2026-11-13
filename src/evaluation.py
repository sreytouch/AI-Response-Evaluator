"""Validate human feedback and prepare it for analysis or safe CSV export."""

from collections.abc import Mapping
from datetime import date, datetime, timezone
import math
from numbers import Integral, Real
from typing import Any
from uuid import UUID, uuid4

import pandas as pd


RUBRIC = {
    "correctness": {
        "label": "Correctness",
        "description": "Does the answer contain accurate information?",
    },
    "instruction_following": {
        "label": "Instruction Following",
        "description": "Did the answer satisfy all explicit user requirements?",
    },
    "relevance": {
        "label": "Relevance",
        "description": "Does the response stay focused on the request?",
    },
    "clarity": {
        "label": "Clarity",
        "description": "Is the response easy to understand?",
    },
    "completeness": {
        "label": "Completeness",
        "description": "Does it address all important parts of the request?",
    },
    "style": {
        "label": "Style",
        "description": "Is the tone and structure appropriate?",
    },
}

EVALUATION_COLUMNS = [
    "evaluation_id", "timestamp", "prompt", "constraints", "response_a",
    "response_b", "preferred_response", "rationale",
    *[f"A_{category}" for category in RUBRIC], "A_average_score",
    *[f"B_{category}" for category in RUBRIC], "B_average_score",
]
SEARCH_COLUMNS = ["prompt", "constraints", "response_a", "response_b", "rationale"]
PREFERENCES = ("A", "B", "Tie")


class ValidationError(ValueError):
    """An evaluation is incomplete or contains invalid values."""


def _validated_scores(scores: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(scores, Mapping) or set(scores) != set(RUBRIC):
        raise ValidationError("Score every rubric category, with no extra categories.")
    validated = {}
    for category in RUBRIC:
        value = scores[category]
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 5:
            raise ValidationError(f"{RUBRIC[category]['label']} must be an integer from 1 to 5.")
        validated[category] = int(value)
    return validated


def calculate_average(scores: Mapping[str, int]) -> float:
    """Return the unrounded mean of all six integer scores from 1 through 5."""
    validated = _validated_scores(scores)
    return sum(validated.values()) / len(RUBRIC)


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} is required.")
    return value


def make_evaluation(
    prompt: str,
    constraints: str | None,
    response_a: str,
    response_b: str,
    preferred_response: str,
    rationale: str,
    scores_a: Mapping[str, int],
    scores_b: Mapping[str, int],
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    """Build a complete feedback record without changing the submitted text.

    Keep the same evaluation_id when retrying a save. New evaluations receive a
    UUID automatically; timestamps are timezone-aware UTC ISO 8601 strings.
    """
    prompt = _required_text(prompt, "Prompt")
    response_a = _required_text(response_a, "Response A")
    response_b = _required_text(response_b, "Response B")
    rationale = _required_text(rationale, "Rationale")
    if constraints is None:
        constraints = ""
    if not isinstance(constraints, str):
        raise ValidationError("Constraints must be text.")
    if preferred_response not in PREFERENCES:
        raise ValidationError("Choose Response A, Response B, or Tie.")
    if evaluation_id is None:
        evaluation_id = str(uuid4())
    else:
        try:
            evaluation_id = str(UUID(evaluation_id))
        except (ValueError, AttributeError, TypeError) as error:
            raise ValidationError("Evaluation ID must be a valid UUID string.") from error

    a = _validated_scores(scores_a)
    b = _validated_scores(scores_b)
    record = {
        "evaluation_id": evaluation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "prompt": prompt,
        "constraints": constraints,
        "response_a": response_a,
        "response_b": response_b,
        "preferred_response": preferred_response,
        "rationale": rationale,
    }
    for prefix, scores in (("A", a), ("B", b)):
        record.update({f"{prefix}_{key}": value for key, value in scores.items()})
        record[f"{prefix}_average_score"] = calculate_average(scores)
    return record


def validate_evaluation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete record at the database boundary, including averages."""
    if not isinstance(record, Mapping) or set(record) != set(EVALUATION_COLUMNS):
        raise ValidationError("An evaluation must contain exactly the documented fields.")
    if not isinstance(record["evaluation_id"], str):
        raise ValidationError("A saved evaluation must have a UUID string as its ID.")
    validated = make_evaluation(
        prompt=record["prompt"], constraints=record["constraints"],
        response_a=record["response_a"], response_b=record["response_b"],
        preferred_response=record["preferred_response"], rationale=record["rationale"],
        scores_a={key: record[f"A_{key}"] for key in RUBRIC},
        scores_b={key: record[f"B_{key}"] for key in RUBRIC},
        evaluation_id=record["evaluation_id"],
    )
    timestamp = record["timestamp"]
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("A UTC timezone is required.")
    except (ValueError, AttributeError, TypeError) as error:
        raise ValidationError("Timestamp must be an ISO 8601 date and time in UTC.") from error
    validated["timestamp"] = parsed.isoformat(timespec="microseconds")
    for prefix in ("A", "B"):
        supplied = record[f"{prefix}_average_score"]
        expected = validated[f"{prefix}_average_score"]
        if isinstance(supplied, bool) or not isinstance(supplied, Real) or not math.isclose(
            supplied, expected, rel_tol=0, abs_tol=1e-9
        ):
            raise ValidationError(f"Response {prefix}'s average must match its rubric scores.")
    return validated


def _as_date(value: date | str | None, label: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError) as error:
        raise ValidationError(f"{label} must be a date in YYYY-MM-DD format.") from error


def filter_evaluations(
    frame: pd.DataFrame,
    preferred_response: str | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Return a copy filtered by preference, inclusive UTC dates, and literal text."""
    if preferred_response is not None and preferred_response not in PREFERENCES:
        raise ValidationError("Preference filter must be A, B, Tie, or None.")
    start = _as_date(start_date, "Start date")
    end = _as_date(end_date, "End date")
    if start and end and start > end:
        raise ValidationError("Start date must be on or before end date.")
    if not isinstance(search, str):
        raise ValidationError("Search must be text.")
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index)
    if preferred_response is not None:
        mask &= frame["preferred_response"].eq(preferred_response)
    if start is not None or end is not None:
        dates = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="ISO8601").dt.date
        if start is not None:
            mask &= dates.ge(start)
        if end is not None:
            mask &= dates.le(end)
    if search:
        text_matches = pd.Series(False, index=frame.index)
        for column in SEARCH_COLUMNS:
            text_matches |= frame[column].fillna("").astype(str).str.contains(search, case=False, regex=False)
        mask &= text_matches
    return frame.loc[mask].copy()


def export_csv(frame: pd.DataFrame) -> bytes:
    """Export all columns as UTF-8 CSV without a dataframe index.

    Spreadsheet applications can execute cells beginning with =, +, -, or @.
    Such strings (including after whitespace), and strings beginning with a tab,
    carriage return, or newline, receive a leading apostrophe in this export.
    Stored data and the supplied dataframe are unchanged. Quotes, commas, and
    line breaks are preserved by standard CSV quoting.
    """
    def escape_formula(value: Any) -> Any:
        if isinstance(value, str) and (
            value.lstrip().startswith(("=", "+", "-", "@"))
            or value.startswith(("\t", "\r", "\n"))
        ):
            return "'" + value
        return value

    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(escape_formula)
    return safe.to_csv(index=False, lineterminator="\n").encode("utf-8")
