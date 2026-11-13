"""Small, reusable summaries for the human-evaluation dashboard."""

from __future__ import annotations

import pandas as pd

from src.evaluation import RUBRIC

PREFERENCES = ("A", "B", "Tie")


def preference_counts(frame: pd.DataFrame) -> pd.Series:
    """Include zero-count choices, even before any evaluations are saved."""
    if frame.empty:
        return pd.Series(0, index=PREFERENCES, name="Count", dtype=int)
    return frame["preferred_response"].value_counts().reindex(PREFERENCES, fill_value=0)


def dashboard_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize every saved evaluation, including ties in the denominator."""
    total = len(frame)
    counts = preference_counts(frame)
    result: dict[str, float | int] = {"total": total}
    for choice in PREFERENCES:
        result[f"percent_{choice}"] = 100.0 * int(counts[choice]) / total if total else 0.0
    for label in ("A", "B"):
        result[f"average_{label}"] = float(frame[f"{label}_average_score"].mean()) if total else 0.0
    return result


def rubric_averages(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per rubric category with separate A and B means."""
    return pd.DataFrame(
        {
            label: [float(frame[f"{label}_{key}"].mean()) if not frame.empty else 0.0 for key in RUBRIC]
            for label in ("A", "B")
        },
        index=[details["label"] for details in RUBRIC.values()],
    )


def score_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """Group equal UTC timestamps and order mean scores chronologically."""
    columns = ["A_average_score", "B_average_score"]
    if frame.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
    data = frame[["timestamp", *columns]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    return data.dropna(subset=["timestamp"]).groupby("timestamp")[columns].mean().sort_index()
