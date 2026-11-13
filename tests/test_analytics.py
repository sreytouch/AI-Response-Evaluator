"""Analytics handle empty data, ties, and chronological aggregation."""

import pandas as pd
import pytest

from src.analytics import dashboard_summary, preference_counts, rubric_averages, score_trend
from src.evaluation import RUBRIC


def example(preference: str, timestamp: str, score_a: int, score_b: int) -> dict:
    record = {
        "preferred_response": preference,
        "timestamp": timestamp,
        "A_average_score": score_a,
        "B_average_score": score_b,
    }
    for key in RUBRIC:
        record[f"A_{key}"] = score_a
        record[f"B_{key}"] = score_b
    return record


def test_empty_dashboard_has_stable_categories():
    frame = pd.DataFrame()
    assert dashboard_summary(frame)["total"] == 0
    assert preference_counts(frame).to_dict() == {"A": 0, "B": 0, "Tie": 0}
    assert rubric_averages(frame).shape == (6, 2)
    assert score_trend(frame).empty


def test_ties_are_included_in_percentages_and_averages():
    frame = pd.DataFrame([
        example("A", "2026-01-01T12:00:00+00:00", 5, 2),
        example("A", "2026-01-02T12:00:00+00:00", 4, 3),
        example("Tie", "2026-01-03T12:00:00+00:00", 3, 3),
    ])
    summary = dashboard_summary(frame)
    assert summary["percent_A"] == pytest.approx(200 / 3)
    assert summary["percent_B"] == 0
    assert summary["percent_Tie"] == pytest.approx(100 / 3)
    assert summary["average_A"] == 4
    assert preference_counts(frame).to_dict() == {"A": 2, "B": 0, "Tie": 1}
    averages = rubric_averages(frame)
    assert list(averages["A"]) == [4] * 6
    assert averages["B"].iloc[0] == pytest.approx(8 / 3)


def test_trend_sorts_and_groups_same_timestamp():
    frame = pd.DataFrame([
        example("B", "2026-01-02T12:00:00+00:00", 1, 5),
        example("A", "2026-01-01T12:00:00+00:00", 5, 1),
        example("Tie", "2026-01-01T12:00:00+00:00", 3, 3),
    ])
    trend = score_trend(frame)
    assert len(trend) == 2
    assert trend.index.is_monotonic_increasing
    assert trend.iloc[0]["A_average_score"] == 4
    assert trend.iloc[0]["B_average_score"] == 2
    assert str(trend.index.tz) == "UTC"
