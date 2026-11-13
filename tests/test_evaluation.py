"""Feedback validation and CSV tests require no model or network access."""

import csv
from datetime import date, datetime
import io
from uuid import UUID, uuid4

import pandas as pd
import pytest

from src.evaluation import (
    EVALUATION_COLUMNS, RUBRIC, ValidationError, calculate_average, export_csv,
    filter_evaluations, make_evaluation, validate_evaluation,
)


def feedback(**changes):
    values = {
        "prompt": "Explain gravity.", "constraints": "Use simple words.",
        "response_a": "Gravity pulls objects toward one another.",
        "response_b": "Gravity makes everything float.",
        "preferred_response": "A", "rationale": "A is accurate and clear.",
        "scores_a": dict.fromkeys(RUBRIC, 5), "scores_b": dict.fromkeys(RUBRIC, 2),
    }
    values.update(changes)
    return make_evaluation(**values)


def test_calculate_average_uses_all_categories_without_rounding():
    scores = dict(zip(RUBRIC, [1, 2, 3, 4, 5, 5]))
    assert calculate_average(scores) == pytest.approx(20 / 6)


@pytest.mark.parametrize("score", [0, 6, -1, 3.0, True, "4", None])
def test_score_must_be_an_integer_between_one_and_five(score):
    scores = dict.fromkeys(RUBRIC, 3)
    scores["correctness"] = score
    with pytest.raises(ValidationError):
        calculate_average(scores)


@pytest.mark.parametrize("scores", [{}, {"correctness": 3}, {**dict.fromkeys(RUBRIC, 3), "extra": 5}])
def test_all_and_only_rubric_categories_are_required(scores):
    with pytest.raises(ValidationError):
        calculate_average(scores)


def test_record_has_exact_columns_uuid_and_utc_timestamp():
    record = feedback()
    assert list(record) == EVALUATION_COLUMNS
    assert str(UUID(record["evaluation_id"])) == record["evaluation_id"]
    assert datetime.fromisoformat(record["timestamp"]).utcoffset().total_seconds() == 0
    assert record["A_average_score"] == 5
    assert record["B_average_score"] == 2
    assert validate_evaluation(record) == record


@pytest.mark.parametrize("field", ["prompt", "response_a", "response_b", "rationale"])
@pytest.mark.parametrize("value", ["", " \n\t", None])
def test_required_text_fields_cannot_be_blank(field, value):
    with pytest.raises(ValidationError):
        feedback(**{field: value})


def test_optional_constraints_and_supplied_id_and_tie():
    identifier = str(uuid4())
    record = feedback(constraints=None, preferred_response="Tie", evaluation_id=identifier)
    assert record["constraints"] == ""
    assert record["evaluation_id"] == identifier
    assert record["preferred_response"] == "Tie"


@pytest.mark.parametrize("changes", [
    {"preferred_response": "Response A"}, {"preferred_response": ""},
    {"evaluation_id": "not-a-uuid"}, {"constraints": 12},
])
def test_invalid_preference_id_or_constraints_are_rejected(changes):
    with pytest.raises(ValidationError):
        feedback(**changes)


@pytest.mark.parametrize("changes", [
    {"A_average_score": 4.0}, {"B_average_score": float("nan")},
    {"timestamp": "2026-11-12"}, {"timestamp": "2026-11-12T14:00:00-08:00"},
    {"timestamp": None}, {"unexpected": "field"}, {"evaluation_id": None},
])
def test_complete_record_validation_checks_timestamp_averages_and_shape(changes):
    with pytest.raises(ValidationError):
        validate_evaluation({**feedback(), **changes})


@pytest.fixture
def history():
    rows = [
        feedback(prompt="Literal [test].", preferred_response="A"),
        feedback(constraints="Keep [TEST] short", preferred_response="B"),
        feedback(rationale="Both are equally clear.", preferred_response="Tie"),
    ]
    for row, timestamp in zip(rows, [
        "2026-11-11T23:59:59+00:00", "2026-11-12T00:00:00+00:00", "2026-11-12T23:59:59+00:00",
    ]):
        row["timestamp"] = timestamp
    return pd.DataFrame(rows, columns=EVALUATION_COLUMNS)


def test_filters_combine_literal_case_insensitive_search_preference_and_dates(history):
    result = filter_evaluations(
        history, preferred_response="B", start_date=date(2026, 11, 12),
        end_date="2026-11-12", search="[test]",
    )
    assert result["preferred_response"].tolist() == ["B"]
    assert len(filter_evaluations(history, search="[test]")) == 2
    assert len(filter_evaluations(history, start_date="2026-11-12", end_date="2026-11-12")) == 2
    result.loc[:, "prompt"] = "changed"
    assert "changed" not in history["prompt"].tolist()


@pytest.mark.parametrize("column", ["prompt", "constraints", "response_a", "response_b", "rationale"])
def test_search_includes_every_text_field(history, column):
    history.loc[0, column] = "A unique needle.* phrase"
    assert len(filter_evaluations(history, search="needle.*")) == 1


def test_empty_history_retains_columns_and_invalid_filters_are_explained():
    empty = pd.DataFrame(columns=EVALUATION_COLUMNS)
    assert list(filter_evaluations(empty).columns) == EVALUATION_COLUMNS
    with pytest.raises(ValidationError):
        filter_evaluations(empty, start_date="2026-11-13", end_date="2026-11-12")
    with pytest.raises(ValidationError):
        filter_evaluations(empty, start_date="not-a-date")
    with pytest.raises(ValidationError):
        filter_evaluations(empty, preferred_response="All")


def test_csv_preserves_column_order_unicode_quotes_and_multiline_text(history):
    original = 'A "quoted", café response\nwith a second line.'
    history.loc[0, "response_a"] = original
    exported = export_csv(history)
    reader = csv.DictReader(io.StringIO(exported.decode("utf-8")))
    rows = list(reader)
    assert reader.fieldnames == EVALUATION_COLUMNS
    assert rows[0]["response_a"] == original
    assert len(rows) == 3
    assert rows[0]["A_correctness"] == "5"


@pytest.mark.parametrize("value", ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "  =1+1", "\ttext", "\ntext", "\rtext"])
def test_csv_escapes_spreadsheet_formulas_without_mutating_data(value):
    frame = pd.DataFrame([feedback(prompt=value)])
    exported = next(csv.DictReader(io.StringIO(export_csv(frame).decode("utf-8"))))
    assert exported["prompt"] == "'" + value
    assert frame.loc[0, "prompt"] == value


def test_empty_export_includes_header():
    exported = export_csv(pd.DataFrame(columns=EVALUATION_COLUMNS)).decode("utf-8")
    assert exported.strip().split(",") == EVALUATION_COLUMNS
