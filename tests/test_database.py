"""Exercise durable SQLite behavior using isolated temporary database files."""

import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.database import get_evaluation, get_evaluations, init_db, save_evaluation
from src.evaluation import EVALUATION_COLUMNS, RUBRIC, ValidationError, make_evaluation


@pytest.fixture
def record():
    return make_evaluation(
        prompt="Explain gravity.", constraints="Three sentences.",
        response_a="Gravity pulls masses toward one another.",
        response_b="Gravity makes everything float.", preferred_response="A",
        rationale="A is accurate.", scores_a=dict.fromkeys(RUBRIC, 5),
        scores_b=dict.fromkeys(RUBRIC, 2),
    )


def test_initialization_creates_directory_schema_and_empty_history(tmp_path):
    path = tmp_path / "nested" / "evaluations.db"
    init_db(path)
    init_db(path)
    assert path.is_file()
    frame = get_evaluations(path)
    assert frame.empty
    assert list(frame.columns) == EVALUATION_COLUMNS
    with sqlite3.connect(path) as connection:
        columns = connection.execute("PRAGMA table_info(evaluations)").fetchall()
    assert [column[1] for column in columns] == EVALUATION_COLUMNS


def test_omitted_path_uses_configured_database(tmp_path, monkeypatch, record):
    path = tmp_path / "configured.db"
    monkeypatch.setattr("src.config.get_settings", lambda: SimpleNamespace(db_path=path))
    assert save_evaluation(record)
    assert get_evaluation(record["evaluation_id"]) == record
    assert len(get_evaluations()) == 1
    assert path.is_file()


def test_save_initializes_database_and_retrieves_all_fields(tmp_path, record):
    path = tmp_path / "feedback.db"
    assert save_evaluation(record, path) is True
    assert get_evaluation(record["evaluation_id"], path) == record
    frame = get_evaluations(path)
    assert list(frame.columns) == EVALUATION_COLUMNS
    assert frame.to_dict("records") == [record]
    assert get_evaluation(str(uuid4()), path) is None


def test_duplicate_save_is_idempotent_and_does_not_replace_feedback(tmp_path, record):
    path = tmp_path / "feedback.db"
    assert save_evaluation(record, path) is True
    assert save_evaluation(record, path) is False
    changed = {**record, "rationale": "A changed rationale."}
    assert save_evaluation(changed, path) is False
    assert len(get_evaluations(path)) == 1
    assert get_evaluation(record["evaluation_id"], path)["rationale"] == record["rationale"]


def test_history_is_newest_first_even_if_inserted_out_of_order(tmp_path, record):
    path = tmp_path / "feedback.db"
    newer = {**record, "timestamp": "2026-11-12T21:00:00.000000+00:00"}
    older = {**record, "evaluation_id": str(uuid4()), "timestamp": "2026-11-11T21:00:00.000000+00:00"}
    save_evaluation(newer, path)
    save_evaluation(older, path)
    assert get_evaluations(path)["evaluation_id"].tolist() == [newer["evaluation_id"], older["evaluation_id"]]


def test_validation_happens_before_database_write(tmp_path, record):
    path = tmp_path / "feedback.db"
    with pytest.raises(ValidationError):
        save_evaluation({**record, "rationale": "  "}, path)
    assert not path.exists()
    with pytest.raises(ValidationError):
        save_evaluation({**record, "A_correctness": 6}, path)
    with pytest.raises(ValidationError):
        save_evaluation({**record, "A_average_score": 1}, path)


def test_sqlite_constraints_also_reject_invalid_scores_and_preferences(tmp_path, record):
    path = tmp_path / "feedback.db"
    save_evaluation(record, path)
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE evaluations SET A_correctness = 6")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE evaluations SET B_style = 1.5")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE evaluations SET preferred_response = 'C'")
    assert get_evaluation(record["evaluation_id"], path) == record


def test_sql_input_is_stored_as_text_and_does_not_change_schema(tmp_path, record):
    path = tmp_path / "feedback.db"
    record["prompt"] = "Robert'); DROP TABLE evaluations; --"
    assert save_evaluation(record, path)
    assert get_evaluation(record["evaluation_id"], path)["prompt"] == record["prompt"]
    assert get_evaluation("' OR 1=1 --", path) is None
    assert len(get_evaluations(path)) == 1
