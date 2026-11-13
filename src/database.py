"""Small, file-backed SQLite storage with validated, retry-safe inserts."""

from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from src.evaluation import EVALUATION_COLUMNS, RUBRIC, validate_evaluation


DatabasePath = str | Path | None
def _score_definitions(prefix: str) -> list[str]:
    definitions = [
        f'"{prefix}_{category}" INTEGER NOT NULL '
        f'CHECK (typeof("{prefix}_{category}") = \'integer\' AND "{prefix}_{category}" BETWEEN 1 AND 5)'
        for category in RUBRIC
    ]
    definitions.append(
        f'"{prefix}_average_score" REAL NOT NULL '
        f'CHECK ("{prefix}_average_score" BETWEEN 1 AND 5)'
    )
    return definitions


_SCORE_DEFINITIONS = ",\n".join(_score_definitions("A") + _score_definitions("B"))
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL CHECK (datetime(timestamp) IS NOT NULL),
    prompt TEXT NOT NULL CHECK (length(trim(prompt)) > 0),
    constraints TEXT NOT NULL,
    response_a TEXT NOT NULL CHECK (length(trim(response_a)) > 0),
    response_b TEXT NOT NULL CHECK (length(trim(response_b)) > 0),
    preferred_response TEXT NOT NULL CHECK (preferred_response IN ('A', 'B', 'Tie')),
    rationale TEXT NOT NULL CHECK (length(trim(rationale)) > 0),
    {_SCORE_DEFINITIONS}
)
"""
_COLUMN_SQL = ", ".join(f'"{column}"' for column in EVALUATION_COLUMNS)


def _resolve_path(db_path: DatabasePath) -> Path:
    if db_path is None:
        from src.config import get_settings

        db_path = get_settings().db_path
    if str(db_path) == ":memory:":
        raise ValueError("Use a file path so evaluations persist across connections.")
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(db_path: DatabasePath = None) -> None:
    """Create the database directory and schema if needed; keep existing data."""
    with closing(sqlite3.connect(_resolve_path(db_path), timeout=30)) as connection:
        with connection:
            connection.execute(_SCHEMA)


def save_evaluation(record: Mapping[str, Any], db_path: DatabasePath = None) -> bool:
    """Insert validated feedback, returning False if its ID was already saved.

    A duplicate ID never overwrites earlier feedback. Other validation or SQLite
    errors are raised so the interface can display an actionable message.
    """
    validated = validate_evaluation(record)
    init_db(db_path)
    placeholders = ", ".join("?" for _ in EVALUATION_COLUMNS)
    query = (
        f"INSERT INTO evaluations ({_COLUMN_SQL}) VALUES ({placeholders}) "
        "ON CONFLICT(evaluation_id) DO NOTHING"
    )
    with closing(sqlite3.connect(_resolve_path(db_path), timeout=30)) as connection:
        with connection:
            cursor = connection.execute(query, [validated[column] for column in EVALUATION_COLUMNS])
            return cursor.rowcount == 1


def get_evaluations(db_path: DatabasePath = None) -> pd.DataFrame:
    """Read every evaluation in the documented column order, newest first."""
    init_db(db_path)
    with closing(sqlite3.connect(_resolve_path(db_path), timeout=30)) as connection:
        return pd.read_sql_query(
            f"SELECT {_COLUMN_SQL} FROM evaluations ORDER BY timestamp DESC, rowid DESC",
            connection,
        )


def get_evaluation(evaluation_id: str, db_path: DatabasePath = None) -> dict[str, Any] | None:
    """Return one complete record, or None when its ID does not exist."""
    init_db(db_path)
    with closing(sqlite3.connect(_resolve_path(db_path), timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            f"SELECT {_COLUMN_SQL} FROM evaluations WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
        return dict(row) if row is not None else None
