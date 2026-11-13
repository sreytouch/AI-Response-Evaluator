"""Configuration tests use a temporary .env and never need a real API key."""

import pytest

from src import config
from src.config import ConfigurationError, get_settings

ENV_NAMES = (
    "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_USE_TEMPERATURE",
    "OPENAI_TEMPERATURE_A", "OPENAI_TEMPERATURE_B", "OPENAI_MAX_OUTPUT_TOKENS",
    "EVALUATOR_DB_PATH",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_a_key(tmp_path):
    settings = get_settings()
    assert settings.api_key is None
    assert settings.model == "gpt-4.1-mini"
    assert settings.use_temperature is True
    assert settings.temperature_a == 0.4
    assert settings.temperature_b == 0.9
    assert settings.db_path == tmp_path / "data" / "evaluations.db"


def test_env_is_project_relative_and_process_environment_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=example-test-key\nOPENAI_MODEL=file-model\n"
        "EVALUATOR_DB_PATH=custom/evaluations.db\nOPENAI_USE_TEMPERATURE=false\n"
    )
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    monkeypatch.chdir(tmp_path.parent)
    settings = get_settings()
    assert settings.api_key == "example-test-key"
    assert settings.model == "process-model"
    assert settings.use_temperature is False
    assert settings.db_path == tmp_path / "custom" / "evaluations.db"
    assert "example-test-key" not in repr(settings)


def test_file_changes_are_read_without_mutating_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_MODEL=first-model\n")
    assert get_settings().model == "first-model"
    env.write_text("OPENAI_MODEL=second-model\n")
    assert get_settings().model == "second-model"
    monkeypatch.setenv("OPENAI_API_KEY", "")
    env.write_text("OPENAI_API_KEY=from-file\n")
    assert get_settings().api_key is None


def test_example_key_is_treated_as_missing(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=your_api_key_here\n")
    assert get_settings().api_key is None


def test_absolute_database_path(tmp_path, monkeypatch):
    expected = tmp_path / "elsewhere" / "records.sqlite"
    monkeypatch.setenv("EVALUATOR_DB_PATH", str(expected))
    assert get_settings().db_path == expected


@pytest.mark.parametrize(
    "name,value",
    [
        ("OPENAI_MODEL", " "),
        ("OPENAI_USE_TEMPERATURE", "sometimes"),
        ("OPENAI_TEMPERATURE_A", "warm"),
        ("OPENAI_TEMPERATURE_A", "nan"),
        ("OPENAI_TEMPERATURE_B", "inf"),
        ("OPENAI_TEMPERATURE_B", "2.1"),
        ("OPENAI_MAX_OUTPUT_TOKENS", "1024.5"),
        ("OPENAI_MAX_OUTPUT_TOKENS", "0"),
        ("EVALUATOR_DB_PATH", ""),
    ],
)
def test_invalid_configuration_has_actionable_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        get_settings()
