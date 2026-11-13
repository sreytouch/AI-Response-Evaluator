"""Read configuration from the project .env file and process environment."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    """A configuration value needs to be corrected before the app can use it."""


@dataclass(frozen=True)
class Settings:
    """Application configuration. The API key is excluded from representations."""

    api_key: str | None = field(default=None, repr=False)
    model: str = "gpt-4.1-mini"
    use_temperature: bool = True
    temperature_a: float = 0.4
    temperature_b: float = 0.9
    max_output_tokens: int = 1024
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "evaluations.db")


def _temperature(value: str, name: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number between 0 and 2.") from exc
    if not math.isfinite(number) or not 0 <= number <= 2:
        raise ConfigurationError(f"{name} must be a number between 0 and 2.")
    return number


def get_settings() -> Settings:
    """Read fresh settings; process environment values take priority over .env.

    Reading instead of mutating os.environ lets changes to .env take effect on
    the next Streamlit rerun, without retaining old file values in the process.
    """
    file_values = dotenv_values(PROJECT_ROOT / ".env")
    values = {**file_values, **os.environ}

    def read(name: str, default: str) -> str:
        value = values.get(name, default)
        return str(value).strip() if value is not None else ""

    key = read("OPENAI_API_KEY", "")
    if key in {"", "your_api_key_here"}:
        key = None
    model = read("OPENAI_MODEL", "gpt-4.1-mini")
    if not model:
        raise ConfigurationError("OPENAI_MODEL must contain a model name.")
    use_temperature = read("OPENAI_USE_TEMPERATURE", "true").lower()
    if use_temperature not in {"true", "false", "1", "0", "yes", "no"}:
        raise ConfigurationError("OPENAI_USE_TEMPERATURE must be true or false.")
    try:
        max_tokens = int(read("OPENAI_MAX_OUTPUT_TOKENS", "1024"))
    except ValueError as exc:
        raise ConfigurationError("OPENAI_MAX_OUTPUT_TOKENS must be an integer from 64 to 16384.") from exc
    if not 64 <= max_tokens <= 16384:
        raise ConfigurationError("OPENAI_MAX_OUTPUT_TOKENS must be an integer from 64 to 16384.")
    db_value = read("EVALUATOR_DB_PATH", "data/evaluations.db")
    if not db_value:
        raise ConfigurationError("EVALUATOR_DB_PATH must contain a database file path.")
    db_path = Path(db_value).expanduser()
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Settings(
        api_key=key,
        model=model,
        use_temperature=use_temperature in {"true", "1", "yes"},
        temperature_a=_temperature(read("OPENAI_TEMPERATURE_A", "0.4"), "OPENAI_TEMPERATURE_A"),
        temperature_b=_temperature(read("OPENAI_TEMPERATURE_B", "0.9"), "OPENAI_TEMPERATURE_B"),
        max_output_tokens=max_tokens,
        db_path=db_path.resolve(),
    )
