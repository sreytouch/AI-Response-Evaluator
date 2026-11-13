"""Mocked SDK tests: these tests cannot send requests to OpenAI."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
    InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError,
)

from src import openai_client
from src.config import Settings
from src.openai_client import GenerationError, generate_pair


def response(text="A complete answer", status="completed", refusal=False):
    content = [SimpleNamespace(type="refusal")] if refusal else []
    return SimpleNamespace(
        output_text=text, status=status,
        output=[SimpleNamespace(content=content)],
    )


@pytest.fixture
def fake_sdk(monkeypatch):
    constructor = MagicMock()
    client = constructor.return_value.__enter__.return_value
    client.responses.create.side_effect = [response("Answer A"), response("Answer B")]
    monkeypatch.setattr(openai_client, "OpenAI", constructor)
    return constructor, client


@pytest.fixture
def settings():
    return Settings(api_key="fake-key-for-tests-only")


def test_pair_uses_independent_requests_and_respects_configuration(fake_sdk, settings):
    constructor, client = fake_sdk
    assert generate_pair("Explain gravity", "Use three sentences", settings) == ("Answer A", "Answer B")
    constructor.assert_called_once_with(api_key=settings.api_key, timeout=30.0, max_retries=1)
    calls = client.responses.create.call_args_list
    assert len(calls) == 2
    first, second = [call.kwargs for call in calls]
    assert first["input"] == second["input"]
    assert "Explain gravity" in first["input"][0]["content"]
    assert "Use three sentences" in first["input"][0]["content"]
    assert first["instructions"] != second["instructions"]
    assert "Answer A" not in str(second)
    assert "previous_response_id" not in first and "previous_response_id" not in second
    assert first["temperature"] == 0.4 and second["temperature"] == 0.9
    assert first["max_output_tokens"] == 1024
    assert first["model"] == settings.model
    assert first["store"] is False and second["store"] is False


def test_temperature_can_be_disabled(fake_sdk, settings):
    _, client = fake_sdk
    generate_pair("A task", settings=settings, use_temperature=False)
    assert all("temperature" not in call.kwargs for call in client.responses.create.call_args_list)


def test_temperature_overrides_are_used(fake_sdk, settings):
    _, client = fake_sdk
    generate_pair("A task", settings=settings, temperature_a=0.1, temperature_b=1.4)
    assert [c.kwargs["temperature"] for c in client.responses.create.call_args_list] == [0.1, 1.4]


@pytest.mark.parametrize("value", [-1, 2.1, float("nan"), True, "hot"])
def test_invalid_temperature_never_calls_api(fake_sdk, settings, value):
    constructor, _ = fake_sdk
    with pytest.raises(GenerationError, match="Temperature"):
        generate_pair("A task", settings=settings, temperature_a=value)
    constructor.assert_not_called()


@pytest.mark.parametrize("key", [None, "", "your_api_key_here"])
def test_missing_key_has_setup_instructions(fake_sdk, key):
    constructor, _ = fake_sdk
    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        generate_pair("A task", settings=Settings(api_key=key))
    constructor.assert_not_called()


def test_empty_prompt_never_calls_api(fake_sdk, settings):
    constructor, _ = fake_sdk
    with pytest.raises(GenerationError, match="Enter a prompt"):
        generate_pair("  ", settings=settings)
    constructor.assert_not_called()


def test_duplicate_b_is_retried_independently_once(fake_sdk, settings):
    _, client = fake_sdk
    client.responses.create.side_effect = [response("Same"), response("Same"), response("Different")]
    assert generate_pair("A task", settings=settings) == ("Same", "Different")
    calls = client.responses.create.call_args_list
    assert len(calls) == 3
    assert calls[2].kwargs["input"] == calls[0].kwargs["input"]
    assert "Same" not in str(calls[2].kwargs)
    assert calls[2].kwargs["instructions"] != calls[1].kwargs["instructions"]


def test_persistent_duplicates_are_not_fabricated(fake_sdk, settings):
    _, client = fake_sdk
    client.responses.create.side_effect = [response("Same")] * 3
    with pytest.raises(GenerationError, match="identical answers"):
        generate_pair("A task", settings=settings)
    assert client.responses.create.call_count == 3


@pytest.mark.parametrize(
    "answer,match",
    [(response("", "completed"), "no answer text"),
     (response("Partial", "incomplete"), "did not finish"),
     (response("Declined", refusal=True), "declined")],
)
def test_unusable_second_response_does_not_return_a_partial_pair(fake_sdk, settings, answer, match):
    _, client = fake_sdk
    client.responses.create.side_effect = [response("Valid A"), answer]
    with pytest.raises(GenerationError, match=match):
        generate_pair("A task", settings=settings)


@pytest.mark.parametrize(
    "error_class,status,match",
    [(AuthenticationError, 401, "API key"),
     (RateLimitError, 429, "rate or usage"),
     (BadRequestError, 400, "temperature"),
     (NotFoundError, 404, "unavailable"),
     (PermissionDeniedError, 403, "unavailable"),
     (InternalServerError, 500, "Try again later")],
)
def test_api_errors_are_safe_and_actionable(fake_sdk, settings, error_class, status, match):
    _, client = fake_sdk
    http_response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    client.responses.create.side_effect = error_class(
        "unsafe raw error containing fake-key-for-tests-only", response=http_response, body=None
    )
    with pytest.raises(GenerationError, match=match) as exc:
        generate_pair("A task", settings=settings)
    assert settings.api_key not in str(exc.value)
    assert "unsafe raw error" not in str(exc.value)


@pytest.mark.parametrize("kind,match", [(APIConnectionError, "connection"), (APITimeoutError, "too long")])
def test_connection_errors(fake_sdk, settings, kind, match):
    _, client = fake_sdk
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    client.responses.create.side_effect = kind(request=request)
    with pytest.raises(GenerationError, match=match):
        generate_pair("A task", settings=settings)
