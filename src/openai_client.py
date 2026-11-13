"""Generate an independent pair of answers using the official OpenAI SDK."""

import math
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from src.config import Settings, get_settings


class GenerationError(RuntimeError):
    """An answer pair could not be generated; the message is safe to show users."""


BASE_INSTRUCTIONS = (
    "Answer the user's task accurately and clearly. Respect all explicit user "
    "constraints, including length, audience, source restrictions, and requested "
    "format. The writing approach below is secondary to those requirements. "
    "Return only your answer, without describing the generation process."
)
APPROACHES = (
    "Prefer direct wording and a straightforward explanation when appropriate.",
    "Use a thoughtful alternative wording or organization when the task allows it.",
    "Find a fresh phrasing or explanatory angle, without adding unsupported facts.",
)


def _checked_temperature(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenerationError("Temperature must be a number between 0 and 2.")
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise GenerationError("Temperature must be a number between 0 and 2.")
    return float(value)


def _answer_text(response: Any) -> str:
    """Reject partial answers and refusals instead of treating them as candidates."""
    if response.status != "completed":
        raise GenerationError(
            "The model did not finish its answer. Try a shorter prompt or increase "
            "OPENAI_MAX_OUTPUT_TOKENS in .env. No new pair was created."
        )
    for item in response.output:
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "refusal":
                raise GenerationError(
                    "The model declined this request. Rephrase the prompt and try again. "
                    "No new pair was created."
                )
    text = response.output_text.strip()
    if not text:
        raise GenerationError("The model returned no answer text. Try another prompt or model.")
    return text


def generate_pair(
    prompt: str,
    constraints: str = "",
    settings: Settings | None = None,
    temperature_a: float | None = None,
    temperature_b: float | None = None,
    use_temperature: bool | None = None,
) -> tuple[str, str]:
    """Generate two candidates without giving either candidate the other's text.

    Two separate API requests use the same task and different writing approaches.
    If both answers are identical, B is retried once independently. A pair is
    returned only when both candidates are complete and different.
    """
    settings = settings or get_settings()
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationError("Enter a prompt before generating responses.")
    if not isinstance(constraints, str):
        raise GenerationError("Constraints must be text.")
    if not settings.api_key or settings.api_key == "your_api_key_here":
        raise GenerationError(
            "Add OPENAI_API_KEY to your project's .env file to generate responses. "
            "Copy .env.example to .env and replace your_api_key_here with your key."
        )
    send_temperature = settings.use_temperature if use_temperature is None else use_temperature
    if not isinstance(send_temperature, bool):
        raise GenerationError("The temperature option must be enabled or disabled.")
    temp_a = settings.temperature_a if temperature_a is None else temperature_a
    temp_b = settings.temperature_b if temperature_b is None else temperature_b
    if send_temperature:
        temp_a, temp_b = _checked_temperature(temp_a), _checked_temperature(temp_b)
    user_input = f"Task:\n{prompt.strip()}"
    if constraints.strip():
        user_input += f"\n\nAdditional constraints:\n{constraints.strip()}"

    try:
        with OpenAI(api_key=settings.api_key, timeout=30.0, max_retries=1) as client:
            def generate(approach: str, temperature: float) -> str:
                request: dict[str, Any] = {
                    "model": settings.model,
                    "instructions": f"{BASE_INSTRUCTIONS}\nWriting approach: {approach}",
                    "input": [{"role": "user", "content": user_input}],
                    "max_output_tokens": settings.max_output_tokens,
                    "store": False,
                }
                if send_temperature:
                    request["temperature"] = temperature
                return _answer_text(client.responses.create(**request))

            response_a = generate(APPROACHES[0], temp_a)
            response_b = generate(APPROACHES[1], temp_b)
            if response_a == response_b:
                response_b = generate(APPROACHES[2], temp_b)
            if response_a == response_b:
                raise GenerationError(
                    "The model returned identical answers, including on one retry. "
                    "Try a more open-ended prompt or different generation settings. "
                    "No new pair was created."
                )
            return response_a, response_b
    except AuthenticationError as exc:
        raise GenerationError("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.") from exc
    except RateLimitError as exc:
        raise GenerationError(
            "OpenAI's rate or usage limit was reached. Check your API billing and limits, "
            "then try again later."
        ) from exc
    except APITimeoutError as exc:
        raise GenerationError("OpenAI took too long to respond. Please try again.") from exc
    except APIConnectionError as exc:
        raise GenerationError("Could not connect to OpenAI. Check your connection and try again.") from exc
    except (PermissionDeniedError, NotFoundError) as exc:
        raise GenerationError(
            "The configured model is unavailable to this API key. Check OPENAI_MODEL and model access."
        ) from exc
    except BadRequestError as exc:
        raise GenerationError(
            "OpenAI rejected the generation settings. Check OPENAI_MODEL and the token limit. "
            "If your model does not support temperature, disable it in Generation settings "
            "or set OPENAI_USE_TEMPERATURE=false in .env."
        ) from exc
    except APIError as exc:
        raise GenerationError("OpenAI could not complete the request. Try again later.") from exc
