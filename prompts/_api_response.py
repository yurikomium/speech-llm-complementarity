"""Provider-neutral helpers for preserving paid API responses.

Every response received from a hosted model is treated as an irreversible
artifact.  Callers store the JSON-safe representation on success and attach it
to errors raised while parsing or validating the response.  Failed retry
attempts are written immediately rather than being overwritten by a later
attempt.
"""

from __future__ import annotations

import json
from typing import Any, TextIO


def serialize_api_response(response: Any) -> Any:
    """Return a JSON-safe representation of an SDK response object."""
    if response is None:
        return None

    candidate = response
    if hasattr(response, "model_dump"):
        try:
            candidate = response.model_dump(mode="json")
        except TypeError:
            candidate = response.model_dump()
    elif hasattr(response, "to_json_dict"):
        candidate = response.to_json_dict()
    elif hasattr(response, "to_dict"):
        candidate = response.to_dict()

    # The round trip recursively converts SDK-specific leaves via ``str``.
    return json.loads(json.dumps(candidate, ensure_ascii=False, default=str))


class ApiResponseError(ValueError):
    """Parsing/validation error that retains the response which caused it."""

    def __init__(self, message: str, raw_output: Any):
        super().__init__(message)
        self.raw_output = raw_output


def write_jsonl_record(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    handle.flush()


def write_attempt_error(
    handle: TextIO,
    context: dict[str, Any],
    attempt: int,
    max_retries: int,
    error: Exception,
    raw_output: Any = None,
) -> None:
    """Persist one failed attempt before any retry is made."""
    retained = getattr(error, "raw_output", raw_output)
    write_jsonl_record(
        handle,
        {
            **context,
            "attempt": attempt + 1,
            "max_attempts": max_retries + 1,
            "will_retry": attempt < max_retries,
            "error": str(error),
            "raw_output": retained,
        },
    )
