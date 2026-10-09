from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import tiktoken

from aide.errors import ErrorInfo
from aide.provider.errors import ModelCallError

_TOKENIZER_FORMAT_VERSION = "tiktoken-v1"
_SUPPORTED_ENCODINGS = frozenset(("o200k_base", "o200k_harmony"))


class ContextTokenizerError(ModelCallError):
    """The configured model's context encoding is unsupported or cannot load."""


def _encoding_name_for_model(model: str) -> str:
    """Return tiktoken's mapped encoding, or the local compatibility fallback."""
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    try:
        name = tiktoken.model.encoding_name_for_model(model)
    except KeyError:
        name = "o200k_base"
    if name not in _SUPPORTED_ENCODINGS:
        raise ContextTokenizerError(
            ErrorInfo(
                "model_invalid_request",
                f"Legacy model {model!r} uses unsupported context encoding {name!r}.",
            )
        )
    return name


def context_estimator_version_for_model(model: str) -> str:
    """Identify both the context measurement format and its encoding."""
    return f"{_TOKENIZER_FORMAT_VERSION}:{_encoding_name_for_model(model)}"


def estimate_context_request_tokens(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] = (),
    *,
    model: str,
) -> int:
    """Count the complete model-visible request with local ordinary-text encoding."""
    encoding = _encoding_for_name(_encoding_name_for_model(model))
    system_prompt = ""
    retained = messages
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if not isinstance(content, str):
            raise TypeError("system message content must be a string")
        system_prompt = content
        retained = messages[1:]

    components = [system_prompt]
    components.extend(_canonical_json(message) for message in retained)
    components.extend(_canonical_json(tool) for tool in tools)
    return sum(len(encoding.encode_ordinary(component)) for component in components)


def estimate_context_run_slice_tokens(
    messages: Sequence[dict[str, Any]],
    *,
    model: str,
) -> int:
    """Count only a target Run's raw User, assistant, and Tool messages."""
    if any(message.get("role") not in {"user", "assistant", "tool"} for message in messages):
        raise ValueError("run slice must contain only user, assistant, and tool messages")
    encoding = _encoding_for_name(_encoding_name_for_model(model))
    return sum(len(encoding.encode_ordinary(_canonical_json(message))) for message in messages)


def _encoding_for_name(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except Exception as error:
        raise ContextTokenizerError(
            ErrorInfo(
                "model_failed",
                f"Unable to load context encoding {name!r} using tiktoken. "
                "Check network access and tiktoken cache configuration.",
            )
        ) from error


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        sort_keys=True,
    )
