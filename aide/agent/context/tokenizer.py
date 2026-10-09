from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from typing import Any, cast

import tiktoken

_TOKENIZER_FORMAT_VERSION = "tiktoken-v1"
_DATA_DIRECTORY = ("tokenizer_data",)


class ContextTokenizerError(RuntimeError):
    """A bundled context tokenizer resource is missing or invalid."""


def _encoding_name_for_model(model: str) -> str:
    """Return tiktoken's mapped encoding, or the local compatibility fallback."""
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    try:
        name = tiktoken.model.encoding_name_for_model(model)
    except KeyError:
        name = "o200k_base"
    if name not in _load_manifest()["encodings"]:
        raise ContextTokenizerError(f"No bundled vocabulary is available for encoding {name!r}.")
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


@lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    try:
        manifest_path = files("aide.agent.context").joinpath(*_DATA_DIRECTORY, "manifest.json")
        manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != 1
            or not isinstance(manifest.get("encodings"), dict)
        ):
            raise ValueError("unsupported tokenizer manifest format")
        return cast(dict[str, Any], manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ContextTokenizerError(
            "The bundled context tokenizer manifest is unavailable or invalid."
        ) from error


@lru_cache(maxsize=8)
def _encoding_for_name(name: str) -> tiktoken.Encoding:
    try:
        manifest = _load_manifest()
        config = manifest["encodings"][name]
        vocabulary = config["vocabulary"]
        if vocabulary["kind"] == "tiktoken_bpe":
            contents = _read_vocabulary(vocabulary)
            mergeable_ranks = {
                base64.b64decode(token, validate=True): int(rank)
                for line in contents.splitlines()
                if line
                for token, rank in (line.split(),)
            }
        elif vocabulary["kind"] == "data_gym":
            resources = vocabulary["files"]
            mergeable_ranks = _data_gym_ranks(
                _read_vocabulary(resources["vocab_bpe"]),
                _read_vocabulary(resources["encoder_json"]),
            )
        else:
            raise ValueError("unsupported vocabulary resource kind")

        special_tokens = dict(config["special_tokens"])
        special_tokens.update(config.get("additional_special_tokens", {}))
        for token_range in config.get("special_token_ranges", ()):
            for token_id in range(token_range["start"], token_range["stop"]):
                special_tokens[token_range["pattern"].format(id=token_id)] = token_id
        return tiktoken.Encoding(
            name=config["name"],
            pat_str=config["pat_str"],
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
            explicit_n_vocab=config.get("explicit_n_vocab"),
        )
    except ContextTokenizerError:
        raise
    except Exception as error:
        raise ContextTokenizerError(
            f"The bundled vocabulary for context encoding {name!r} is unavailable or invalid."
        ) from error


def _read_vocabulary(resource: dict[str, Any]) -> bytes:
    contents = files("aide.agent.context").joinpath(*_DATA_DIRECTORY, resource["file"]).read_bytes()
    if sha256(contents).hexdigest() != resource["sha256"]:
        raise ContextTokenizerError(f"Bundled vocabulary checksum mismatch: {resource['file']}.")
    return contents


def _data_gym_ranks(vocabulary: bytes, encoder: bytes) -> dict[bytes, int]:
    # Match tiktoken's GPT-2 byte alphabet and merge priority without its disk cache.
    byte_order = [byte for byte in range(256) if chr(byte).isprintable() and chr(byte) != " "]
    byte_decoder = {chr(byte): byte for byte in byte_order}
    next_codepoint = 256
    for byte in range(256):
        if byte not in byte_order:
            byte_decoder[chr(next_codepoint)] = byte
            byte_order.append(byte)
            next_codepoint += 1
    ranks = {bytes([byte]): rank for rank, byte in enumerate(byte_order)}
    for line in vocabulary.decode("utf-8").splitlines()[1:]:
        first, second = line.split()
        token = bytes(byte_decoder[char] for char in first + second)
        ranks[token] = len(ranks)
    encoder_ranks = {
        bytes(byte_decoder[char] for char in token): rank
        for token, rank in json.loads(encoder).items()
    }
    encoder_ranks.pop(b"<|endoftext|>", None)
    encoder_ranks.pop(b"<|startoftext|>", None)
    if ranks != encoder_ranks:
        raise ValueError("GPT-2 encoder ranks differ from vocabulary merge priorities")
    return ranks


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        sort_keys=True,
    )
