from __future__ import annotations

import json
from typing import Any

import pytest
import tiktoken

from aide.agent.context import tokenizer
from aide.agent.context.tokenizer import (
    context_estimator_version_for_model,
    estimate_context_request_tokens,
)
from aide.provider.models import ModelUsage


@pytest.mark.parametrize(
    ("model", "encoding"),
    [
        ("gpt-5", "o200k_base"),
        ("gpt-5.4", "o200k_base"),
        ("gpt-4.1", "o200k_base"),
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-mini", "o200k_base"),
        ("o1", "o200k_base"),
        ("o3", "o200k_base"),
        ("o4-mini", "o200k_base"),
        ("gpt-oss-120b", "o200k_harmony"),
    ],
)
def test_openai_models_use_tiktoken_encoding_mapping(model: str, encoding: str) -> None:
    assert (
        tokenizer._encoding_name_for_model(model)
        == tiktoken.model.encoding_name_for_model(model)
        == encoding
    )


@pytest.mark.parametrize("model", ["", "claude-sonnet-4", "vendor-model-7"])
def test_unknown_models_use_the_local_o200k_fallback(model: str) -> None:
    assert tokenizer._encoding_name_for_model(model) == "o200k_base"
    assert context_estimator_version_for_model(model) == "tiktoken-v1:o200k_base"


def test_o200k_pattern_counts_mixed_letter_runs() -> None:
    content = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0 " * 12

    assert (
        estimate_context_request_tokens([{"role": "user", "content": content}], model="gpt-4o")
        == 488
    )


def test_context_estimator_versions_include_encoding_identity() -> None:
    assert context_estimator_version_for_model("gpt-oss-120b") == "tiktoken-v1:o200k_harmony"
    assert context_estimator_version_for_model("gpt-4o") == "tiktoken-v1:o200k_base"


def test_request_estimate_counts_system_messages_and_exposed_tool_schemas() -> None:
    messages = [
        {"role": "system", "content": "hello world"},
        {"role": "user", "content": "read config.json"},
    ]
    tools = [{"name": "read_file", "parameters": {"type": "object"}}]

    assert estimate_context_request_tokens(messages, tools, model="gpt-4o") == 25
    assert estimate_context_request_tokens(
        messages, tools, model="gpt-4o"
    ) > estimate_context_request_tokens(messages, model="gpt-4o", tools=())


def test_special_token_literals_are_counted_as_ordinary_user_text() -> None:
    messages = [{"role": "user", "content": "literal <|endoftext|> token"}]

    assert estimate_context_request_tokens(messages, model="gpt-4o") == 17


@pytest.mark.parametrize(
    "model", ["gpt2", "text-davinci-003", "text-davinci-edit-001", "davinci", "gpt-4"]
)
def test_legacy_models_are_rejected_before_loading_an_encoding(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_load(name: str) -> tiktoken.Encoding:
        pytest.fail(f"legacy encoding {name} must not load")

    monkeypatch.setattr(tiktoken, "get_encoding", reject_load)
    with pytest.raises(tokenizer.ContextTokenizerError, match="Legacy model"):
        estimate_context_request_tokens([{"role": "user", "content": "hello"}], model=model)


def test_request_count_uses_the_official_encoding_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official = tiktoken.get_encoding("o200k_base")
    calls: list[str] = []

    def load(name: str) -> tiktoken.Encoding:
        calls.append(name)
        return official

    monkeypatch.setattr(tiktoken, "get_encoding", load)

    assert (
        estimate_context_request_tokens([{"role": "user", "content": "hello"}], model="gpt-4o") == 9
    )
    assert calls == ["o200k_base"]


def test_encoding_download_failure_is_explicit_without_character_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("download failed")

    def reject_load(name: str) -> tiktoken.Encoding:
        raise failure

    monkeypatch.setattr(tiktoken, "get_encoding", reject_load)
    with pytest.raises(tokenizer.ContextTokenizerError, match="Check network access") as raised:
        estimate_context_request_tokens([{"role": "user", "content": "hello"}], model="gpt-4o")
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    "content",
    [
        "中文上下文计量",
        "hello world",
        "def f(x):\n    return x + 1",
        '{"path":"配置.json"}',
        "<|endoftext|>",
    ],
)
@pytest.mark.parametrize("model", ["gpt-5", "gpt-4o", "gpt-oss-120b", "claude-sonnet-4"])
def test_request_count_uses_ordinary_encoding_for_mixed_user_text(content: str, model: str) -> None:
    name = tokenizer._encoding_name_for_model(model)
    official = tiktoken.get_encoding(name)
    serialized_message = json.dumps(
        {"role": "user", "content": content},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert estimate_context_request_tokens(
        [{"role": "user", "content": content}], model=model
    ) == len(official.encode_ordinary(serialized_message))


def test_estimator_identity_change_uses_full_local_count() -> None:
    from aide.agent.context.budget import ContextUsageSnapshot, project_next_request_usage

    messages = [{"role": "user", "content": "中文代码 print('hello')"}]
    route: dict[str, Any] = dict(
        requested_route="chat",
        selected_route="chat",
        provider_id="provider",
        model="gpt-4o",
        context_window=1000,
        max_output=200,
    )
    reported = dict(model_calls=1, input_tokens=300, output_tokens=100, total_tokens=400)
    for version in ("utf8-bytes-div4-v1", "tiktoken-v1:cl100k_base"):
        snapshot = ContextUsageSnapshot(
            **route,
            anchor_estimated_tokens=10,
            estimator_version=version,
            run_projected_tokens=0,
            run_projection_source="estimated",
        )
        projection = project_next_request_usage(
            messages, snapshot=snapshot, reported_usage=reported, **route
        )
        assert projection.source == "estimated"
        assert projection.projected_tokens == estimate_context_request_tokens(
            messages, model="gpt-4o"
        )


def test_tokenizer_projection_applies_prompt_and_schema_reduction_without_adding_cached_usage() -> (
    None
):
    from aide.agent.context.budget import ContextUsageSnapshot, project_next_request_usage

    messages = [{"role": "system", "content": "中文上下文" * 500}]
    tools = [{"name": "read_file", "description": "字段说明" * 500}]
    route: dict[str, Any] = dict(
        requested_route="chat",
        selected_route="chat",
        provider_id="provider",
        model="gpt-4o",
        context_window=10000,
        max_output=200,
    )
    snapshot = ContextUsageSnapshot(
        **route,
        anchor_estimated_tokens=estimate_context_request_tokens(messages, tools, model="gpt-4o"),
        estimator_version=context_estimator_version_for_model("gpt-4o"),
        run_projected_tokens=0,
        run_projection_source="estimated",
    )
    model_usage = ModelUsage(
        input_tokens=200,
        cached_input_tokens=100,
        output_tokens=100,
        total_tokens=300,
    )
    usage = {"model_calls": 1, **model_usage.to_dict()}
    unchanged = project_next_request_usage(
        messages, tools, snapshot=snapshot, reported_usage=usage, **route
    )
    reduced = project_next_request_usage(
        [{"role": "system", "content": "short"}], snapshot=snapshot, reported_usage=usage, **route
    )
    assert unchanged.source == reduced.source == "reported_delta"
    assert unchanged.projected_tokens == 300
    assert reduced.projected_tokens == 0
