from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import tiktoken
import tiktoken_ext.openai_public  # type: ignore[import-untyped]

from aide.agent.context import tokenizer
from aide.agent.context.tokenizer import (
    context_estimator_version_for_model,
    estimate_context_request_tokens,
)
from aide.provider.models import ModelUsage


@pytest.mark.parametrize(
    ("model", "encoding"),
    [
        ("gpt-4o", "o200k_base"),
        ("gpt-4", "cl100k_base"),
        ("text-davinci-edit-001", "p50k_edit"),
        ("gpt2", "gpt2"),
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
    assert context_estimator_version_for_model("gpt-4") == "tiktoken-v1:cl100k_base"
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


def test_request_count_does_not_use_tiktoken_remote_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer._encoding_for_name.cache_clear()

    def reject_network(*args: object, **kwargs: object) -> None:
        pytest.fail("context tokenization attempted a network request")

    monkeypatch.setattr("requests.get", reject_network)

    assert (
        estimate_context_request_tokens([{"role": "user", "content": "offline"}], model="gpt-4o")
        == 9
    )


def test_all_tiktoken_vocabulary_resources_are_packaged_and_validated() -> None:
    expected = {
        "cl100k_base",
        "gpt2",
        "o200k_base",
        "o200k_harmony",
        "p50k_base",
        "p50k_edit",
        "r50k_base",
    }
    assert set(tokenizer._load_manifest()["encodings"]) == expected
    for name in expected:
        assert tokenizer._encoding_for_name(name).name == name


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
@pytest.mark.parametrize(
    "model", ["gpt-4o", "gpt-4", "gpt2", "text-davinci-edit-001", "gpt-oss-120b"]
)
def test_local_encoding_matches_official_encoding_for_user_text(
    content: str, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = tokenizer._encoding_name_for_model(model)
    local = tokenizer._encoding_for_name(name)
    # Supply verified local ranks to the official constructor; it must not download data.
    monkeypatch.setattr(
        tiktoken_ext.openai_public, "load_tiktoken_bpe", lambda *a, **k: local._mergeable_ranks
    )
    monkeypatch.setattr(
        tiktoken_ext.openai_public,
        "data_gym_to_mergeable_bpe_ranks",
        lambda *a, **k: local._mergeable_ranks,
    )
    official = tiktoken.Encoding(**tiktoken_ext.openai_public.ENCODING_CONSTRUCTORS[name]())
    assert local.encode_ordinary(content) == official.encode_ordinary(content)


@pytest.mark.parametrize(
    "model,resource_name", [("gpt-4o", "o200k_base.tiktoken"), ("gpt2", "gpt2-vocab.bpe")]
)
@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_invalid_bundled_vocabulary_fails_even_after_a_successful_load(
    model: str, resource_name: str, fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    shutil.copytree(Path(tokenizer.__file__).parent / "tokenizer_data", package / "tokenizer_data")
    monkeypatch.setattr(tokenizer, "files", lambda _: package)
    cache = tmp_path / "cache"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))
    tokenizer._load_manifest.cache_clear()
    tokenizer._encoding_for_name.cache_clear()
    try:
        assert (
            estimate_context_request_tokens([{"role": "user", "content": "offline"}], model=model)
            > 0
        )
        assert not cache.exists()
        resource = package / "tokenizer_data" / resource_name
        if fault == "missing":
            resource.unlink()
        else:
            resource.write_bytes(b"corrupt vocabulary")
        tokenizer._encoding_for_name.cache_clear()
        with pytest.raises(
            tokenizer.ContextTokenizerError, match=r"unavailable or invalid|checksum mismatch"
        ):
            estimate_context_request_tokens([{"role": "user", "content": "offline"}], model=model)
    finally:
        tokenizer._load_manifest.cache_clear()
        tokenizer._encoding_for_name.cache_clear()


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
