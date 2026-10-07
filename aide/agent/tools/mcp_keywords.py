"""Prepare and persist discovery keywords for discovered MCP Tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal, Protocol

from aide.agent.tools.mcp import MCPTool
from aide.config.config import (
    ConfigLoader,
    MCPServerConfiguration,
    normalize_mcp_tool_keywords,
)
from aide.templates import render_template

_MAX_KEYWORD_PREPARATION_CONCURRENCY = 4


class KeywordModelRouter(Protocol):
    """The direct chat completion seam used for startup metadata."""

    async def complete(
        self,
        route: Literal["chat"],
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _KeywordCacheKey:
    server_name: str
    remote_name: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreparedKeywords:
    identity: tuple[str, str]
    model_name: str
    keywords: tuple[str, ...]
    generated: bool


class MCPKeywordPreparer:
    """Prepare one complete MCP snapshot without affecting Session state."""

    def __init__(
        self,
        model_router: KeywordModelRouter,
        config_loader: ConfigLoader,
    ) -> None:
        self._router = model_router
        self._config_loader = config_loader
        self._cache: dict[_KeywordCacheKey, tuple[str, ...]] = {}

    async def prepare(
        self,
        tools: Sequence[MCPTool],
        servers: Mapping[str, MCPServerConfiguration],
    ) -> Mapping[str, tuple[str, ...]]:
        """Return prepared keywords keyed by each Tool's model-facing name."""
        candidates: list[tuple[str, str, str, str, dict[str, Any]]] = []
        for tool in tools:
            candidates.append(
                (
                    tool.server_name,
                    tool.remote_name,
                    tool.name,
                    tool.description,
                    deepcopy(tool.parameters),
                )
            )
        if not candidates:
            return MappingProxyType({})

        semaphore = asyncio.Semaphore(_MAX_KEYWORD_PREPARATION_CONCURRENCY)

        async def prepare_one(
            metadata: tuple[str, str, str, str, dict[str, Any]],
        ) -> _PreparedKeywords:
            async with semaphore:
                server_name, remote_name, model_name, description, parameters = metadata
                identity = (server_name, remote_name)
                server = servers.get(server_name)
                configured = () if server is None else server.tool_keywords.get(remote_name, ())
                if configured:
                    return _PreparedKeywords(identity, model_name, configured, generated=False)

                cache_key = _cache_key(
                    server_name=server_name,
                    remote_name=remote_name,
                    description=description,
                    parameters=parameters,
                )
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return _PreparedKeywords(identity, model_name, cached, generated=False)

                try:
                    generated = await self._generate(
                        remote_name=remote_name,
                        description=description,
                        parameters=parameters,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    generated = (remote_name,)
                    persist = False
                else:
                    persist = True
                self._cache[cache_key] = generated
                return _PreparedKeywords(identity, model_name, generated, generated=persist)

        tasks = tuple(asyncio.create_task(prepare_one(metadata)) for metadata in candidates)
        try:
            prepared = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        generated_values = {item.identity: item.keywords for item in prepared if item.generated}
        effective_values: Mapping[tuple[str, str], tuple[str, ...]] = MappingProxyType({})
        if generated_values:
            try:
                effective_values = self._config_loader.fill_mcp_tool_keywords(generated_values)
            except Exception:
                effective_values = MappingProxyType({})

        result: dict[str, tuple[str, ...]] = {}
        for metadata, item in zip(candidates, prepared, strict=True):
            effective = effective_values.get(item.identity, item.keywords)
            result[item.model_name] = effective
            if item.generated:
                server_name, remote_name, _model_name, description, parameters = metadata
                self._cache[
                    _cache_key(
                        server_name=server_name,
                        remote_name=remote_name,
                        description=description,
                        parameters=parameters,
                    )
                ] = effective
        return MappingProxyType(result)

    async def _generate(
        self,
        *,
        remote_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> tuple[str, ...]:
        payload = json.dumps(
            {
                "name": remote_name,
                "description": description,
                "input_schema": deepcopy(parameters),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        response = await self._router.complete(
            "chat",
            messages=[
                {
                    "role": "system",
                    "content": render_template("mcp-keyword-system-prompt.md"),
                },
                {"role": "user", "content": payload},
            ],
            tools=(),
        )
        content = response.message.content
        if not isinstance(content, str):
            raise ValueError("MCP keyword generation returned non-text content")
        decoded = json.loads(content)
        if not isinstance(decoded, list):
            raise ValueError("MCP keyword generation returned a non-array")
        keywords = normalize_mcp_tool_keywords(decoded)
        if not keywords:
            raise ValueError("MCP keyword generation returned no keywords")
        return keywords


def _cache_key(
    *,
    server_name: str,
    remote_name: str,
    description: str,
    parameters: dict[str, Any],
) -> _KeywordCacheKey:
    try:
        fingerprint_source = json.dumps(
            {"description": description, "parameters": parameters},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        fingerprint_source = repr((description, parameters))
    fingerprint = sha256(fingerprint_source.encode("utf-8")).hexdigest()
    return _KeywordCacheKey(server_name, remote_name, fingerprint)


__all__ = ["MCPKeywordPreparer"]
