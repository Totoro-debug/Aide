from __future__ import annotations

import pytest

from omni.agent.tools.network_safety import TargetAssessment, assess_target


class _UnexpectedResolver:
    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        raise AssertionError(f"DNS must not resolve a literal address: {hostname}:{port}")


class _StaticResolver:
    def __init__(self, answers: tuple[str, ...]) -> None:
        self._answers = answers
        self.requests: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.requests.append((hostname, port))
        return self._answers


class _FailingResolver:
    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        del hostname, port
        raise OSError("DNS unavailable")


@pytest.mark.asyncio
async def test_non_global_literal_is_assessed_without_dns() -> None:
    assessment = await assess_target("127.0.0.1", 80, _UnexpectedResolver())

    assert assessment == TargetAssessment(risk="literal_non_global")


@pytest.mark.asyncio
async def test_dns_target_is_public_when_every_answer_is_global() -> None:
    resolver = _StaticResolver(("8.8.8.8", "2606:4700:4700::1111"))

    assessment = await assess_target("example.com", 443, resolver)

    assert assessment == TargetAssessment()
    assert resolver.requests == [("example.com", 443)]


@pytest.mark.asyncio
async def test_dns_target_is_unsafe_when_any_answer_is_non_global() -> None:
    resolver = _StaticResolver(("8.8.8.8", "127.0.0.1"))

    assessment = await assess_target("example.com", 443, resolver)

    assert assessment == TargetAssessment(risk="dns_non_global")


@pytest.mark.asyncio
async def test_dns_target_with_no_answers_is_unverifiable() -> None:
    assessment = await assess_target("example.com", 80, _StaticResolver(()))

    assert assessment == TargetAssessment(risk="dns_empty")


@pytest.mark.asyncio
async def test_dns_failure_is_unverifiable() -> None:
    assessment = await assess_target("example.com", 80, _FailingResolver())

    assert assessment == TargetAssessment(risk="dns_failure")
