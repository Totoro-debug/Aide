"""Resident Session lifecycle tests migrated from executor management seams."""

import asyncio
from pathlib import Path

import pytest

from omni.agent.message_bus import InboundMessage
from omni.agent.session.session import Session
from tests.agent.test_loop import (
    _BlockingRouter,
    _response,
    _Router,
    _runtime,
    _terminals,
    _TitleBehaviorRouter,
)
from tests.fixtures import collect_foreground_outbound


@pytest.mark.asyncio
async def test_loop_normal_close_saves_only_its_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, first, _bus = _runtime(tmp_path, _Router(()))
    closed: list[str] = []
    first_close = first.close

    def close_first() -> None:
        closed.append(first.session_id)
        first_close()

    monkeypatch.setattr(first, "close", close_first)

    await loop.execution.close()
    await loop.close()

    assert closed == [first.session_id]


@pytest.mark.asyncio
async def test_close_normally_cancels_active_run_without_dequeuing_the_next_message(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    router = _BlockingRouter(started)
    loop, session, _bus = _runtime(tmp_path, router)
    queued = InboundMessage("remains queued")
    await loop.start()
    await _bus.put_inbound(InboundMessage("active input"))
    await _bus.put_inbound(queued)
    await started.wait()

    await loop.execution.close()
    await loop.close()
    terminal = (await _terminals(_bus, 1))[0]

    assert terminal.metadata == {
        "finish_reason": "cancelled",
        "error_code": "turn_cancelled",
        "_streamed": True,
    }
    assert [message["content"] for message in session.messages if message["role"] == "user"] == [
        "active input"
    ]
    assert await _bus.inbound_snapshot() == (queued,)
    assert router.calls == ["call"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_turn", [False, True])
async def test_close_applies_first_input_title_fallback_before_final_save(
    tmp_path: Path,
    cancel_turn: bool,
) -> None:
    router = _TitleBehaviorRouter(
        (_response("Foreground response."),),
        title=_response("Unreleased title"),
        delay_title=True,
        block_first_foreground=cancel_turn,
    )
    loop, session, _bus = _runtime(tmp_path, router, title_prompt="Generate a title")

    await loop.start()
    if cancel_turn:
        turn = asyncio.create_task(collect_foreground_outbound(_bus, "  Cancelled first title.  "))
        await router.foreground_started.wait()
        await router.title_started.wait()
        await loop.cancel_active_run()
        terminal = (await turn)[-1]
        assert terminal.metadata["finish_reason"] == "cancelled"
        expected_title = "Cancelled first title."
    else:
        await collect_foreground_outbound(_bus, "  Shutdown fallback title.  ")
        await router.title_started.wait()
        expected_title = "Shutdown fallback title."

    await loop.execution.close()
    await loop.close()

    assert session.metadata["title"] == expected_title
    reloaded = Session.load(session.workspace_state, session.session_id)
    assert reloaded.metadata["title"] == expected_title


@pytest.mark.asyncio
async def test_closed_session_rejects_input_before_and_during_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, _, _ = _runtime(tmp_path, _Router(()))
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_finish() -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(loop.execution, "finish_work", blocked_finish)
    closing = asyncio.create_task(loop.execution.close())
    await started.wait()
    try:
        assert not loop.execution.foreground_input_admitted()
        with pytest.raises(RuntimeError, match="unavailable"):
            await loop.execution.run_foreground(InboundMessage("during close"))
    finally:
        release.set()
        await closing
    with pytest.raises(RuntimeError, match="unavailable"):
        await loop.execution.run_foreground(InboundMessage("after close"))
    await loop.close()


@pytest.mark.asyncio
async def test_session_close_surfaces_persistence_failure(tmp_path: Path) -> None:
    loop, session, _ = _runtime(tmp_path, _Router(()))

    def fail_close() -> None:
        raise OSError("fixture close failure")

    session.close = fail_close  # type: ignore[method-assign]
    with pytest.raises(OSError, match="fixture close failure"):
        await loop.execution.close()
