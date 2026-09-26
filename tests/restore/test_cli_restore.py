from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from textual.widgets import OptionList, Static

import myclaw.terminal.cli as cli
from myclaw.agent.loop import AgentLoop, ForegroundConversationProjection
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.session.restore import RestoreManager as SessionRestoreManager
from myclaw.agent.session.restore import RestoreMode
from myclaw.agent.session.session import RestoreAnchor, Session
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import UserConfiguration
from myclaw.management.commands import ManagementCommandDispatcher, ManagementPort
from myclaw.management.service import FatalManagementError, RestoreListingReport
from myclaw.terminal.conversation import TerminalConversationApp

SESSION_ID = "20260926-120000-000000_550e8400-e29b-41d4-a716-446655440000"
ANCHOR_TOKEN = UUID("550e8400-e29b-41d4-a716-446655440001")


@pytest.mark.asyncio
async def test_restore_command_lists_persisted_anchors_without_exposing_restore_state() -> None:
    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=SESSION_ID,
                anchors=(
                    RestoreAnchor(
                        anchor_id=7,
                        run_token=ANCHOR_TOKEN,
                        content="Restore this input",
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

    result = await ManagementCommandDispatcher(cast(ManagementPort, Management())).dispatch(
        "/restore"
    )

    assert result.handled is True
    assert result.restore_listing is not None
    assert result.restore_listing.session_id == SESSION_ID
    assert [anchor.anchor_id for anchor in result.restore_listing.anchors] == [7]
    assert (
        result.output == "Restore anchors:\n7. 2026-09-26T12:00:00.000+08:00 | Restore this input"
    )
    assert "restore_before" not in (result.output or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "admission",
    (
        None,
        "active",
        "queued",
        "cancel_wait",
        "double_listing",
        "lock_race",
        "stale",
        "persist_failure",
        "rebuild_failure",
    ),
)
async def test_cli_restore_rebuilds_same_session_id_and_persists_empty_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admission: str | None,
) -> None:
    events: list[str] = []
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    now = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
    loops: list[object] = []
    bus_ref: dict[str, object] = {}
    schedule_waiting = asyncio.Event()
    loop_pause_waiting = asyncio.Event()
    loop_pause_release = asyncio.Event()

    class FakeBus:
        def __init__(self) -> None:
            self.inbound: list[object] = []
            self.paused = False
            bus_ref["value"] = self

        async def inbound_snapshot(self) -> tuple[object, ...]:
            return tuple(self.inbound)

        async def pause_inbound_delivery(self) -> None:
            self.paused = True
            events.append("bus_pause")

        async def resume_inbound_delivery(self) -> None:
            self.paused = False
            events.append("bus_resume")

        async def reset(self) -> None:
            self.inbound.clear()
            events.append("bus_reset")

    class FakeMCP:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        async def start(self, _configuration: object) -> object:
            return SimpleNamespace(
                snapshot=(),
                failed_servers=(),
                failures=(),
                skipped_tool_counts=(),
            )

        async def prepare_generation(self) -> object:
            events.append("mcp_prepare")
            return SimpleNamespace(
                snapshot=(),
                failed_servers=(),
                failures=(),
                skipped_tool_counts=(),
            )

        def activate_generation(self, _report: object) -> None:
            events.append("mcp_activate")

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeKeywords:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def prepare(self, *_args: object) -> dict[str, tuple[str, ...]]:
            return {}

    class FakeRouter:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def route_status(self, _route: str) -> object:
            return SimpleNamespace()

        async def close(self) -> None:
            events.append("router_close")

    class FakeMemory:
        def __init__(self, _state: WorkspaceState) -> None:
            events.append("memory_init")

    class FakeDream:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def run(self) -> object:
            raise AssertionError("Dream is not part of restore")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeSchedule:
        def __init__(self, **_kwargs: object) -> None:
            self.paused = False

        def _prepare_start(self) -> None:
            return None

        async def register_dream_job(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        async def pause_and_wait_idle(self) -> None:
            self.paused = True
            events.append("schedule_idle")
            if admission == "cancel_wait":
                schedule_waiting.set()
                await asyncio.Event().wait()

        def resume(self) -> None:
            self.paused = False
            events.append("schedule_resume")

        async def pause_and_drain(self) -> None:
            return None

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

        async def drain_confirmation_aborts(self, **_kwargs: object) -> None:
            return None

        async def close(self) -> None:
            events.append("schedule_close")

    class FakeLoop:
        def __init__(self, **kwargs: object) -> None:
            state = cast(WorkspaceState, kwargs["workspace_state"])
            session_id = cast(str | None, kwargs["session_id"])
            if session_id is None:
                session = Session.create(state, now=lambda: now, new_uuid=uuid4)
            else:
                session = Session.load(state, session_id, now=lambda: now)
            self.session = session
            self.control = self
            self.skill_metadata = ()
            self.generation_id = uuid4()
            self.barrier = False
            self.active = False
            self.started = False
            self.aborted = False
            loops.append(self)

        @property
        def has_active_run(self) -> bool:
            return self.active

        def foreground_input_admitted(self) -> bool:
            return not self.barrier

        def bind_confirmation_requester(self, _requester: object) -> None:
            return None

        def preflight(self) -> None:
            if admission == "rebuild_failure" and len(loops) > 1:
                raise RuntimeError("injected restored generation preflight failure")
            return None

        async def start(self) -> None:
            self.started = True
            events.append("loop_start")

        async def wait_for_restore_idle(self) -> None:
            events.append("session_idle")
            await self.session.wait_for_pending_persist()

        async def _pause_for_replacement(self) -> None:
            if admission in {"double_listing", "lock_race"} and not loop_pause_waiting.is_set():
                loop_pause_waiting.set()
                await loop_pause_release.wait()
            self.barrier = True
            events.append("loop_pause")

        async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
            self.barrier = False
            events.append(f"loop_release:{resume_inbound}")

        async def abort(self) -> None:
            self.aborted = True
            self.session.abandon()
            await self.session.wait_for_pending_persist()
            events.append("loop_abort")

        async def close(self) -> None:
            self.session.close()
            await self.session.wait_for_pending_persist()
            events.append("loop_close")

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return ForegroundConversationProjection(
                session_id=self.session.session_id,
                messages=tuple(self.session.messages),
            )

    class FakeApp:
        fatal_management_error = None

        def __init__(self, **kwargs: object) -> None:
            self.dispatcher = cast(
                ManagementCommandDispatcher,
                kwargs["management_dispatcher"],
            )
            self.control = kwargs["control"]

        async def quiesce_for_rebind(self) -> None:
            events.append("terminal_quiesce")

        async def rebind_agent_loop(self, *, control: object, **_kwargs: object) -> None:
            self.control = control
            events.append("terminal_rebind")

        async def run_async(self) -> None:
            loop = cast(FakeLoop, self.control)
            before = loop.session.capture_restore_before()
            loop.session.commit_agent_run(
                [{"role": "user", "content": "discard this input"}],
                pending_last_compacted=loop.session.last_compacted,
                pending_action_summary="",
                restore_before=before,
                restore_run_token=uuid4(),
            )
            await loop.session.wait_for_pending_persist()
            bus = cast(FakeBus, bus_ref["value"])
            if admission == "active":
                loop.active = True
            elif admission == "queued":
                bus.inbound.append(object())
            if admission == "double_listing":
                listing_task = asyncio.create_task(self.dispatcher.dispatch("/restore"))
                await loop_pause_waiting.wait()
                competing_task = asyncio.create_task(self.dispatcher.dispatch("/restore"))
                await asyncio.sleep(0)
                assert competing_task.done() is False
                loop_pause_release.set()
                listing, competing = await asyncio.gather(
                    asyncio.wait_for(listing_task, timeout=1),
                    asyncio.wait_for(competing_task, timeout=1),
                )
                assert listing.restore_listing is not None
                assert competing.output == (
                    "model_invalid_request: Session Restore is waiting for confirmation."
                )
                assert loop.barrier is True
                inspected = await self.dispatcher.restore_inspect(1)
                assert inspected.restore_plan is not None
                cancelled = await self.dispatcher.restore_cancel()
                assert cancelled.output == "Session Restore cancelled."
                assert loop.barrier is False
                return
            if admission == "lock_race":
                original_wait = Session.wait_for_pending_persist
                resume_guard_passed = asyncio.Event()
                resume_continue = asyncio.Event()
                resume_wait_used = False

                async def wait_after_resume_guard(current: Session) -> None:
                    nonlocal resume_wait_used
                    if current is loop.session and not resume_wait_used:
                        resume_wait_used = True
                        resume_guard_passed.set()
                        await resume_continue.wait()
                    await original_wait(current)

                monkeypatch.setattr(
                    Session,
                    "wait_for_pending_persist",
                    wait_after_resume_guard,
                )
                listing_task = asyncio.create_task(self.dispatcher.dispatch("/restore"))
                await loop_pause_waiting.wait()
                resume_task = asyncio.create_task(self.dispatcher.resume(loop.session.session_id))
                await resume_guard_passed.wait()
                resume_continue.set()
                await asyncio.sleep(0)
                loop_pause_release.set()
                listing, blocked_resume = await asyncio.gather(
                    asyncio.wait_for(listing_task, timeout=1),
                    asyncio.wait_for(resume_task, timeout=1),
                )
                assert listing.restore_listing is not None
                assert blocked_resume.output == (
                    "model_invalid_request: Session Restore is waiting for confirmation."
                )
                cancelled = await self.dispatcher.restore_cancel()
                assert cancelled.output == "Session Restore cancelled."
                assert loop.barrier is False
                assert bus.paused is False
                assert "mcp_prepare" not in events
                return
            listing = await self.dispatcher.dispatch("/restore")
            if admission in {"active", "queued"}:
                assert listing.output == (
                    "model_invalid_request: Finish or cancel the active foreground run "
                    "and clear queued input before restoring."
                )
                assert listing.restore_listing is None
                assert loop.barrier is False
                assert bus.paused is False
                return
            assert listing.restore_listing is not None
            assert [anchor.anchor_id for anchor in listing.restore_listing.anchors] == [1]
            blocked_resume = await self.dispatcher.resume(loop.session.session_id)
            assert blocked_resume.output == (
                "model_invalid_request: Session Restore is waiting for confirmation."
            )
            if admission == "cancel_wait":
                inspection = asyncio.create_task(self.dispatcher.restore_inspect(1))
                await schedule_waiting.wait()
                cancelled = await self.dispatcher.restore_cancel()
                assert cancelled.output == "Session Restore cancelled."
                with pytest.raises(asyncio.CancelledError):
                    await inspection
                assert loop.barrier is False
                assert bus.paused is False
                return
            inspected = await self.dispatcher.restore_inspect(1)
            assert inspected.restore_plan is not None
            cancelled = await self.dispatcher.restore_cancel()
            assert cancelled.output == "Session Restore cancelled."
            assert loop.barrier is False
            assert bus.paused is False
            relisted = await self.dispatcher.dispatch("/restore")
            assert relisted.restore_listing is not None
            inspected = await self.dispatcher.restore_inspect(1)
            assert inspected.restore_plan is not None
            plan = inspected.restore_plan
            if admission == "stale":
                loop.session.update_metadata(title="Changed after inspection")
                loop.session.persist()
                await loop.session.wait_for_pending_persist()
                stale = await self.dispatcher.restore_commit(
                    plan,
                    RestoreMode.CONVERSATION_ONLY,
                )
                assert stale.output == (
                    "model_invalid_request: The selected Restore plan is stale; "
                    "no changes were made."
                )
                assert len(loops) == 1
                assert loop.session.messages[0]["content"] == "discard this input"
                assert loop.barrier is False
                assert bus.paused is False
                assert (await self.dispatcher.restore_result()).restore_result is None
                return
            if admission == "persist_failure":

                def fail_session_write(_session: Session, _anchor_id: int) -> object:
                    raise OSError("injected strict Session write failure")

                monkeypatch.setattr(Session, "restore_before_durably", fail_session_write)
                await self.dispatcher.restore_commit(
                    plan,
                    RestoreMode.CONVERSATION_ONLY,
                )
            blocked_resume = await self.dispatcher.resume(loop.session.session_id)
            assert blocked_resume.output == (
                "model_invalid_request: Session Restore is waiting for confirmation."
            )
            committed = await self.dispatcher.restore_commit(
                plan,
                RestoreMode.CONVERSATION_ONLY,
            )
            assert committed.restore_result is not None
            assert committed.restore_result.session_id == loop.session.session_id
            assert committed.restore_result.removed_messages == 1
            result = await self.dispatcher.restore_result()
            assert result.restore_result == committed.restore_result
            assert len(loops) == 2
            target = cast(FakeLoop, loops[1])
            assert target.session.session_id == loop.session.session_id
            assert target.session.messages == []
            assert (
                Session.load(
                    target.session.workspace_state,
                    target.session.session_id,
                    now=lambda: now,
                ).messages
                == []
            )
            assert bus.paused is False

    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCP)
    monkeypatch.setattr(cli, "MCPKeywordPreparer", FakeKeywords)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemory)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeSchedule)
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    configuration = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(
            max_iterations=50,
            permission_level="workspace-write",
            exec_shell="auto",
        ),
    )
    if admission in {"persist_failure", "rebuild_failure"}:
        with pytest.raises(FatalManagementError) as raised:
            await cli._run_cli_conversation(
                agent_home=home,
                workspace=workspace,
                configuration=cast(UserConfiguration, configuration),
            )
        assert raised.value.error.code == "persistence_error"
        expected_message = (
            "Workspace Restore could not be recovered."
            if admission == "persist_failure"
            else "Runtime Session replacement could not be completed."
        )
        assert raised.value.error.message == expected_message
        if admission == "rebuild_failure":
            old_loop = cast(FakeLoop, loops[0])
            assert old_loop.aborted is True
            assert (
                Session.load(
                    old_loop.session.workspace_state,
                    old_loop.session.session_id,
                    now=lambda: now,
                ).messages
                == []
            )
        return
    await cli._run_cli_conversation(
        agent_home=home,
        workspace=workspace,
        configuration=cast(UserConfiguration, configuration),
    )

    if admission is not None:
        return
    assert events.count("loop_pause") == 2
    assert events.count("loop_release:True") == 2
    assert events.count("schedule_resume") == 2
    assert events.index("loop_pause") < events.index("schedule_idle")
    assert events.index("schedule_idle") < events.index("mcp_prepare")
    assert events.index("loop_abort") < max(
        index for index, event in enumerate(events) if event == "loop_release:True"
    )


@pytest.mark.asyncio
async def test_cli_recovers_pending_restore_before_runtime_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=home.path)
    session = Session.create(state)
    before = session.capture_restore_before()
    session.commit_agent_run(
        [{"role": "user", "content": "discard during startup recovery"}],
        pending_last_compacted=session.last_compacted,
        pending_action_summary="",
        restore_before=before,
        restore_run_token=uuid4(),
    )
    await session.wait_for_pending_persist()
    plan = SessionRestoreManager(state, session.session_id).inspect(session, 1)

    def interrupt_after_pending(phase: str) -> None:
        if phase == "pending_intent":
            raise RuntimeError("injected startup recovery interruption")

    with pytest.raises(RuntimeError, match="startup recovery interruption"):
        await SessionRestoreManager(
            state,
            session.session_id,
            phase_hook=interrupt_after_pending,
        ).execute(plan, RestoreMode.CONVERSATION_ONLY)
    loaded_session_ids: list[str | None] = []

    class FakeRestoreManager:
        def __init__(self, runtime_state: WorkspaceState) -> None:
            events.append("restore_init")
            self._manager = SessionRestoreManager(runtime_state)

        async def recover_pending(self) -> object:
            events.append("restore_recover")
            return await self._manager.recover_pending()

    class FakeMCP:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            events.append("mcp_init")

        async def start(self, _configuration: object) -> object:
            events.append("mcp_start")
            return SimpleNamespace(
                snapshot=(),
                failed_servers=(),
                failures=(),
                skipped_tool_counts=(),
            )

        async def close(self) -> None:
            events.append("mcp_close")

    class FakeBus:
        def __init__(self) -> None:
            events.append("bus_init")

    class FakeRouter:
        def __init__(self, **_kwargs: object) -> None:
            events.append("router_init")

        def route_status(self, _route: str) -> object:
            return SimpleNamespace()

        async def close(self) -> None:
            events.append("router_close")

    class FakeKeywords:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def prepare(self, *_args: object) -> dict[str, tuple[str, ...]]:
            return {}

    class FakeMemory:
        def __init__(self, _state: WorkspaceState) -> None:
            events.append("memory_init")

    class FakeDream:
        def __init__(self, **_kwargs: object) -> None:
            events.append("dream_init")

        async def run(self) -> object:
            raise AssertionError("Dream must not run during startup")

        async def close(self) -> None:
            events.append("dream_close")

    class FakeSchedule:
        def __init__(self, **_kwargs: object) -> None:
            events.append("schedule_init")

        def _prepare_start(self) -> None:
            events.append("schedule_preflight")

        async def register_dream_job(self, **_kwargs: object) -> None:
            events.append("dream_register")

        def start(self) -> None:
            events.append("schedule_start")

        async def pause_and_drain(self) -> None:
            events.append("schedule_pause")

        async def drain_confirmation_aborts(self, **_kwargs: object) -> None:
            return None

        async def close(self) -> None:
            events.append("schedule_close")

        def status_snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {})

    class FakeLoop:
        def __init__(self, **kwargs: object) -> None:
            state = cast(WorkspaceState, kwargs["workspace_state"])
            session_id = cast(str | None, kwargs["session_id"])
            loaded_session_ids.append(session_id)
            if session_id is None:
                self.session = Session.create(state)
            else:
                self.session = Session.load(state, session_id)
            self.control = self
            self.skill_metadata = ()

        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return True

        def bind_confirmation_requester(self, _requester: object) -> None:
            return None

        def preflight(self) -> None:
            events.append("loop_preflight")

        async def start(self) -> None:
            events.append("loop_start")

        async def close(self) -> None:
            events.append("loop_close")

        async def abort(self) -> None:
            events.append("loop_abort")

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return ForegroundConversationProjection(
                session_id=self.session.session_id,
                messages=tuple(self.session.messages),
            )

    class FakeApp:
        fatal_management_error = None

        def __init__(self, **_kwargs: object) -> None:
            events.append("app_init")

        async def run_async(self) -> None:
            events.append("app_run")

    monkeypatch.setattr(cli, "RestoreManager", FakeRestoreManager)
    monkeypatch.setattr(cli, "MCPRuntimeManager", FakeMCP)
    monkeypatch.setattr(cli, "MessageBus", FakeBus)
    monkeypatch.setattr(cli, "ModelRouter", FakeRouter)
    monkeypatch.setattr(cli, "MCPKeywordPreparer", FakeKeywords)
    monkeypatch.setattr(cli, "MemoryManager", FakeMemory)
    monkeypatch.setattr(cli, "Dream", FakeDream)
    monkeypatch.setattr(cli, "ScheduleService", FakeSchedule)
    monkeypatch.setattr(cli, "AgentLoop", FakeLoop)
    monkeypatch.setattr(cli, "TerminalConversationApp", FakeApp)

    configuration = SimpleNamespace(
        mcp={},
        memory=SimpleNamespace(schedule="0 * * * *", batch_size=10),
        runtime=SimpleNamespace(
            max_iterations=50,
            permission_level="workspace-write",
            exec_shell="auto",
        ),
    )
    await cli._run_cli_conversation(
        agent_home=home,
        workspace=workspace,
        configuration=cast(UserConfiguration, configuration),
    )

    assert events.index("restore_init") < events.index("restore_recover")
    assert events.index("restore_recover") < events.index("mcp_init")
    assert events.index("restore_recover") < events.index("memory_init")
    assert events.index("restore_recover") < events.index("schedule_init")
    assert events.index("restore_recover") < events.index("loop_preflight")
    assert loaded_session_ids == [session.session_id]
    assert Session.load(state, session.session_id).messages == []


@pytest.mark.asyncio
async def test_restore_waiter_cancellation_does_not_cancel_title_work() -> None:
    title_finished = asyncio.Event()
    pending_persist_called = False

    async def title_work() -> None:
        await title_finished.wait()

    class PendingSession:
        async def wait_for_pending_persist(self) -> None:
            nonlocal pending_persist_called
            pending_persist_called = True

    title_task = asyncio.create_task(title_work())
    loop = object.__new__(AgentLoop)
    cast(Any, loop)._aborted = False
    cast(Any, loop)._title_work = {"session": SimpleNamespace(task=title_task)}
    cast(Any, loop)._session = PendingSession()
    waiter = asyncio.create_task(loop.wait_for_restore_idle())
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert not title_task.done()
    assert pending_persist_called is False
    title_finished.set()
    await title_task


@pytest.mark.asyncio
async def test_terminal_rejects_ordinary_input_while_restore_barrier_is_held() -> None:
    class Control:
        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return False

        def bind_confirmation_callback(self, _callback: object) -> None:
            return None

        def unbind_confirmation_callback(self, _callback: object) -> None:
            return None

        async def cancel_active_run(self) -> None:
            return None

        def respond_to_confirmation(self, _confirmation_id: object, _decision: object) -> None:
            return None

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return ForegroundConversationProjection(session_id=SESSION_ID, messages=())

    bus = MessageBus()
    app = TerminalConversationApp(
        bus=bus,
        control=Control(),
        management_dispatcher=ManagementCommandDispatcher(cast(ManagementPort, object())),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("ordinary input"), "enter")
        await pilot.pause()

        assert not app._pending_inputs
        assert await bus.inbound_snapshot() == ()
        assert any(
            "restore_in_progress" in str(cast(Static, row).content)
            for row in app.query(".management-row")
        )


@pytest.mark.asyncio
async def test_restore_anchor_picker_shows_local_preview_and_cancels_without_mutation() -> None:
    cancel_calls = 0

    class Management:
        async def restore_listing(self) -> RestoreListingReport:
            return RestoreListingReport(
                session_id=SESSION_ID,
                anchors=(
                    RestoreAnchor(
                        anchor_id=1,
                        run_token=ANCHOR_TOKEN,
                        content="Restore\nthis input",
                        timestamp="2026-09-26T12:00:00.000+08:00",
                    ),
                ),
            )

        async def restore_cancel(self) -> None:
            nonlocal cancel_calls
            cancel_calls += 1

    class Control:
        @property
        def has_active_run(self) -> bool:
            return False

        def foreground_input_admitted(self) -> bool:
            return cancel_calls == 1

        def bind_confirmation_callback(self, _callback: object) -> None:
            return None

        def unbind_confirmation_callback(self, _callback: object) -> None:
            return None

        async def cancel_active_run(self) -> None:
            return None

        def respond_to_confirmation(self, _confirmation_id: object, _decision: object) -> None:
            return None

        def project_foreground_conversation(self) -> ForegroundConversationProjection:
            return ForegroundConversationProjection(session_id=SESSION_ID, messages=())

    bus = MessageBus()
    app = TerminalConversationApp(
        bus=bus,
        control=Control(),
        management_dispatcher=ManagementCommandDispatcher(cast(ManagementPort, Management())),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("/restore"), "enter")
        async with asyncio.timeout(1):
            while app.screen.id != "restore-anchor-picker":
                await pilot.pause()

        picker_text = "\n".join(
            str(option.prompt)
            for option in app.screen.query_one("#restore-anchor-options", OptionList).options
        )
        assert "1." in picker_text
        assert "Restore this input" in picker_text
        assert (
            datetime.fromisoformat("2026-09-26T12:00:00.000+08:00")
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
            in picker_text
        )
        assert app.screen.focused is app.screen.query_one("#restore-anchor-options")

        await pilot.click(offset=(1, 1))
        assert app.screen.id == "restore-anchor-picker"

        await pilot.press("escape")
        await pilot.pause()
        assert cancel_calls == 1
        assert app.screen.id != "restore-anchor-picker"

        await pilot.press(*list("ordinary input"), "enter")
        await pilot.pause()

        assert tuple(message.content for message in await bus.inbound_snapshot()) == (
            "ordinary input",
        )
