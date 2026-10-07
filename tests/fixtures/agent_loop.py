"""Reusable AgentRunExecutor public-seam helpers."""

import asyncio
from collections.abc import Callable
from typing import Any
from uuid import UUID

from aide.agent.confirmation import ConfirmationEnvelope
from aide.agent.loop import (
    AgentRunExecutor,
    ConfirmationCallback,
    ForegroundConversationProjection,
    session_runtime_status_input,
)
from aide.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from aide.agent.tools.tool_gateway import ConfirmationDecision
from aide.config.agent_home import AgentHome
from aide.config.config import UserConfiguration
from aide.management.commands import MANAGEMENT_COMMANDS
from aide.management.service import RuntimeStatusInput
from aide.service.execution import SessionExecution
from aide.skills.catalog import SkillLoader


class CallbackConfirmationRequester:
    """Test adapter driving foreground confirmations through a callback."""

    def __init__(self, callback: Callable[[Any], None]) -> None:
        self._callback = callback
        self._request: Any | None = None
        self._future: asyncio.Future[ConfirmationDecision] | None = None

    async def request(self, request: Any) -> ConfirmationDecision:
        if self._request is not None:
            raise RuntimeError("A foreground confirmation request is already pending")
        future: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
        self._request = request
        self._future = future
        try:
            self._callback(request)
            return await future
        finally:
            if self._request is request:
                self._request = None
                self._future = None

    def respond(self, confirmation_id: UUID, decision: ConfirmationDecision) -> None:
        if decision not in {"approved", "declined"}:
            raise ValueError("confirmation decision must be approved or declined")
        request = self._request
        future = self._future
        if (
            request is None
            or future is None
            or request.confirmation_id != confirmation_id
            or future.done()
        ):
            raise ValueError("Confirmation response is late or unknown")
        future.set_result(decision)

    def cancel(self) -> None:
        future = self._future
        if future is not None and not future.done():
            future.cancel()

    def unbind(self, callback: Callable[[Any], None]) -> None:
        if self._callback is callback:
            self._callback = lambda _request: None
            self.cancel()


def loaded_skill_loader(home: AgentHome, configuration: UserConfiguration) -> SkillLoader:
    """Create the injected Skill snapshot at a test composition point."""
    loader = SkillLoader(
        root=home.skills_directory,
        reserved_names=tuple(command.token for command in MANAGEMENT_COMMANDS),
        enable_always_load=configuration.runtime.enable_skill_always_load,
    )
    loader.load()
    return loader


async def collect_foreground_outbound(
    bus: MessageBus,
    content: str,
) -> tuple[OutboundMessage, ...]:
    """Submit one foreground input and collect through its terminal marker."""
    await bus.put_inbound(InboundMessage(content=content))
    messages: list[OutboundMessage] = []
    while True:
        message = await bus.get_outbound()
        messages.append(message)
        if message.metadata.get("_streamed") is True:
            return tuple(messages)


class DrivenExecutor(AgentRunExecutor):
    """Test harness driving bus inputs through the explicit single-Run seam."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._consumer_task: asyncio.Task[None] | None = None
        self._foreground_consumer_enabled = True
        self._retired = False
        self.execution = SessionExecution(
            self.session, self._bus, lambda _state: self,
            lambda: self._skill_loader.metadata, self._status_input,
            run_state=self._session_run_state,
        )
        self.control = ExecutorControl(self)


    def _activate_prepared(self) -> None:
        if self._started:
            return
        if not self._preflighted:
            raise RuntimeError("Executor was not preflighted")
        started_at = self._monotonic_now()
        if self._foreground_consumer_enabled:
            consumer = self._consume_foreground()
            try:
                consumer_task = asyncio.create_task(consumer)
            except BaseException:
                consumer.close()
                raise
            self._consumer_task = consumer_task
        self._generation_started_at = started_at
        self._started = True

    def _owned_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        tasks = super()._owned_tasks()
        return tasks if self._consumer_task is None else (*tasks, self._consumer_task)

    def _clear_owned_task_references(self) -> None:
        super()._clear_owned_task_references()
        self._consumer_task = None

    def disable_foreground_consumer(self) -> None:
        """Keep activation compatible while reserving execution for run_foreground."""
        if self._retired or self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            raise RuntimeError("Agent Loop is already started")
        self._foreground_consumer_enabled = False

    async def _consume_foreground(self) -> None:
        while not self._retired and self.execution.foreground_input_admitted():
            inbound = await self._bus.get_inbound()
            if self._retired:
                break
            await self.run_foreground(inbound)

    def _status_input(self) -> RuntimeStatusInput:
        return session_runtime_status_input(
            self.session, configuration=self._configuration, context_builder=self._context_builder,
            tool_schemas=self.tool_schemas, generation_started_at=self._generation_started_at,
        )

    async def start(self) -> None:
        if self._retired:
            raise RuntimeError("Agent Loop is closed")
        await super().start()

    async def run_foreground(self, inbound: InboundMessage) -> None:
        self.execution._active = self
        try:
            await super().run_foreground(inbound)
        finally:
            self.execution._active = None

    async def close(self) -> None:
        """Stop the test input driver and close through the resident Session owner."""
        self._retired = True
        if self._aborted:
            await self.abort()
            return
        try:
            if not self.execution._closed:
                await self.execution.close()
        finally:
            tasks = self._owned_tasks()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._clear_owned_task_references()


class ExecutorControl:
    """Terminal test adapter to SessionExecution and an explicit requester."""

    def __init__(self, executor: DrivenExecutor) -> None:
        self._executor = executor
        self._requester: CallbackConfirmationRequester | None = None

    @property
    def has_active_run(self) -> bool:
        return self._executor.execution.has_active_run

    def foreground_input_admitted(self) -> bool:
        return self._executor.execution.foreground_input_admitted()

    async def cancel_active_run(self) -> None:
        await self._executor.cancel_active_run()

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        return self._executor.execution.project_foreground_conversation()

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        self._requester = CallbackConfirmationRequester(callback)
        self._executor.bind_confirmation_requester(self._request)

    async def _request(self, envelope: ConfirmationEnvelope) -> ConfirmationDecision:
        assert self._requester is not None
        return await self._requester.request(envelope.request)

    def respond_to_confirmation(self, confirmation_id: UUID, decision: ConfirmationDecision) -> None:
        if self._requester is None:
            raise ValueError("Confirmation response is late or unknown")
        self._requester.respond(confirmation_id, decision)
