"""Reusable AgentRunExecutor public-seam helpers."""

import asyncio
from typing import Any

from omni.agent.loop import AgentRunExecutor
from omni.agent.message_bus import InboundMessage, MessageBus, OutboundMessage


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
        if self._closed or self._aborted or self._closing or self._close_task is not None:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            raise RuntimeError("Agent Loop is already started")
        self._foreground_consumer_enabled = False

    async def _consume_foreground(self) -> None:
        while not self._closing:
            inbound = await self._bus.get_inbound()
            if self._closing:
                break
            await self.run_foreground(inbound)
