"""Reusable AgentLoop public-seam helpers."""

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
