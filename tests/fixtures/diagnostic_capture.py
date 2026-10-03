"""In-memory Loguru capture for assertions about diagnostic events."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from loguru import logger

from omni.logging.process import configure_process_logging

if TYPE_CHECKING:
    from loguru import Message


class DiagnosticCapture:
    """Capture WARNING+ events without introducing a persistent logging contract."""

    def __init__(self) -> None:
        self._messages: list[str] = []
        self._event_messages: list[str] = []
        self._handler_id = logger.add(
            self._write,
            level="WARNING",
            catch=False,
            backtrace=False,
            diagnose=False,
        )

    @property
    def text(self) -> str:
        return "".join(self._messages)

    @property
    def event_text(self) -> str:
        return "\n".join(self._event_messages)

    @contextmanager
    def session(self, session_id: str) -> Iterator[None]:
        with logger.contextualize(session_id=session_id):
            yield

    def close(self) -> None:
        logger.remove(self._handler_id)

    def _write(self, message: Message) -> None:
        self._messages.append(str(message))
        self._event_messages.append(str(message.record["message"]))


def capture_diagnostics() -> DiagnosticCapture:
    return DiagnosticCapture()


@contextmanager
def configured_process_logging() -> Iterator[None]:
    configure_process_logging()
    try:
        yield
    finally:
        logger.remove()
