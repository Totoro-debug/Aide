"""Capability-gated enhanced keyboard support for Terminal Conversation."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Final, Protocol, cast

__all__ = ["EnhancedKeyboardAction", "EnhancedKeyboardAdapter"]

_KITTY_ENABLE_SEQUENCE: Final = "\x1b[>1u"
_KITTY_RESTORE_SEQUENCE: Final = "\x1b[<u"
_KITTY_ENABLE_PATTERN: Final = re.compile(r"\x1b\[>\d+u\Z")
_KITTY_ENTER_PATTERN: Final = re.compile(r"\x1b\[13(?:;(?P<modifiers>\d{1,3}))?u\Z")

_KITTY_SHIFT_MODIFIER: Final = 1
_KITTY_ALT_MODIFIER: Final = 2
_KITTY_LOCK_MODIFIERS: Final = 64 | 128
_KITTY_MAX_MODIFIER_PARAMETER: Final = 256

_KNOWN_TERM_PROGRAMS: Final = frozenset(
    {
        "alacritty",
        "ghostty",
        "iterm.app",
        "rio",
        "warpterminal",
        "wezterm",
        "windows terminal",
    }
)
_KNOWN_TERMS: Final = frozenset(
    {
        "alacritty",
        "foot",
        "kitty",
        "xterm-ghostty",
        "xterm-kitty",
    }
)


class EnhancedKeyboardAction(StrEnum):
    """Meaning of an Enter-family key event at the composition boundary."""

    SUBMIT = "submit"
    NEWLINE = "newline"


class _DriverHooks(Protocol):
    write: Callable[[str], None]
    flush: Callable[[], None]
    start_application_mode: Callable[[], None]
    stop_application_mode: Callable[[], None]


class EnhancedKeyboardAdapter:
    """Adapt Kitty keyboard reports while keeping Ctrl+J as the fallback."""

    def __init__(
        self,
        *,
        write: Callable[[str], None] | None = None,
        flush: Callable[[], None] | None = None,
        supports: Callable[[], bool] | None = None,
    ) -> None:
        self._write = write
        self._flush = flush
        self._supports = supports or self.detect_capability
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether this adapter has pushed a keyboard mode onto the terminal."""
        return self._enabled

    @staticmethod
    def detect_capability(environment: Mapping[str, str] | None = None) -> bool:
        """Conservatively identify hosts known to implement the Kitty protocol."""
        values = os.environ if environment is None else environment
        term_program = values.get("TERM_PROGRAM", "").casefold()
        term = values.get("TERM", "").casefold()
        return bool(
            values.get("WT_SESSION")
            or values.get("KITTY_WINDOW_ID")
            or term_program in _KNOWN_TERM_PROGRAMS
            or term in _KNOWN_TERMS
        )

    @staticmethod
    def is_enable_sequence(value: str) -> bool:
        """Return whether Textual is asking to push a Kitty keyboard mode."""
        return _KITTY_ENABLE_PATTERN.fullmatch(value) is not None

    @staticmethod
    def is_restore_sequence(value: str) -> bool:
        """Return whether Textual is asking to pop the Kitty keyboard mode."""
        return value == _KITTY_RESTORE_SEQUENCE

    @staticmethod
    def _action_for_modifiers(modifiers: int) -> EnhancedKeyboardAction | None:
        if not 1 <= modifiers <= _KITTY_MAX_MODIFIER_PARAMETER:
            return None
        actual_modifiers = modifiers - 1
        action_modifiers = actual_modifiers & ~_KITTY_LOCK_MODIFIERS
        if action_modifiers == 0:
            return EnhancedKeyboardAction.SUBMIT
        if action_modifiers in {_KITTY_SHIFT_MODIFIER, _KITTY_ALT_MODIFIER}:
            return EnhancedKeyboardAction.NEWLINE
        return None

    @classmethod
    def parse(cls, value: str) -> EnhancedKeyboardAction | None:
        """Parse one raw or Textual-normalized Enter-family report."""
        if value in {"\r", "enter"}:
            return EnhancedKeyboardAction.SUBMIT
        if value in {"\n", "ctrl+j", "shift+enter", "alt+enter"}:
            return EnhancedKeyboardAction.NEWLINE

        match = _KITTY_ENTER_PATTERN.fullmatch(value)
        if match is None:
            return None
        modifier_text = match.group("modifiers")
        modifiers = 1 if modifier_text is None else int(modifier_text)
        return cls._action_for_modifiers(modifiers)

    @classmethod
    def install_on_driver(
        cls,
        driver: object,
        *,
        supports: Callable[[], bool] | None = None,
    ) -> EnhancedKeyboardAdapter:
        """Gate Textual's Kitty mode writes while preserving its lifecycle ordering."""
        hooks = cast(_DriverHooks, driver)
        adapter = cls(write=hooks.write, flush=hooks.flush, supports=supports)
        original_write = hooks.write
        original_start = hooks.start_application_mode
        original_stop = hooks.stop_application_mode

        def write(value: str) -> None:
            if adapter.is_enable_sequence(value):
                adapter.enable()
            elif adapter.is_restore_sequence(value):
                adapter.restore()
            else:
                original_write(value)

        def start_application_mode() -> None:
            try:
                original_start()
            except BaseException as primary_error:
                try:
                    adapter.restore()
                except BaseException as cleanup_error:
                    raise primary_error from cleanup_error
                raise

        def stop_application_mode() -> None:
            body_error = sys.exception()
            try:
                original_stop()
            except BaseException as stop_error:
                try:
                    adapter.restore()
                except BaseException as restore_error:
                    stop_error.__cause__ = restore_error
                if body_error is not None:
                    raise body_error from stop_error
                raise stop_error
            else:
                try:
                    adapter.restore()
                except BaseException as cleanup_error:
                    if body_error is not None:
                        raise body_error from cleanup_error
                    raise

        hooks.write = write
        hooks.start_application_mode = start_application_mode
        hooks.stop_application_mode = stop_application_mode
        return adapter

    def enable(self) -> bool:
        """Push the modifier-aware mode only when the host capability is known."""
        if self._enabled:
            return True
        try:
            supported = self._supports()
        except Exception:
            supported = False
        if not supported or self._write is None or self._flush is None:
            return False

        self._write(_KITTY_ENABLE_SEQUENCE)
        self._enabled = True
        try:
            self._flush()
        except BaseException:
            try:
                self.restore()
            except BaseException:
                pass
            raise
        return True

    def restore(self) -> bool:
        """Pop the mode previously pushed by :meth:`enable`."""
        if not self._enabled:
            return False
        assert self._write is not None
        assert self._flush is not None
        try:
            self._write(_KITTY_RESTORE_SEQUENCE)
            self._flush()
        finally:
            self._enabled = False
        return True
