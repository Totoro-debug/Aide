from __future__ import annotations

import pytest

from omni.terminal.keyboard import EnhancedKeyboardAction, EnhancedKeyboardAdapter


class RecordingTerminal:
    def __init__(self, *, fail_start: bool = False, fail_stop: bool = False) -> None:
        self.operations: list[tuple[str, str]] = []
        self.fail_start = fail_start
        self.fail_stop = fail_stop

    def write(self, value: str) -> None:
        self.operations.append(("write", value))

    def flush(self) -> None:
        self.operations.append(("flush", ""))

    def start_application_mode(self) -> None:
        self.write("application:start")
        self.write("\x1b[>25u")
        self.flush()
        if self.fail_start:
            raise RuntimeError("startup failed")

    def stop_application_mode(self) -> None:
        self.write("\x1b[<u")
        self.write("application:stop")
        self.flush()
        if self.fail_stop:
            raise RuntimeError("cleanup failed")


def test_kitty_enter_sequences_preserve_modifier_meaning() -> None:
    adapter = EnhancedKeyboardAdapter()

    assert adapter.parse("\r") == EnhancedKeyboardAction.SUBMIT
    assert adapter.parse("\n") == EnhancedKeyboardAction.NEWLINE
    assert adapter.parse("\x1b[13;2u") == EnhancedKeyboardAction.NEWLINE
    assert adapter.parse("\x1b[13;3u") == EnhancedKeyboardAction.NEWLINE
    assert adapter.parse("\x1b[13;1u") == EnhancedKeyboardAction.SUBMIT
    assert adapter.parse("\x1b[13;66u") == EnhancedKeyboardAction.NEWLINE
    assert adapter.parse("\x1b[13;67u") == EnhancedKeyboardAction.NEWLINE
    assert adapter.parse("\x1b[13;65u") == EnhancedKeyboardAction.SUBMIT
    assert adapter.parse("\x1b[13;131u") == EnhancedKeyboardAction.NEWLINE


def test_ordinary_carriage_return_is_not_treated_as_shift_enter() -> None:
    adapter = EnhancedKeyboardAdapter()

    assert adapter.parse("\r") is EnhancedKeyboardAction.SUBMIT
    assert adapter.parse("\x1b[13;6u") is None
    assert adapter.parse("\x1b[13;2x") is None
    assert adapter.parse("\x1b[13;" + ("9" * 5000) + "u") is None


def test_key_events_map_only_distinct_enter_chords_to_newline() -> None:
    adapter = EnhancedKeyboardAdapter()

    assert adapter.parse("shift+enter") is EnhancedKeyboardAction.NEWLINE
    assert adapter.parse("alt+enter") is EnhancedKeyboardAction.NEWLINE
    assert adapter.parse("enter") is EnhancedKeyboardAction.SUBMIT
    assert adapter.parse("ctrl+enter") is None


def test_capability_detection_is_conservative() -> None:
    assert EnhancedKeyboardAdapter.detect_capability({"WT_SESSION": "1"})
    assert EnhancedKeyboardAdapter.detect_capability({"TERM": "xterm-kitty"})
    assert EnhancedKeyboardAdapter.detect_capability({"TERM_PROGRAM": "WezTerm"})
    assert not EnhancedKeyboardAdapter.detect_capability({"TERM_PROGRAM": "Apple_Terminal"})
    assert not EnhancedKeyboardAdapter.detect_capability({})


def test_enable_and_restore_use_the_kitty_keyboard_stack() -> None:
    terminal = RecordingTerminal()
    adapter = EnhancedKeyboardAdapter(
        write=terminal.write,
        flush=terminal.flush,
        supports=lambda: True,
    )

    assert adapter.enable()
    assert adapter.enabled
    assert adapter.restore()
    assert not adapter.enabled
    assert terminal.operations == [
        ("write", "\x1b[>1u"),
        ("flush", ""),
        ("write", "\x1b[<u"),
        ("flush", ""),
    ]


def test_unsupported_terminal_keeps_keyboard_mode_untouched() -> None:
    terminal = RecordingTerminal()
    adapter = EnhancedKeyboardAdapter(
        write=terminal.write,
        flush=terminal.flush,
        supports=lambda: False,
    )

    assert not adapter.enable()
    assert not adapter.restore()
    assert terminal.operations == []


def test_driver_hooks_preserve_normal_lifecycle_order_and_ignore_duplicate_restore() -> None:
    driver = RecordingTerminal()
    adapter = EnhancedKeyboardAdapter.install_on_driver(driver, supports=lambda: True)

    driver.start_application_mode()
    driver.stop_application_mode()
    driver.stop_application_mode()

    keyboard_writes = [
        value
        for operation, value in driver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[<u"}
    ]
    assert keyboard_writes == ["\x1b[>1u", "\x1b[<u"]
    assert driver.operations.index(("write", "application:start")) < driver.operations.index(
        ("write", "\x1b[>1u")
    )
    assert driver.operations.index(("write", "\x1b[<u")) < driver.operations.index(
        ("write", "application:stop")
    )
    assert not adapter.enabled


def test_driver_hooks_restore_keyboard_mode_when_startup_fails() -> None:
    driver = RecordingTerminal(fail_start=True)
    adapter = EnhancedKeyboardAdapter.install_on_driver(driver, supports=lambda: True)

    with pytest.raises(RuntimeError, match="startup failed"):
        driver.start_application_mode()

    keyboard_writes = [
        value
        for operation, value in driver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[<u"}
    ]
    assert keyboard_writes == ["\x1b[>1u", "\x1b[<u"]
    assert not adapter.enabled


def test_driver_hooks_preserve_a_body_failure_when_terminal_stop_also_fails() -> None:
    driver = RecordingTerminal(fail_stop=True)
    adapter = EnhancedKeyboardAdapter.install_on_driver(driver, supports=lambda: True)

    with pytest.raises(RuntimeError, match="body failed") as raised:
        try:
            driver.start_application_mode()
            raise RuntimeError("body failed")
        finally:
            driver.stop_application_mode()

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "cleanup failed"
    keyboard_writes = [
        value
        for operation, value in driver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[<u"}
    ]
    assert keyboard_writes == ["\x1b[>1u", "\x1b[<u"]
    assert not adapter.enabled


def test_driver_hooks_leave_unsupported_terminal_keyboard_mode_untouched() -> None:
    driver = RecordingTerminal()
    adapter = EnhancedKeyboardAdapter.install_on_driver(driver, supports=lambda: False)

    driver.start_application_mode()
    driver.stop_application_mode()

    keyboard_writes = [
        value
        for operation, value in driver.operations
        if operation == "write" and value in {"\x1b[>1u", "\x1b[>25u", "\x1b[<u"}
    ]
    assert keyboard_writes == []
    assert ("write", "application:start") in driver.operations
    assert ("write", "application:stop") in driver.operations
    assert not adapter.enabled
