from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from loguru import logger

from omni.agent.workspace_state import WorkspaceState
from omni.logging.session import session_log, without_session_log


def _session_id() -> str:
    return f"20260802-120000-000000_{uuid4()}"


def _state(tmp_path: Path) -> WorkspaceState:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return WorkspaceState(workspace)


def _write_boundary_warning(message: str) -> None:
    logger.warning("{}", message)


def _fail_loguru_add(*args: object, **kwargs: object) -> int:
    del args, kwargs
    raise OSError("injected add failure")


def _windows_acl_inheritance(path: Path) -> tuple[bool, int]:
    environment = dict(os.environ)
    environment["OMNI_ACL_PATH"] = str(path)
    script = (
        "$acl = Get-Acl -LiteralPath $env:OMNI_ACL_PATH; "
        "$count = @($acl.Access | Where-Object { $_.IsInherited }).Count; "
        'Write-Output ("{0}|{1}" -f $acl.AreAccessRulesProtected, $count)'
    )
    completed = subprocess.run(
        ["pwsh.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    protected, inherited_count = completed.stdout.strip().split("|")
    return protected.casefold() == "true", int(inherited_count)


def test_session_log_directory_is_lazy_and_filters_records(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    assert state.logs_directory == state.path / "logs"
    assert not state.logs_directory.exists()

    with session_log(state, session_id):
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")

    active = state.logs_directory / f"{session_id}.log"
    content = active.read_text(encoding="utf-8")
    assert "warning" in content
    assert "error" in content
    assert "debug" not in content
    assert "info" not in content


def test_schedule_session_log_uses_the_shared_workspace_log_partition(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = f"schedule_{uuid4()}"

    with session_log(state, session_id):
        logger.warning("scheduled warning")

    assert "scheduled warning" in (state.logs_directory / f"{session_id}.log").read_text(
        encoding="utf-8"
    )
    assert not state.schedule_sessions_directory.exists()


def test_session_log_contexts_do_not_cross_write(tmp_path: Path) -> None:
    state = _state(tmp_path)
    first, second = _session_id(), _session_id()

    with session_log(state, first):
        logger.warning("first")
    with session_log(state, second):
        logger.warning("second")

    assert "first" in (state.logs_directory / f"{first}.log").read_text(encoding="utf-8")
    assert "second" not in (state.logs_directory / f"{first}.log").read_text(encoding="utf-8")
    assert "second" in (state.logs_directory / f"{second}.log").read_text(encoding="utf-8")


def test_session_log_ignores_third_party_standard_library_records(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    with session_log(state, session_id):
        logging.getLogger("third_party").error("third-party diagnostic")
        logger.error("Omni diagnostic")

    content = (state.logs_directory / f"{session_id}.log").read_text(encoding="utf-8")
    assert "Omni diagnostic" in content
    assert "third-party diagnostic" not in content


def test_without_session_log_temporarily_clears_session_ownership(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    with session_log(state, session_id):
        with without_session_log():
            logger.error("unowned failure")
        logger.error("owned failure")

    content = (state.logs_directory / f"{session_id}.log").read_text(encoding="utf-8")
    assert "unowned failure" not in content
    assert "owned failure" in content


def test_session_log_does_not_create_a_file_without_a_warning(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    with session_log(state, session_id):
        logger.info("ordinary progress")

    assert state.logs_directory.exists()
    assert not (state.logs_directory / f"{session_id}.log").exists()


def test_session_log_retains_an_ordinary_exception_traceback(tmp_path: Path) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    with session_log(state, session_id):
        try:
            raise RuntimeError("technical detail")
        except RuntimeError as error:
            logger.opt(exception=error).error("operation failed")

    content = (state.logs_directory / f"{session_id}.log").read_text(encoding="utf-8")
    assert "operation failed" in content
    assert "Traceback (most recent call last)" in content
    assert "RuntimeError: technical detail" in content


def test_session_log_rejects_invalid_session_id_before_creating_state(tmp_path: Path) -> None:
    state = _state(tmp_path)

    with pytest.raises(ValueError, match="valid Session ID"):
        with session_log(state, "invalid"):
            pass

    assert not state.logs_directory.exists()


def test_session_log_directory_failure_isolated_and_retried(
    tmp_path: Path, monkeypatch: Any
) -> None:
    state = _state(tmp_path)
    session_id = _session_id()
    original = Path.mkdir
    calls = 0

    def fail_once(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal calls
        if path == state.logs_directory and calls == 0:
            calls += 1
            raise OSError("blocked")
        return original(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", fail_once)
    with session_log(state, session_id):
        logger.warning("first")
    assert not state.logs_directory.exists()

    with session_log(state, session_id):
        logger.warning("second")
    assert "second" in (state.logs_directory / f"{session_id}.log").read_text(encoding="utf-8")


def test_session_log_rejects_a_junction_logs_directory_without_stopping_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = _state(tmp_path)
    state.path.mkdir(parents=True)
    outside = tmp_path / "outside-logs"
    outside.mkdir()
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(state.logs_directory), str(outside)],
        check=True,
        capture_output=True,
        text=True,
    )
    business_result = "not-run"

    with session_log(state, _session_id()):
        logger.warning("must not escape")
        business_result = "continued"

    assert business_result == "continued"
    assert tuple(outside.iterdir()) == ()
    diagnostic = capsys.readouterr().err
    assert "Session Log failure: UnsafeFilesystemPath" in diagnostic
    assert "Traceback" not in diagnostic


def test_session_log_rejects_a_hard_linked_active_file_and_retries(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    state.logs_directory.mkdir(parents=True)
    session_id = _session_id()
    active = state.logs_directory / f"{session_id}.log"
    outside = tmp_path / "outside.log"
    original = b"outside bytes must remain unchanged\x00\xff"
    outside.write_bytes(original)
    active.hardlink_to(outside)
    business_result = "not-run"

    with session_log(state, session_id):
        logger.warning("must not follow hard link")
        business_result = "continued"

    assert business_result == "continued"
    assert active.read_bytes() == original
    assert outside.read_bytes() == original

    active.unlink()
    with session_log(state, session_id):
        logger.warning("retry succeeded")
    assert "retry succeeded" in active.read_text(encoding="utf-8")
    assert outside.read_bytes() == original


def test_session_log_add_failure_isolated_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    session_id = _session_id()
    active = state.logs_directory / f"{session_id}.log"

    with monkeypatch.context() as patch:
        patch.setattr(logger, "add", _fail_loguru_add)
        with session_log(state, session_id):
            business_result = "continued"

    assert business_result == "continued"
    assert not active.exists()
    with session_log(state, session_id):
        logger.warning("retry succeeded")
    assert "retry succeeded" in active.read_text(encoding="utf-8")


def test_session_log_failure_report_is_latched_until_a_successful_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _state(tmp_path)

    with session_log(state, _session_id()):
        pass
    capsys.readouterr()

    with monkeypatch.context() as patch:
        patch.setattr(logger, "add", _fail_loguru_add)
        with session_log(state, _session_id()):
            pass
        with session_log(state, _session_id()):
            pass
    first_failures = capsys.readouterr().err
    assert first_failures.count("Session Log failure: OSError") == 1
    assert "Traceback" not in first_failures

    with session_log(state, _session_id()):
        pass
    with monkeypatch.context() as patch:
        patch.setattr(logger, "add", _fail_loguru_add)
        with session_log(state, _session_id()):
            pass
    after_success = capsys.readouterr().err
    assert after_success.count("Session Log failure: OSError") == 1
    assert "Traceback" not in after_success


def test_session_log_open_failure_isolated_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    session_id = _session_id()
    active = state.logs_directory / f"{session_id}.log"
    original_open = os.open

    def fail_log_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        candidate = Path(os.fsdecode(path))
        if candidate.name == active.name and candidate.parent.resolve() == active.parent.resolve():
            raise OSError("injected open failure")
        return original_open(path, flags, mode)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", fail_log_open)
        with session_log(state, session_id):
            logger.warning("write is isolated")
            business_result = "continued"

    assert business_result == "continued"
    assert not active.exists()
    with session_log(state, session_id):
        logger.warning("retry succeeded")
    assert "retry succeeded" in active.read_text(encoding="utf-8")


def test_session_log_write_failure_does_not_stop_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    session_id = _session_id()

    def fail_write(_sink: object, _message: object) -> None:
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        patch.setattr("loguru._file_sink.FileSink.write", fail_write)
        with session_log(state, session_id):
            logger.warning("write is isolated")
            business_result = "continued"

    assert business_result == "continued"


def test_session_log_rotation_failure_does_not_stop_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import omni.logging.session as session_logging

    state = _state(tmp_path)
    state.logs_directory.mkdir(parents=True)
    session_id = _session_id()
    active = state.logs_directory / f"{session_id}.log"
    original = b"x" * 256
    active.write_bytes(original)
    original_rename = os.rename

    def fail_rotation(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        if Path(os.fsdecode(source)).resolve() == active.resolve():
            raise OSError("injected rotation failure")
        original_rename(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(session_logging, "_ROTATION_BYTES", 128)
        patch.setattr(os, "rename", fail_rotation)
        with session_log(state, session_id):
            logger.warning("rotation is isolated")
            business_result = "continued"

    assert business_result == "continued"
    assert active.read_bytes() == original


def test_session_log_preserves_windows_acl_inheritance(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.path.mkdir(parents=True)
    session_id = _session_id()
    state_acl_before = _windows_acl_inheritance(state.path)

    with session_log(state, session_id):
        logger.warning("inherits workspace permissions")

    active = state.logs_directory / f"{session_id}.log"
    assert _windows_acl_inheritance(state.path) == state_acl_before
    for path in (state.logs_directory, active):
        protected, inherited_count = _windows_acl_inheritance(path)
        assert not protected
        assert inherited_count > 0


def test_session_log_rotates_at_exact_byte_boundary_and_retains_one_history(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    session_id = _session_id()
    active = state.logs_directory / f"{session_id}.log"
    rotation_bytes = 10_485_760

    with session_log(state, session_id):
        _write_boundary_warning("boundary")
    record_size = active.stat().st_size

    active.write_bytes(b"x" * (rotation_bytes - record_size))
    with session_log(state, session_id):
        _write_boundary_warning("boundary")

    assert active.stat().st_size == rotation_bytes
    assert list(state.logs_directory.glob(f"{session_id}.*.log")) == []

    with session_log(state, session_id):
        _write_boundary_warning("overflow")

    for marker in (b"rotation-one", b"rotation-two"):
        active.write_bytes(b"x" * (rotation_bytes - len(marker)) + marker)
        with session_log(state, session_id):
            _write_boundary_warning("overflow")

    files = list(state.logs_directory.glob(f"{session_id}*.log"))
    history = [path for path in files if path != active]
    assert active.exists()
    assert len(history) == 1
    assert "overflow" in active.read_text(encoding="utf-8")
    with history[0].open("rb") as stream:
        stream.seek(-len(b"rotation-two"), os.SEEK_END)
        assert stream.read() == b"rotation-two"
