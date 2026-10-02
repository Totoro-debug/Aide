"""Platform refusal precedes logging, runtime state, and validation artifacts."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from myclaw.utils import platform


@pytest.mark.parametrize("host", ["posix", "unknown"])
def test_runtime_host_check_accepts_only_windows(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.setattr(platform, "os", SimpleNamespace(name=host))
    assert not platform.is_windows_host()


@pytest.mark.parametrize(
    "module_name, entry, exits",
    [
        ("myclaw.terminal.process_entry", "run", True),
        ("myclaw.service.process", "run", False),
        ("scripts.release_validation", "main", False),
        ("scripts.installed_web_validation", "main", True),
    ],
)
def test_unsupported_host_stops_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    module_name: str,
    entry: str,
    exits: bool,
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "is_windows_host", lambda: False)
    monkeypatch.chdir(tmp_path)
    if module_name == "myclaw.terminal.process_entry":
        def forbidden_logging() -> None:
            pytest.fail("Platform refusal must precede logging initialization")

        monkeypatch.setattr(module, "configure_process_logging", forbidden_logging)
    if exits:
        with pytest.raises(SystemExit) as error:
            getattr(module, entry)()
        assert error.value.code == 1
    else:
        assert getattr(module, entry)([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "MyClaw requires Windows.\n"
    assert tuple(tmp_path.iterdir()) == ()
