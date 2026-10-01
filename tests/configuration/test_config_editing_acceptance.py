from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigError, ConfigLoader
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import MINIMAL_VALID_CONFIG

_WORKER = """
import sys, time
from pathlib import Path
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ConfigRevisionConflict
home, ready, go, revision, operation = sys.argv[1:]
loader = ConfigLoader(AgentHome(Path(home)))
Path(ready).touch()
deadline = time.monotonic() + 20
while not Path(go).exists():
    if time.monotonic() > deadline: raise TimeoutError('writer barrier')
    time.sleep(0.01)
try:
    if operation == 'effort': loader.update_reasoning_effort('high')
    else: loader.patch_editable_fields(revision, {'memory': {'batch_size': int(operation)}})
    print('saved')
except ConfigRevisionConflict:
    print('conflict')
"""


def _loader(tmp_path: Path) -> ConfigLoader:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    content = (
        "# acceptance comment\n" + MINIMAL_VALID_CONFIG + "\n[future]\nvalue = 9 # untouched\n"
    )
    (home.path / "config.toml").write_text(content, encoding="utf-8")
    return ConfigLoader(home)


def _writers(loader: ConfigLoader, tmp_path: Path, operations: tuple[str, str]) -> list[str]:
    go = tmp_path / "go"
    revision = loader.revision()
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WORKER,
                str(loader.agent_home.path),
                str(tmp_path / f"ready-{index}"),
                str(go),
                revision,
                operation,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index, operation in enumerate(operations)
    ]
    try:
        deadline = time.monotonic() + 20
        while not all((tmp_path / f"ready-{index}").exists() for index in range(2)):
            assert time.monotonic() < deadline, "writers did not reach barrier"
            time.sleep(0.01)
        go.touch()
        outcomes = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            assert "minimal-secret" not in stdout + stderr
            outcomes.append(stdout.strip())
        return outcomes
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)


def test_real_crossprocess_cas_only_one_writer_succeeds(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    assert sorted(_writers(loader, tmp_path, ("11", "12"))) == ["conflict", "saved"]
    assert loader.load().memory.batch_size in {11, 12}
    assert "# acceptance comment" in loader.path.read_text(encoding="utf-8")
    assert "value = 9 # untouched" in loader.path.read_text(encoding="utf-8")


def test_crossprocess_effort_and_cas_preserve_latest_bytes(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    patch, effort = _writers(loader, tmp_path, ("13", "effort"))
    assert effort == "saved"
    assert patch in {"saved", "conflict"}
    configuration = loader.load()
    assert configuration.models.routes["default"].reasoning_effort == "high"
    assert configuration.memory.batch_size == (13 if patch == "saved" else 10)
    assert "value = 9 # untouched" in loader.path.read_text(encoding="utf-8")


def test_atomic_write_failure_keeps_configuration_bytes_and_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _loader(tmp_path)
    revision = loader.editable_snapshot().revision
    before, diagnostics = loader.path.read_bytes(), loader.diagnostics

    def fail(target: Path, content: str) -> None:
        raise OSError("write canary")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail)
    with pytest.raises(OSError):
        loader.patch_editable_fields(revision, {"memory": {"batch_size": 14}})
    assert loader.path.read_bytes() == before
    assert loader.diagnostics == diagnostics


def test_invalid_untouched_candidate_keeps_original_bytes(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    content = loader.path.read_text(encoding="utf-8").replace(
        "context_window = 8192", "context_window = 1"
    )
    assert content != loader.path.read_text(encoding="utf-8")
    loader.path.write_text(content, encoding="utf-8")
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.patch_editable_fields(loader.revision(), {"memory": {"batch_size": 15}})
    assert loader.path.read_bytes() == before
