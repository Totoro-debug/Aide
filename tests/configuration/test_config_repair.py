from __future__ import annotations

import ctypes
import json
import os
import subprocess
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigError, ConfigLoader, ConfigRevisionConflict
from aide.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import MINIMAL_VALID_CONFIG
from tests.configuration.test_config_editing import FULL_EDITABLE_CONFIG
from tests.fixtures.model_configuration import TEST_MODEL_PARAMETERS, complete_model_settings


def _loader(tmp_path: Path, content: bytes) -> ConfigLoader:
    home = AgentHome(tmp_path / "home")
    home.initialize()
    loader = ConfigLoader(home)
    loader.path.write_bytes(content)
    return loader


def _repair_fields(loader: ConfigLoader) -> dict[str, Any]:
    snapshot = loader.web_snapshot()
    fields = cast(dict[str, Any], json.loads(json.dumps(dict(snapshot.fields))))
    complete_model_settings(fields["models"])
    for provider in fields["models"]["providers"].values():
        provider.pop("api_key")
    for server in fields["mcp"].values():
        if server["transport"] == "streamable-http":
            for name in ("command", "args", "cwd"):
                server.pop(name)
        else:
            server.pop("url")
            server.pop("headers")
    return fields


@pytest.mark.parametrize(
    "invalid",
    [
        'models = ["retired-model"]',
        "context_window = 16384",
        "call_timeout = 60",
    ],
)
def test_invalid_row_retains_siblings_secrets_and_unknown_fields(
    tmp_path: Path, invalid: str
) -> None:
    content = FULL_EDITABLE_CONFIG.replace(invalid, invalid.split(" = ")[0] + ' = "broken"', 1)
    content += "\n[future]\nvalue = 303\n"
    loader = _loader(tmp_path, content.encode())
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    assert set(cast(Mapping[str, object], snapshot.fields["models"]["providers"])) == {
        "primary",
        "retired",
    }
    assert set(cast(Mapping[str, object], snapshot.fields["models"]["routes"])) == {
        "title",
        "chat",
        "memory",
        "schedule",
    }
    assert set(snapshot.fields["mcp"]) == {"http", "stdio"}
    assert "canary" not in str(snapshot.fields)
    result = loader.repair_editable_fields(snapshot.revision, _repair_fields(loader))
    assert result.configuration.models.providers["retired"].api_key == "retired-secret-canary-302"
    assert result.configuration.mcp["http"].headers["Authorization"] == "mcp-header-canary-302"
    saved = loader.path.read_text(encoding="utf-8")
    assert "# preserve this comment" in saved
    assert "[future]" in saved and "value = 303" in saved


def test_valid_projection_uses_original_optional_defaults(tmp_path: Path) -> None:
    content = MINIMAL_VALID_CONFIG.replace('reasoning_effort = "mid"\n', "")
    loader = _loader(tmp_path, content.encode())
    assert loader.web_snapshot().state == "active"
    assert loader.web_snapshot().configuration == loader.load_for_startup()


def test_repair_preserves_valid_default_chat_workspace(tmp_path: Path) -> None:
    content = MINIMAL_VALID_CONFIG + '\n[memory]\nbatch_size = "bad"\n'
    content += '\n[web]\ndefault_chat_workspace = "D:/custom-chat"\n'
    loader = _loader(tmp_path, content.encode())
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    assert snapshot.fields["web"]["default_chat_workspace"] == "D:/custom-chat"
    loader.repair_editable_fields(snapshot.revision, _repair_fields(loader))
    assert loader.load().web.default_chat_workspace == "D:/custom-chat"


def test_invalid_untouched_field_never_activates_projection_fallback(tmp_path: Path) -> None:
    loader = _loader(tmp_path, (MINIMAL_VALID_CONFIG + '\n[memory]\nbatch_size = "bad"\n').encode())
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.repair_editable_fields(snapshot.revision, {"runtime": {"max_iterations": 80}})
    assert loader.path.read_bytes() == before


def test_invalid_collection_names_remain_visible_until_explicitly_removed(tmp_path: Path) -> None:
    content = FULL_EDITABLE_CONFIG.replace("models.providers.retired", "models.providers.BAD")
    content = content.replace("mcp.servers.stdio", 'mcp.servers."bad name"')
    loader = _loader(tmp_path, content.encode())
    fields = loader.web_snapshot().fields
    assert "BAD" in cast(Mapping[str, object], fields["models"]["providers"])
    assert "bad name" in fields["mcp"]
    assert "retired-secret" not in str(fields)


def test_invalid_root_table_can_be_repaired_structurally(tmp_path: Path) -> None:
    content = 'runtime = "bad"\n' + MINIMAL_VALID_CONFIG.replace(
        "[runtime]\ncompact_ratio = 0.9\n", ""
    )
    loader = _loader(tmp_path, content.encode())
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    fields = _repair_fields(loader)
    loader.repair_editable_fields(snapshot.revision, fields)
    assert loader.web_snapshot().state == "active"


def test_invalid_mcp_nested_table_can_be_repaired_without_losing_row(tmp_path: Path) -> None:
    content = FULL_EDITABLE_CONFIG.replace(
        'headers = { Authorization = "mcp-header-canary-302" }', 'headers = "bad"'
    )
    loader = _loader(tmp_path, content.encode())
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    fields = _repair_fields(loader)
    assert "http" in fields["mcp"]
    loader.repair_editable_fields(snapshot.revision, fields)
    assert loader.web_snapshot().state == "active"


def _malformed_repair(
    loader: ConfigLoader,
) -> tuple[str, Mapping[str, object], Mapping[str, object]]:
    snapshot = loader.web_snapshot()
    fields = _repair_fields(loader)
    providers = fields["models"]["providers"]
    providers["openai-local"]["base_url"] = "http://127.0.0.1/v1"
    providers["openai-local"]["models"] = {"test": dict(TEST_MODEL_PARAMETERS)}
    for route in fields["models"]["routes"].values():
        route["model"] = "test"
    return (
        snapshot.revision,
        fields,
        {"models.providers.openai-local.api_key": {"action": "replace", "value": "new-secret"}},
    )


def _windows_dacl(path: Path) -> str:
    from ctypes import wintypes

    api = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get = api.GetFileSecurityW
    get.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get.restype = wintypes.BOOL
    needed = wintypes.DWORD()
    get(str(path), 4, None, 0, ctypes.byref(needed))
    buffer = ctypes.create_string_buffer(needed.value)
    assert get(str(path), 4, buffer, needed.value, ctypes.byref(needed))
    convert = api.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    convert.restype = wintypes.BOOL
    text = ctypes.c_void_p()
    assert convert(buffer, 1, 4, ctypes.byref(text), None)
    free = kernel.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    try:
        return ctypes.wstring_at(text)
    finally:
        free(text)


def test_malformed_non_utf8_backup_is_exact_and_private(tmp_path: Path) -> None:
    original = b'api_key="old-secret"\r\n\xff\x00[broken'
    loader = _loader(tmp_path, original)
    result = loader.repair_editable_fields(*_malformed_repair(loader))
    backup = loader.agent_home.path / f"config.toml.backup.{sha256(original).hexdigest()}"
    assert backup.read_bytes() == original
    assert result.backup_id == f"sha256:{sha256(original).hexdigest()}"
    assert "old-secret" not in loader.path.read_text()
    assert "new-secret" in loader.path.read_text()
    assert _windows_dacl(backup) == "D:P(A;;FA;;;OW)(A;;FA;;;SY)"


def test_malformed_secret_keep_cannot_recover_secret_from_source(tmp_path: Path) -> None:
    original = b'api_key="old-secret"\n[broken'
    loader = _loader(tmp_path, original)
    revision, fields, _ = _malformed_repair(loader)
    with pytest.raises(ConfigError):
        loader.repair_editable_fields(
            revision, fields, {"models.providers.openai-local.api_key": {"action": "keep"}}
        )
    assert loader.path.read_bytes() == original
    assert not tuple(loader.agent_home.path.glob("config.toml.backup.*"))


def test_backup_collision_preserves_existing_content(tmp_path: Path) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)
    collision = loader.agent_home.path / f"config.toml.backup.{sha256(original).hexdigest()}"
    collision.write_bytes(b"unrelated")
    loader.repair_editable_fields(*_malformed_repair(loader))
    assert collision.read_bytes() == b"unrelated"
    assert any(
        path.read_bytes() == original
        for path in loader.agent_home.path.glob("config.toml.backup.*")
    )


def test_backup_sync_failure_leaves_original_and_cleans_partial_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)
    args = _malformed_repair(loader)

    def fail(_descriptor: int) -> None:
        raise OSError("sync failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "sync_file", fail)
    with pytest.raises(OSError):
        loader.repair_editable_fields(*args)
    assert loader.path.read_bytes() == original
    assert not tuple(loader.agent_home.path.glob(".config-backup-*"))
    assert not tuple(loader.agent_home.path.glob("config.toml.backup.*"))


def test_backup_time_external_change_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _loader(tmp_path, b"[broken")
    args = _malformed_repair(loader)
    real_sync = HOST_FILESYSTEM.sync_parent_directory

    def change(path: Path) -> None:
        real_sync(path)
        loader.path.write_bytes(b"external-change")

    monkeypatch.setattr(HOST_FILESYSTEM, "sync_parent_directory", change)
    with pytest.raises(ConfigRevisionConflict):
        loader.repair_editable_fields(*args)
    assert loader.path.read_bytes() == b"external-change"


def test_backup_existing_directory_blocks_repair_without_overwrite(tmp_path: Path) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)
    blocker = loader.agent_home.path / f"config.toml.backup.{sha256(original).hexdigest()}"
    blocker.mkdir()
    with pytest.raises(OSError):
        loader.repair_editable_fields(*_malformed_repair(loader))
    assert loader.path.read_bytes() == original


def test_unsupported_mcp_secret_environment_is_not_silently_removed(tmp_path: Path) -> None:
    content = FULL_EDITABLE_CONFIG.replace(
        'command = "python"', 'command = "python"\nenv = { KEY = "env-secret" }'
    )
    loader = _loader(tmp_path, content.encode())
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    assert "stdio" in snapshot.fields["mcp"]
    assert "env-secret" not in str(snapshot.fields)
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError):
        loader.repair_editable_fields(snapshot.revision, _repair_fields(loader))
    assert loader.path.read_bytes() == before


def test_incompatible_secret_requires_explicit_transport_change(tmp_path: Path) -> None:
    content = FULL_EDITABLE_CONFIG.replace(
        'command = "python"', 'command = "python"\nheaders = { Authorization = "hidden-secret" }'
    )
    loader = _loader(tmp_path, content.encode())
    snapshot = loader.web_snapshot()
    assert snapshot.state == "invalid"
    fields = _repair_fields(loader)
    fields["runtime"]["max_iterations"] = 80
    before = loader.path.read_bytes()
    with pytest.raises(ConfigError) as error:
        loader.repair_editable_fields(snapshot.revision, fields)
    assert "hidden-secret" not in str(error.value)
    assert loader.path.read_bytes() == before
    fields["mcp"]["stdio"] = {
        "enabled": False,
        "transport": "streamable-http",
        "url": "http://127.0.0.1",
        "headers": {"Authorization": {"configured": True}},
    }
    result = loader.repair_editable_fields(snapshot.revision, fields)
    assert result.configuration.mcp["stdio"].headers["Authorization"] == "hidden-secret"


def test_reused_backup_is_restricted_before_repair(tmp_path: Path) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)
    backup = loader.agent_home.path / f"config.toml.backup.{sha256(original).hexdigest()}"
    backup.write_bytes(original)
    loader.repair_editable_fields(*_malformed_repair(loader))
    assert backup.read_bytes() == original
    assert _windows_dacl(backup) == "D:P(A;;FA;;;OW)(A;;FA;;;SY)"


@pytest.mark.parametrize("link_source", [False, True])
def test_hardlinked_source_or_backup_is_rejected_without_external_write(
    tmp_path: Path, link_source: bool
) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)
    external = tmp_path / "external"
    external.write_bytes(original)
    if link_source:
        loader.path.unlink()
        os.link(external, loader.path)
    else:
        os.link(
            external, loader.agent_home.path / f"config.toml.backup.{sha256(original).hexdigest()}"
        )
    with pytest.raises(OSError):
        loader.repair_editable_fields(*_malformed_repair(loader))
    assert external.read_bytes() == original
    assert loader.path.read_bytes() == original


def test_private_acl_failure_leaves_original_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)

    def fail(_path: Path) -> None:
        raise PermissionError("ACL refused")

    monkeypatch.setattr(HOST_FILESYSTEM, "protect_private_file", fail)
    with pytest.raises(PermissionError):
        loader.repair_editable_fields(*_malformed_repair(loader))
    assert loader.path.read_bytes() == original
    assert not tuple(loader.agent_home.path.glob(".config-backup-*"))


def test_replacement_failure_retains_original_and_exact_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)

    def fail(_path: Path, _content: str) -> None:
        raise PermissionError("replace refused")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", fail)
    with pytest.raises(PermissionError):
        loader.repair_editable_fields(*_malformed_repair(loader))
    assert loader.path.read_bytes() == original
    assert any(
        path.read_bytes() == original
        for path in loader.agent_home.path.glob("config.toml.backup.*")
    )


def test_redirected_backup_target_rejects_repair_without_touching_external_directory(
    tmp_path: Path,
) -> None:
    original = b"[broken"
    loader = _loader(tmp_path, original)
    external = tmp_path / "outside"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_bytes(b"untouched")
    target = loader.agent_home.path / f"config.toml.backup.{sha256(original).hexdigest()}"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target), str(external)],
        check=True,
        capture_output=True,
    )
    try:
        with pytest.raises(OSError):
            loader.repair_editable_fields(*_malformed_repair(loader))
        assert loader.path.read_bytes() == original
        assert marker.read_bytes() == b"untouched"
        assert set(external.iterdir()) == {marker}
    finally:
        target.rmdir()
