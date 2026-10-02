"""Validate normal installed Web/CLI lifecycles, separate from the Node controller.

Run with python -m scripts.installed_web_validation --output <new external directory>.
Requires the existing web/ Playwright development dependencies on the controller.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import sys
from hashlib import sha256
from pathlib import Path

from scripts.release_validation import (
    ROOT,
    _artifact_environment,
    _assert_sdist_web_assets,
    _assert_wheel_web_assets,
    _extract_sdist,
    _run_command,
    _smoke_installed_wheel,
    _source_web_asset_bytes,
)
from web.scripts.e2e_service import _start_fixture_provider


def build_distributions(output: Path) -> tuple[Path, Path]:
    expected = _source_web_asset_bytes()
    direct = output / "distribution"
    direct.mkdir()
    environment = _artifact_environment()
    _run_command(
        [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(direct)],
        env=environment,
    )
    (wheel,) = direct.glob("*.whl")
    (sdist,) = direct.glob("*.tar.gz")
    _assert_wheel_web_assets(wheel, expected)
    _assert_sdist_web_assets(sdist, expected)
    source = _extract_sdist(sdist, output / "extracted")
    rebuilt = output / "rebuilt"
    rebuilt.mkdir()
    _run_command(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(rebuilt)],
        cwd=source,
        env=environment,
    )
    (rebuilt_wheel,) = rebuilt.glob("*.whl")
    _assert_wheel_web_assets(rebuilt_wheel, expected)
    return wheel, rebuilt_wheel


async def validate_install(wheel: Path, root: Path, node: str) -> dict[str, object]:
    installed = await asyncio.to_thread(_smoke_installed_wheel, wheel, root)
    entry = str(installed["entry_point"])
    python = Path(entry).parent / ("python.exe" if os.name == "nt" else "python")
    profile = root / "用户配置"
    home = profile / ".myclaw"
    home.mkdir(parents=True)
    workspace = root / "workspace"
    workspace.mkdir()
    capture = root / ("capture-browser.cmd" if os.name == "nt" else "capture-browser.sh")
    url_file = root / "browser-url.txt"
    if os.name == "nt":
        capture.write_text(f'@echo off\n>"{url_file}" echo %*\n', encoding="utf-8")
    else:
        capture.write_text('#!/bin/sh\nprintf "%s" "$1" > "$MYCLAW_CAPTURE_URL"\n')
        capture.chmod(0o700)
    environment = {
        **_artifact_environment(),
        "USERPROFILE": str(profile),
        "HOME": str(profile),
        "BROWSER": str(capture),
        "MYCLAW_CAPTURE_URL": str(url_file),
    }
    provider, base_url = await _start_fixture_provider()
    config = f'''[models.providers.fixture]
protocol = "openai-compatible"
base_url = "{base_url}"
api_key = "installed-fixture-key"
models = ["small-model"]

[models.routes.default]
provider_id = "fixture"
model = "small-model"
context_window = 8192
max_output = 1024
temperature = 0
timeout = 30
'''
    discovery_path = home / "service.json"
    try:
        (home / "config.toml").write_text(config, encoding="utf-8")
        # The production entry owns a fixed port. Refuse an occupied port without stopping it.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 8765))
        await asyncio.to_thread(
            _run_command, [entry, "web"], cwd=workspace, env=environment, timeout=60
        )
        launch_url = url_file.read_text(encoding="utf-8").strip().strip('"')
        assert launch_url.startswith("http://127.0.0.1:8765/#ticket=")
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        browser_environment = {
            **os.environ,
            "MYCLAW_E2E_URL": "http://127.0.0.1:8765",
            "MYCLAW_E2E_TICKET": launch_url.split("#ticket=", 1)[1],
            "MYCLAW_E2E_WORKSPACE": str(workspace),
            "MYCLAW_E2E_OUTPUT": str(root),
            "MYCLAW_E2E_INSTANCE": discovery["service_instance_id"],
        }
        try:
            await asyncio.to_thread(
                _run_command,
                [node, str(ROOT / "web/scripts/installed-web-e2e.mjs")],
                cwd=workspace,
                env=browser_environment,
                timeout=120,
            )
        except RuntimeError as error:
            raise RuntimeError(
                str(error).replace(launch_url.split("#ticket=", 1)[1], "[redacted]")
            ) from None
        cli_result = await asyncio.to_thread(
            _run_command,
            [
                str(python),
                "-I",
                str(ROOT / "web/scripts/installed_cli_probe.py"),
                discovery["service_instance_id"],
                str(discovery["pid"]),
            ],
            cwd=workspace,
            env=environment,
            timeout=60,
        )
        cli_evidence = json.loads(cli_result.stdout.strip().splitlines()[-1])
        assert cli_evidence["marker"] == "INSTALLED_CLI_CONNECT_OK"
        assert json.loads(discovery_path.read_text(encoding="utf-8")) == discovery
        await stop_install(entry, workspace, environment, discovery_path)
        transcripts = list((workspace / ".myclaw/sessions").glob("*.jsonl"))
        messages = [
            json.loads(line)
            for path in transcripts
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert (
            sum(
                message.get("role") == "user"
                and message.get("content") == "installed package conversation\nstreaming markdown"
                for message in messages
            )
            == 1
        )
        assert any(
            message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("# Streamed answer")
            and message.get("error") is None
            for message in messages
        )
        return {
            **installed,
            "wheel_path": str(wheel),
            "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
            "web": json.loads((root / "browser.json").read_text(encoding="utf-8")),
            "cli": cli_evidence,
            "stop": "passed",
            "persisted_conversation": "passed",
        }
    finally:
        primary = sys.exception()
        cleanup_errors: list[BaseException] = []
        try:
            if discovery_path.exists():
                await stop_install(entry, workspace, environment, discovery_path)
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            await provider.cleanup()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            url_file.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "installed validation cleanup failed",
                ([primary] if primary is not None else []) + cleanup_errors,
            )


async def stop_install(
    entry: str, workspace: Path, environment: dict[str, str], discovery: Path
) -> None:
    await asyncio.to_thread(
        _run_command, [entry, "service", "stop"], cwd=workspace, env=environment, timeout=60
    )
    deadline = asyncio.get_running_loop().time() + 30
    while True:
        if not discovery.exists():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("127.0.0.1", 8765))
                return
            except OSError:
                pass
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError("installed service did not remove discovery and release its port")
        await asyncio.sleep(0.05)


async def validate(wheels: tuple[Path, Path], output: Path, node: str) -> None:
    results = {}
    for kind, wheel in zip(("direct", "rebuilt"), wheels, strict=True):
        results[kind] = await validate_install(wheel, output / kind, node)
    revision = _run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = bool(_run_command(["git", "status", "--porcelain"]).stdout.strip())
    payload = {
        "marker": "INSTALLED_WEB_LIFECYCLE_OK",
        "installations": results,
        "source_root": str(ROOT),
        "source_head": revision,
        "source_dirty": dirty,
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("output must be outside the source repository")
    node = shutil.which("node")
    if node is None:
        parser.error("the Playwright controller requires Node; installed applications do not")
    output.mkdir(parents=True, exist_ok=False)
    wheels = build_distributions(output)
    asyncio.run(validate(wheels, output, node))


if __name__ == "__main__":
    main()
