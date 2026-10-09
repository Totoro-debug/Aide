from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_OFFLINE_PROGRAM = r"""
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
network_calls = []
event_loop = asyncio.new_event_loop()

def reject_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        network_calls.append(event)
        raise AssertionError("offline tokenization attempted network access")

sys.addaudithook(reject_network)
from aide.agent.context import tokenizer
from aide.agent.context.run_context import AgentRunContextSnapshot, ContextController
from aide.provider.model_router import ModelRouteStatus

assert Path(tokenizer.__file__).resolve().is_relative_to(installed)
counts = {}
for model in ("gpt-4o", "gpt-4", "text-davinci-003", "text-davinci-edit-001", "davinci", "gpt2", "gpt-oss-120b", "claude-sonnet-4", "unknown-model"):
    counts[model] = ContextController.estimate_request_tokens(
        [{"role": "system", "content": "SYSTEM 中文"}, {"role": "user", "content": "print('hello') <|endoftext|>"}],
        [{"name": "read_file", "parameters": {"type": "object"}}], model=model,
    )
    assert counts[model] > 0

class Router:
    def call_route_status(self, route, *, continuation):
        return ModelRouteStatus(requested_route=route, selected_route=route, provider_id="local", model="gpt-4o", context_window=10000, max_output=100, used_fallback=False)

async def reject_summary(*args):
    raise AssertionError("empty history must not summarize")

controller = ContextController(
    snapshot=AgentRunContextSnapshot(messages=(), metadata={}, last_compacted=0),
    provider=Router(), append_summary=reject_summary,
    now=lambda: datetime.now(timezone.utc), request_router=Router(), requested_route="chat",
    current_user={"role": "user", "content": "offline 首次请求"},
    project_messages=lambda history, user, increment, cursor, summary: [{"role": "system", "content": "SYSTEM"}, user, *increment],
)
prepared = event_loop.run_until_complete(controller.prepare(increment=(), latest_cycle_start=None, tools=(), continuation=None, continuation_revision=0, is_micro_compression_eligible=None))
event_loop.close()
assert len(prepared) == 2
assert not network_calls
print(json.dumps({"counts": counts, "network_calls": len(network_calls), "module": tokenizer.__file__}))
"""


def _run(arguments: list[str], *, cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_installed_wheel_and_sdist_rebuild_count_offline_without_a_cache(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    output = tmp_path / "distribution"
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(output),
        ],
        cwd=ROOT,
        environment=environment,
    )
    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    expected = {
        path.name: path.read_bytes()
        for path in (ROOT / "aide/agent/context/tokenizer_data").iterdir()
        if path.is_file()
    }
    prefix = "aide/agent/context/tokenizer_data/"
    with zipfile.ZipFile(wheel) as archive:
        assert {
            name.removeprefix(prefix) for name in archive.namelist() if name.startswith(prefix)
        } == set(expected)
        for name, contents in expected.items():
            assert archive.read(prefix + name) == contents
    extracted = tmp_path / "source"
    with tarfile.open(sdist) as archive:
        members = {
            member.name.split("/" + prefix, 1)[1]: member
            for member in archive.getmembers()
            if "/" + prefix in member.name and member.isfile()
        }
        assert set(members) == set(expected)
        for name, contents in expected.items():
            resource = archive.extractfile(members[name])
            assert resource is not None and resource.read() == contents
        archive.extractall(extracted, filter="data")
    rebuilt = tmp_path / "rebuilt"
    _run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(rebuilt)],
        cwd=next(extracted.iterdir()),
        environment=environment,
    )
    for number, artifact in enumerate((wheel, next(rebuilt.glob("*.whl")))):
        installed = tmp_path / f"installed-{number}"
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                str(installed),
                str(artifact),
            ],
            cwd=tmp_path,
            environment=environment,
        )
        cache = tmp_path / f"cache-{number}"
        environment["TIKTOKEN_CACHE_DIR"] = str(cache)
        environment["DATA_GYM_CACHE_DIR"] = str(cache)
        evidence = json.loads(
            _run(
                [sys.executable, "-I", "-c", _OFFLINE_PROGRAM, str(installed)],
                cwd=tmp_path,
                environment=environment,
            )
        )
        assert evidence["network_calls"] == 0
        assert not cache.exists()
