import json
import runpy
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
from setuptools import Distribution  # type: ignore[import-untyped]

from scripts.validate_web_assets import WebAssetError, build_manifest, validate_web_assets

ASSET_ROOT = Path(__file__).parents[1] / "omni" / "web_assets"


def test_packaged_web_assets_have_a_verified_manifest() -> None:
    manifest = validate_web_assets(ASSET_ROOT)

    assert manifest["schema_version"] == 1
    assert manifest["entry"] == "index.html"
    assert manifest["files"]
    assert all("sha256" in record and "bytes" in record for record in manifest["files"])


def test_manifest_is_deterministic_and_records_independent_file_hashes() -> None:
    manifest = build_manifest(ASSET_ROOT)

    assert manifest == json.loads(json.dumps(manifest, sort_keys=True))
    for record in manifest["files"]:
        path = ASSET_ROOT / str(record["path"])
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert path.stat().st_size == record["bytes"]


def test_missing_entry_asset_is_a_build_error(tmp_path: Path) -> None:
    root = tmp_path / "web_assets"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "index.html").write_text(
        '<script type="module" src="/assets/app.js"></script>\n'
        '<link rel="stylesheet" href="/assets/app.css">\n',
        encoding="utf-8",
    )
    (root / "favicon.svg").write_text("<svg />\n", encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "app.css").write_text("body {}\n", encoding="utf-8")

    with pytest.raises(WebAssetError, match=r"missing referenced asset.*app.js"):
        validate_web_assets(root)


def test_manifest_rejects_stale_file_hash(tmp_path: Path) -> None:
    root = tmp_path / "web_assets"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "index.html").write_text(
        '<script type="module" src="/assets/app.js"></script>\n'
        '<link rel="stylesheet" href="/assets/app.css">\n',
        encoding="utf-8",
    )
    (root / "favicon.svg").write_text("<svg />\n", encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "app.js").write_text("console.log('ok');\n", encoding="utf-8")
    (root / "assets" / "app.css").write_text("body {}\n", encoding="utf-8")
    manifest = build_manifest(root)
    manifest["files"][0]["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WebAssetError, match="sha256 mismatch"):
        validate_web_assets(root)


def test_build_cleanup_rejects_source_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    import setuptools

    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    namespace = runpy.run_path(str(ASSET_ROOT.parents[1] / "setup.py"))
    command = namespace["build_py"](Distribution())
    command.build_lib = str(ASSET_ROOT.parents[1])
    before = {path: path.read_bytes() for path in ASSET_ROOT.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="inside project/build"):
        command.run()

    assert all(path.read_bytes() == content for path, content in before.items())


def test_windows_git_checkout_preserves_manifest_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(
        ASSET_ROOT, source / "omni/web_assets", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(ASSET_ROOT.parents[1] / ".gitattributes", source / ".gitattributes")

    def git(*arguments: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=true",
                "-c",
                "commit.gpgsign=false",
                "-c",
                f"core.hooksPath={tmp_path / 'no-hooks'}",
                *arguments,
            ],
            cwd=source,
            check=True,
            capture_output=True,
        )

    git("init", "--quiet")
    git("add", ".")
    git(
        "-c",
        "user.name=Asset fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "asset fixture",
    )
    checkout = tmp_path / "checkout"
    git("clone", "--quiet", "-c", "core.autocrlf=true", str(source), str(checkout))

    assert validate_web_assets(checkout / "omni/web_assets") == validate_web_assets(ASSET_ROOT)
