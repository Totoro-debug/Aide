"""Validate and describe the Web assets shipped in the Python package."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from typing import TypedDict, cast
from urllib.parse import urlsplit


class AssetFile(TypedDict):
    path: str
    sha256: str
    bytes: int


class AssetManifest(TypedDict):
    schema_version: int
    entry: str
    files: list[AssetFile]


class WebAssetError(ValueError):
    """Raised when a Web asset tree cannot be safely published."""


class _EntryReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self.script_count = 0
        self.stylesheet_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and (source := attributes.get("src")) is not None:
            self.references.append(source)
            self.script_count += 1
        elif tag == "link" and (href := attributes.get("href")) is not None:
            self.references.append(href)
            if "stylesheet" in (attributes.get("rel") or "").split():
                self.stylesheet_count += 1


def build_manifest(root: Path) -> AssetManifest:
    """Validate an asset tree and return its deterministic manifest."""
    if root.is_symlink() or root.is_junction():
        raise WebAssetError("redirected Web asset root is not allowed")
    root = root.resolve()
    _require_asset_root(root)
    referenced = _validate_entry(root)
    records = [_file_record(root, relative) for relative in _asset_paths(root)]
    record_paths = {record["path"] for record in records}
    missing = sorted(referenced - record_paths)
    if missing:
        raise WebAssetError(f"missing referenced asset(s): {', '.join(missing)}")
    if not any(path.startswith("assets/") and path.endswith(".js") for path in record_paths):
        raise WebAssetError("required JavaScript asset is missing")
    if not any(path.startswith("assets/") and path.endswith(".css") for path in record_paths):
        raise WebAssetError("required CSS asset is missing")
    return {
        "schema_version": 1,
        "entry": "index.html",
        "files": records,
    }


def validate_web_assets(root: Path) -> AssetManifest:
    """Validate the tree and its on-disk manifest against current file bytes."""
    expected = build_manifest(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise WebAssetError("required asset manifest.json is missing")
    try:
        raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WebAssetError("asset manifest.json is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise WebAssetError("asset manifest.json must contain an object")
    actual = cast(dict[str, object], raw)
    if set(actual) != {"schema_version", "entry", "files"}:
        raise WebAssetError("asset manifest.json has an unsupported shape")
    if (
        type(actual.get("schema_version")) is not int
        or actual.get("schema_version") != 1
        or actual.get("entry") != "index.html"
    ):
        raise WebAssetError("asset manifest.json has an unsupported version or entry")
    actual_files = _manifest_files(actual.get("files"))
    expected_by_path = {record["path"]: record for record in expected["files"]}
    actual_by_path = {record["path"]: record for record in actual_files}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    if missing:
        raise WebAssetError(f"asset manifest.json is missing: {', '.join(missing)}")
    unexpected = sorted(set(actual_by_path) - set(expected_by_path))
    if unexpected:
        raise WebAssetError(f"asset manifest.json lists unknown files: {', '.join(unexpected)}")
    for path, expected_record in expected_by_path.items():
        actual_record = actual_by_path[path]
        if actual_record["sha256"] != expected_record["sha256"]:
            raise WebAssetError(f"sha256 mismatch for {path}")
        if actual_record["bytes"] != expected_record["bytes"]:
            raise WebAssetError(f"byte-size mismatch for {path}")
    if actual_files != expected["files"]:
        raise WebAssetError("asset manifest.json is not deterministic")
    return expected


def write_manifest(root: Path) -> AssetManifest:
    """Write a canonical manifest after validating the asset tree."""
    manifest = build_manifest(root)
    destination = root / "manifest.json"
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return validate_web_assets(root)


def _require_asset_root(root: Path) -> None:
    if not root.is_dir():
        raise WebAssetError(f"Web asset directory does not exist: {root}")
    for required in ("__init__.py", "index.html", "favicon.svg", "assets"):
        candidate = root / required
        if not candidate.exists():
            raise WebAssetError(f"required Web asset is missing: {required}")
    if not (root / "__init__.py").is_file():
        raise WebAssetError("required Web asset is not a file: __init__.py")
    if not (root / "assets").is_dir():
        raise WebAssetError("required Web asset directory is not a directory: assets")


def _validate_entry(root: Path) -> set[str]:
    parser = _EntryReferenceParser()
    try:
        parser.feed((root / "index.html").read_text(encoding="utf-8"))
        parser.close()
    except (OSError, UnicodeError, ValueError) as error:
        raise WebAssetError("index.html could not be parsed as UTF-8 HTML") from error
    if parser.script_count == 0:
        raise WebAssetError("index.html does not reference a JavaScript entry")
    if parser.stylesheet_count == 0:
        raise WebAssetError("index.html does not reference a stylesheet")
    references = {_reference_path(reference) for reference in parser.references}
    references.update({"index.html", "favicon.svg"})
    return references


def _reference_path(reference: str) -> str:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or "\\" in parsed.path:
        raise WebAssetError(f"index.html contains a non-local asset reference: {reference}")
    value = parsed.path.lstrip("/")
    parts = value.split("/")
    if not value or any(part in {"", ".", ".."} for part in parts) or ":" in value:
        raise WebAssetError(f"index.html contains an unsafe asset reference: {reference}")
    return "/".join(parts)


def _asset_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or candidate.is_junction():
            raise WebAssetError(f"redirected Web asset is not allowed: {candidate.name}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if "__pycache__" in relative.parts or candidate.suffix == ".pyc":
            continue
        path = relative.as_posix()
        if path in {"__init__.py", "manifest.json"}:
            continue
        _reference_path(path)
        paths.append(path)
    return sorted(paths)


def _file_record(root: Path, relative: str) -> AssetFile:
    content = (root / Path(*relative.split("/"))).read_bytes()
    return {
        "path": relative,
        "sha256": sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _manifest_files(value: object) -> list[AssetFile]:
    if not isinstance(value, list):
        raise WebAssetError("asset manifest.json files must be an array")
    records: list[AssetFile] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise WebAssetError("asset manifest.json contains an invalid file record")
        record = cast(dict[str, object], item)
        if set(record) != {"path", "sha256", "bytes"}:
            raise WebAssetError("asset manifest.json contains an unsupported file record")
        path = record.get("path")
        digest = record.get("sha256")
        size = record.get("bytes")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise WebAssetError("asset manifest.json contains an invalid file record")
        if _reference_path(path) != path:
            raise WebAssetError(f"asset manifest.json contains an unsafe path: {path}")
        if path in seen:
            raise WebAssetError(f"asset manifest.json contains duplicate file: {path}")
        seen.add(path)
        records.append({"path": path, "sha256": digest, "bytes": size})
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write a canonical manifest before validating the tree",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = (
            write_manifest(arguments.root)
            if arguments.write_manifest
            else validate_web_assets(arguments.root)
        )
    except (OSError, WebAssetError) as error:
        print(f"Web asset validation failed: {error}")
        return 1
    print(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
