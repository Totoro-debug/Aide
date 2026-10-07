"""Setuptools backend with a fail-closed Web asset packaging gate."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from setuptools import build_meta as _setuptools  # type: ignore[import-untyped]

from scripts.validate_web_assets import validate_web_assets

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "aide" / "web_assets"


def _validate_assets() -> None:
    validate_web_assets(ASSET_ROOT)


def get_requires_for_build_sdist(config_settings: dict[str, object] | None = None) -> list[str]:
    return cast(list[str], _setuptools.get_requires_for_build_sdist(config_settings))


def get_requires_for_build_wheel(config_settings: dict[str, object] | None = None) -> list[str]:
    return cast(list[str], _setuptools.get_requires_for_build_wheel(config_settings))


def get_requires_for_build_editable(config_settings: dict[str, object] | None = None) -> list[str]:
    return cast(list[str], _setuptools.get_requires_for_build_editable(config_settings))


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, object] | None = None,
) -> str:
    _validate_assets()
    return cast(
        str,
        _setuptools.prepare_metadata_for_build_wheel(metadata_directory, config_settings),
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, object] | None = None,
) -> str:
    _validate_assets()
    return cast(
        str,
        _setuptools.prepare_metadata_for_build_editable(metadata_directory, config_settings),
    )


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, object] | None = None,
) -> str:
    _validate_assets()
    return cast(str, _setuptools.build_sdist(sdist_directory, config_settings))


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _validate_assets()
    return cast(
        str,
        _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory),
    )


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _validate_assets()
    return cast(
        str,
        _setuptools.build_editable(wheel_directory, config_settings, metadata_directory),
    )
