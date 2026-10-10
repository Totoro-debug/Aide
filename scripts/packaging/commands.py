"""Setuptools command customizations used by the PEP 517 backend."""

from __future__ import annotations

from pathlib import Path
from shutil import rmtree

from setuptools.command.build_py import build_py as _build_py  # type: ignore[import-untyped]


class build_py(_build_py):  # type: ignore[misc]  # Setuptools commands are not typed.
    """Prevent stale generated Web files from leaking into a new wheel."""

    def run(self) -> None:
        stale_assets = Path(self.build_lib) / "aide" / "web_assets"
        if stale_assets.is_dir():
            build_root = Path(__file__).resolve().parents[2] / "build"
            if not stale_assets.resolve().is_relative_to(build_root.resolve()):
                raise ValueError(
                    "Web asset cleanup requires a build directory inside project/build"
                )
            for candidate in (build_root, *stale_assets.parents, stale_assets):
                if candidate.is_symlink() or candidate.is_junction():
                    raise ValueError("Web asset cleanup refuses redirected build directories")
            rmtree(stale_assets)
        super().run()

