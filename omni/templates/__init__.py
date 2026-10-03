"""Load versioned Markdown templates bundled with MyClaw."""

from functools import cache
from importlib.resources import files


@cache
def load_template(name: str) -> str:
    """Return one package-local Markdown template without altering its content."""
    if not name or not name.endswith(".md") or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("template name must identify one package-local Markdown file")
    return files("omni.templates").joinpath(name).read_text(encoding="utf-8")


def render_template(name: str, /, **values: object) -> str:
    """Render a text template while ignoring its source-file terminator."""
    return load_template(name).removesuffix("\n").format_map(values)
