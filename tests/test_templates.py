from importlib.resources import files
from string import Formatter

import pytest

from omni.templates import load_template, render_template

TEMPLATE_NAMES = {
    "blackboard-system-prompt.md",
    "conversation-compaction-system-prompt.md",
    "conversation-summary-system-prompt.md",
    "default-config.md",
    "foreground-chat-system-prompt.md",
    "long-term-memory.md",
    "memory-task-prompt.md",
    "mcp-keyword-system-prompt.md",
    "session-title-prompt.md",
    "skill-always-load.md",
    "skill-catalog.md",
}
ASSEMBLY_ONLY_TEMPLATE_NAMES = {
    "blackboard.md",
    "conversation-summary-input.md",
    "current-user-input.md",
    "interrupted-assistant-content.md",
    "memory-task-input.md",
    "runtime-context.md",
    "user-input.md",
}
TEMPLATE_VALUES = {
    "workspace": r"D:\workspace",
    "agent_home": r"D:\agent-home",
    "runtime": "Windows AMD64, Python 3.12.13",
    "long_term_memory": "memory content",
    "long_term_path": r"D:\workspace\memory.md",
    "entries": '{"name":"planner"}',
    "User input": "Current input",
    "Last Task": "null",
    "Latest assistant content": "Latest assistant content",
}


def _template_fields(source: str) -> tuple[str, ...]:
    fields: list[str] = []
    for _, field_name, _, _ in Formatter().parse(source):
        if field_name is not None:
            fields.append(field_name)
    return tuple(fields)


def test_all_versioned_templates_are_package_resources() -> None:
    root = files("omni.templates")
    names = {
        resource.name
        for resource in root.iterdir()
        if resource.is_file() and resource.name != "__init__.py"
    }

    assert names == TEMPLATE_NAMES


def test_assembly_only_templates_are_not_packaged() -> None:
    root = files("omni.templates")

    assert [
        name for name in sorted(ASSEMBLY_ONLY_TEMPLATE_NAMES) if root.joinpath(name).is_file()
    ] == []


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "legacy.txt", "nested/name.md", r"nested\name.md"],
)
def test_template_loader_rejects_non_markdown_or_nonlocal_names(name: str) -> None:
    with pytest.raises(ValueError, match="package-local"):
        load_template(name)


def test_generated_file_templates_preserve_their_terminal_newline() -> None:
    assert load_template("default-config.md").endswith("timeout = 120\n")
    assert load_template("long-term-memory.md").endswith("## Lesson\n")


def test_prompt_renderer_removes_only_the_source_file_terminator() -> None:
    source = load_template("session-title-prompt.md")
    rendered = render_template("session-title-prompt.md")

    assert source.endswith("\n")
    assert rendered == source.removesuffix("\n")
    assert not rendered.endswith("\n")


@pytest.mark.parametrize("name", sorted(TEMPLATE_NAMES))
def test_retained_templates_render_all_declared_values_without_placeholders(name: str) -> None:
    source = load_template(name)
    rendered = render_template(name, **TEMPLATE_VALUES)

    for field_name in _template_fields(source):
        assert f"{{{field_name}}}" not in rendered
        assert TEMPLATE_VALUES[field_name] in rendered
