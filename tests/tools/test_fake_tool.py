import pytest

from tests.fixtures.tool import FakeTool, FakeToolCall


@pytest.mark.asyncio
async def test_fake_tool_returns_a_scripted_result_and_records_the_call() -> None:
    tool = FakeTool(
        name="read_file",
        description="Read a UTF-8 file.",
        required=("path",),
        outcomes=["file contents"],
    )
    arguments = {"path": "CONTEXT.md"}

    result = await tool.execute(path="CONTEXT.md")

    assert result == "file contents"
    assert tool.calls == [FakeToolCall(arguments=arguments)]


@pytest.mark.asyncio
async def test_fake_tool_raises_a_scripted_failure_after_recording_the_call() -> None:
    failure = OSError("disk unavailable")
    tool = FakeTool(
        name="read_file",
        description="Read a UTF-8 file.",
        required=("path",),
        outcomes=[failure],
    )

    with pytest.raises(OSError, match="disk unavailable"):
        await tool.execute(path="missing.txt")

    assert tool.calls == [FakeToolCall(arguments={"path": "missing.txt"})]
